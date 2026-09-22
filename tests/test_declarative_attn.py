import pytest
import torch

from declarative_attention import (
    DeclarativeAttention,
    StateMachineViolationError,
    derive_declarative_mask,
    extract_chunk_spans,
    format_declarative_prompt,
    parse_chunk_ids,
    segment_context
)

# end to end test

def test_declarative_attention_e2e():
    prompt = """
    Question: Where was the meeting held?

    <chunk id="1">
    Acme Corp was founded in 2003 in Seattle.
    </chunk>
    <chunk id="2">
    The annual meeting was held in Chicago in 2011.
    </chunk>
    """

    chunk_spans = extract_chunk_spans(prompt)

    da = DeclarativeAttention(chunk_spans = chunk_spans)
    assert da.mode == 'global'

    # 1. focus on chunk 2
    da.step('<focus magic_chunks="2">')
    assert da.mode == 'focus' and da.active_chunks == {2}

    mask = da.get_mask(total_len = 8192)
    kept_blocks = da.get_kept_blocks(total_len = 8192)
    assert len(kept_blocks) > 0

    # 2. local mode: all chunks masked, only instructions and response attended
    da.step('<local>')
    assert da.mode == 'local' and da.active_chunks == set()

    # 3. closing tags unwinds stack back to previous state
    da.step('</local>')
    assert da.mode == 'focus' and da.active_chunks == {2}

    da.step('</focus>')
    assert da.mode == 'global'

# modes

def test_modes_switch_on_tags():
    da = DeclarativeAttention({1: (16, 32), 2: (32, 48)})
    assert da.is_global and da.get_mask(64).all()

    da.step('<focus magic_chunks="1">')
    assert da.is_focus and da.active_chunks == {1}
    assert not da.get_mask(64)[32:48].any()

    da.step('</focus>')
    assert da.is_global and da.get_mask(64).all()

    da.step('<local>')
    assert da.is_local and not da.get_mask(64)[16:48].any()

    da.step('</local>')
    assert da.is_global

def test_focus_parses_attribute_variants():
    da = DeclarativeAttention({1: (0, 10), 2: (10, 20), 3: (20, 30)})

    da.step('<focus magic_chunks="1, 3">')
    assert da.active_chunks == {1, 3}

    da.step('<focus ids="1, 2"/>')
    assert da.active_chunks == {1, 2}

    da.step('<focus id="2">')
    assert da.active_chunks == {2}

    da.step('<focus 3>')
    assert da.active_chunks == {3}

    # unknown chunk ids are filtered out
    da.step('<focus magic_chunks="99">')
    assert da.active_chunks == set()

def test_focus_without_chunk_ids_is_local():
    da = DeclarativeAttention([(0, 8), (8, 16)])

    da.step('<focus>')
    assert da.is_local and da.active_chunks == set()

def test_self_closing_tags_apply_until_closed():
    da = DeclarativeAttention([(0, 8), (8, 16)])

    da.step('<local/>')
    assert da.is_local

    da.step('</local>')
    assert da.is_global

def test_unknown_tags_are_ignored():
    da = DeclarativeAttention([(0, 8)])

    da.step('<b>bold</b> and <i>italic</i>')
    assert da.is_global

def test_empty_chunk_spans_stay_global():
    da = DeclarativeAttention({})

    assert da.mode == 'global'
    assert da.get_mask(8).all()

# streaming

def test_tags_split_across_steps_are_buffered():
    da = DeclarativeAttention([(0, 8), (8, 16)])

    da.step('<focus magic_chunks=')
    assert da.is_global, 'no transition until the tag closes on ">"'

    da.step('"1">')
    assert da.is_focus and da.active_chunks == {1}

    da.step('</fo')
    assert da.is_focus

    da.step('cus>')
    assert da.is_global

def test_step_accepts_lists_and_multi_element_tensors():
    vocab = {1: '<focus magic_chunks="2">', 2: '</focus>'}
    da = DeclarativeAttention([(0, 8), (8, 16)], tokenizer_decode = lambda t: vocab.get(t, ''))

    da.step([1])
    assert da.is_focus and da.active_chunks == {2}

    da.step(torch.tensor([2, 1]))
    assert da.is_focus and da.active_chunks == {2}

    # nested structures are flattened
    da.step(torch.tensor([[2, 1], [2, 1]]))
    assert da.is_focus and da.active_chunks == {2}

def test_stray_less_than_does_not_break_parsing():
    da = DeclarativeAttention([(0, 8)])

    da.step('if x < 10 and y < 20: pass ')
    da.step(' ' * 600)
    da.step('<local>')
    assert da.is_local

# custom tags

def test_custom_tags_can_modify_visibility_and_sampling():
    da = DeclarativeAttention([(0, 8), (8, 16), (16, 24)])

    @da.on('compare')
    def compare(da, attrs, is_closing):
        da.active_chunks = parse_chunk_ids(attrs['chunks']) if not is_closing else da.all_chunks

    @da.on('verbatim')
    def verbatim(da, attrs, is_closing):
        da.active_chunks = {int(attrs['chunk'])} if not is_closing else da.all_chunks
        da.state['temperature'] = 0. if not is_closing else 1.

    da.step('<compare chunks="1, 3">')
    assert da.active_chunks == {1, 3}

    da.step('</compare>')
    assert da.is_global

    da.step('<verbatim chunk="2">')
    assert da.active_chunks == {2} and da.state['temperature'] == 0.

    da.step('</verbatim>')
    assert da.is_global and da.state['temperature'] == 1.

def test_custom_tags_auto_restore_state_and_chunks():
    da = DeclarativeAttention([(0, 8), (8, 16), (16, 24)])

    @da.on('compare')
    def compare(da, attrs):
        da.active_chunks = parse_chunk_ids(attrs['chunks'])

    @da.on('verbatim')
    def verbatim(da, attrs):
        da.active_chunks = {int(attrs['chunk'])}
        da.state['temperature'] = 0.

    da.step('<compare chunks="1, 3">')
    assert da.active_chunks == {1, 3}

    # nested verbatim
    da.step('<verbatim chunk="2">')
    assert da.active_chunks == {2}
    assert da.state['temperature'] == 0.

    # closing verbatim automatically restores to compare's active chunks and removes temperature
    da.step('</verbatim>')
    assert da.active_chunks == {1, 3}
    assert 'temperature' not in da.state

    # closing compare restores to global
    da.step('</compare>')
    assert da.is_global

def test_parse_chunk_ids():
    assert parse_chunk_ids('1,3') == {1, 3}
    assert parse_chunk_ids('1 3') == {1, 3}
    assert parse_chunk_ids(2) == {2}
    assert parse_chunk_ids([1, 2]) == {1, 2}
    assert parse_chunk_ids('') == set()

# block aligned masks

def test_block_mask_keeps_scaffold_and_active_chunks():
    # sink 0..16, chunks 1..4 of 32 tokens, question 144..170
    da = DeclarativeAttention([(16, 48), (48, 80), (80, 112), (112, 144)])

    da.step('<focus magic_chunks="2">')
    assert da.get_kept_blocks(170) == [0, 3, 4, 9, 10]

    da.step('</focus><local>')
    assert da.get_kept_blocks(170) == [0, 9, 10]

    da.step('</local>')
    assert da.get_kept_blocks(170) == list(range(11))

def test_block_mask_rounds_kept_spans_outward():
    da = DeclarativeAttention({1: (20, 48)}, block_size = 16)
    da.step('<local>')

    # block 1 is only partly masked, so it stays kept
    assert da.get_kept_blocks(64) == [0, 1, 3]

def test_mask_clips_spans_past_total_len():
    da = DeclarativeAttention({1: (8, 32)})
    da.step('<local>')

    mask = da.get_mask(16)
    assert mask.shape == (16,)
    assert mask[:8].all() and not mask[8:].any()

def test_masks_for_empty_sequence():
    da = DeclarativeAttention({1: (0, 8)})

    assert da.get_mask(0).shape == (0,)
    assert da.get_kept_blocks(0) == []

# 2d mask for training

def test_derive_declarative_mask():
    chunk_spans = [(4, 8), (8, 12)]
    text = 'ABCD11112222<focus magic_chunks="1">xyz</focus><local>abc</local>'
    tokens = torch.tensor([ord(c) for c in text], dtype = torch.long)

    mask = derive_declarative_mask(tokens, chunk_spans)
    assert mask.shape == (len(text), len(text))

    focus_pos, local_pos = text.index('xyz'), text.index('abc')

    assert mask[focus_pos, 4:8].all() and not mask[focus_pos, 8:12].any()
    assert not mask[local_pos, 4:12].any()
    assert mask[local_pos, :4].all()

def test_derive_declarative_mask_is_causal():
    tokens = torch.randint(1, 10, (2, 16))
    chunk_spans = [[(2, 4), (4, 6)], [(4, 6), (6, 8)]]

    mask = derive_declarative_mask(tokens, chunk_spans)
    assert mask.shape == (2, 1, 16, 16)
    assert not mask.triu(diagonal = 1).any()

def test_derive_declarative_mask_ignores_prompt_tags_with_prompt_len():
    text = 'AB11<local>xyz'
    tokens = torch.tensor([ord(c) for c in text])
    chunk_spans = [(2, 4)]
    response_pos = text.index('x')

    assert not derive_declarative_mask(tokens, chunk_spans)[response_pos, 2:4].any()
    assert derive_declarative_mask(tokens, chunk_spans, prompt_len = response_pos)[response_pos, 2:4].all()

def test_derive_declarative_mask_accepts_list_of_list_spans():
    tokens = torch.randint(1, 10, (16,))
    mask = derive_declarative_mask(tokens, [[2, 4], [4, 6]])

    assert mask.shape == (16, 16)

def test_derive_declarative_mask_accepts_batched_list_of_list_spans():
    tokens = torch.randint(1, 10, (2, 16))
    mask = derive_declarative_mask(tokens, [[(2, 4), (4, 6)], [(4, 6), (6, 8)]])

    assert mask.shape == (2, 1, 16, 16)

# context segmentation

def test_segment_context_is_lossless():
    text = 'Para one.\n\nPara two. Another sentence.\nLast line.'
    segments = segment_context(text, tokenizer_encode = list, target_tokens = 10, max_tokens = 20)

    assert ''.join(segments) == text
    assert len(segments) > 1

def test_segment_context_splits_units_over_the_cap():
    text = 'one two three four five six seven eight nine ten'
    segments = segment_context(text, tokenizer_encode = lambda s: s.split(), target_tokens = 3, max_tokens = 5)

    assert ''.join(segments) == text
    assert all(len(s.split()) <= 5 for s in segments)

def test_segment_context_keeps_whitespace_free_runs_atomic():
    blob = 'x' * 100
    assert segment_context(blob, tokenizer_encode = list, target_tokens = 10, max_tokens = 20) == [blob]

def test_segment_context_counts_tokens_not_characters():
    # byte fallback tokenizers can spend many tokens per character
    text = 'a b c d e f g h i j'
    encode = lambda s: [1] * (len(s) * 3)

    segments = segment_context(text, tokenizer_encode = encode, target_tokens = 15, max_tokens = 20)
    assert ''.join(segments) == text
    assert len(segments) > 1

def test_segment_context_short_text_is_a_single_segment():
    text = 'Short context.'
    assert segment_context(text, tokenizer_encode = list, target_tokens = 2048, max_tokens = 2560) == [text]

def test_segment_context_empty():
    assert segment_context('') == ['<empty_context>']
    assert segment_context('   ') == ['<empty_context>']

# prompt construction

def test_format_declarative_prompt_char_spans():
    chunks = ['Acme Corp was founded in 2003.', 'Acme went public on the NYSE in 2011.']
    prompt, spans = format_declarative_prompt('How long between founding and IPO?', chunks)

    assert 'Magic Chunk 1:' in prompt and 'Magic Chunk 2:' in prompt
    assert prompt[spans[1][0]:spans[1][1]] == chunks[0]
    assert prompt[spans[2][0]:spans[2][1]] == chunks[1]

def test_format_declarative_prompt_token_spans():
    chunks = ['Fact one', 'Fact two']
    encode = lambda s: [ord(c) for c in s]
    decode = lambda ids: ''.join(chr(i) for i in ids)

    tokens, spans = format_declarative_prompt('What is the answer?', chunks, tokenizer_encode = encode)

    assert isinstance(tokens, list)
    assert decode(tokens[spans[1][0]:spans[1][1]]) == chunks[0]
    assert decode(tokens[spans[2][0]:spans[2][1]]) == chunks[1]

def test_format_declarative_prompt_accepts_tensor_tokenizer():
    encode = lambda s: torch.tensor([ord(c) for c in s])
    tokens, spans = format_declarative_prompt('Query?', ['Alpha', 'Beta'], tokenizer_encode = encode)

    assert isinstance(tokens, list)
    assert len(spans) == 2

def test_format_declarative_prompt_segments_raw_context():
    context = 'First fact. ' * 50 + 'Second fact. ' * 50
    prompt, spans = format_declarative_prompt('What?', context, tokenizer_encode = lambda s: s.split())

    assert len(spans) == 1

def test_format_declarative_prompt_no_chunks():
    prompt, spans = format_declarative_prompt('Question?', [], tokenizer_encode = list)

    assert spans == {}
    assert 'Magic Chunk' not in ''.join(prompt)

def test_machine_carries_its_protocol_prompt():
    class ResearchAttention(DeclarativeAttention):
        instructions = 'Use <compare chunks="K,M"> to view several magic chunks.'

    prompt, spans = ResearchAttention.format_prompt('What?', ['a', 'b'])

    assert 'Use <compare chunks="K,M">' in prompt
    assert len(spans) == 2

# end to end

def test_prompt_construction_to_training_mask():
    question = 'When did Acme go public?'
    chunks = ['Acme was founded in 2003.', 'Acme went public on the NYSE in 2011.']
    encode = lambda text: [ord(c) for c in text]

    prompt, chunk_spans = format_declarative_prompt(question, chunks, tokenizer_encode = encode)

    response = '<focus magic_chunks="1">Founded in 2003.</focus><local>The IPO was in 2011.</local>'
    tokens = torch.tensor(prompt + encode(response))

    mask = derive_declarative_mask(tokens, chunk_spans, prompt_len = len(prompt))

    focus_pos = len(prompt) + response.index('Founded')
    local_pos = len(prompt) + response.index('The IPO')
    chunk_1, chunk_2 = chunk_spans[1], chunk_spans[2]

    assert mask[focus_pos, chunk_1[0]:chunk_1[1]].all()
    assert not mask[focus_pos, chunk_2[0]:chunk_2[1]].any()
    assert not mask[local_pos, chunk_1[0]:chunk_1[1]].any()
    assert not mask[local_pos, chunk_2[0]:chunk_2[1]].any()

# extract chunk spans

def test_extract_chunk_spans_paper_format():
    text = """You are a helpful assistant.

Magic Chunk 1:
Acme Corp was founded in 2003.

Magic Chunk 2:
Acme went public in 2011.

Question:
When was Acme founded?
"""
    spans = extract_chunk_spans(text)
    assert spans == {1: (45, 75), 2: (92, 117)}
    assert text[spans[1][0]:spans[1][1]] == 'Acme Corp was founded in 2003.'
    assert text[spans[2][0]:spans[2][1]] == 'Acme went public in 2011.'

def test_extract_chunk_spans_xml_tags():
    text = """
<chunk id="1">
First chunk content.
</chunk>
<magic_chunk id="2">
Second chunk content.
</magic_chunk>
"""
    spans = extract_chunk_spans(text)
    assert text[spans[1][0]:spans[1][1]] == 'First chunk content.'
    assert text[spans[2][0]:spans[2][1]] == 'Second chunk content.'

def test_extract_chunk_spans_with_tokenizer():
    text = "Magic Chunk 1:\nParis is nice.\n\nMagic Chunk 2:\nRome is warm.\n\nQuestion: Where?"
    encode = lambda s: s.split()
    tokens = encode(text)

    spans = extract_chunk_spans(text, tokenizer_encode = encode)
    assert tokens[spans[1][0]:spans[1][1]] == ['Paris', 'is', 'nice.']
    assert tokens[spans[2][0]:spans[2][1]] == ['Rome', 'is', 'warm.']

def test_extract_chunk_spans_custom_tag():
    text = "<document id=\"4\">Special info.</document>"
    spans = extract_chunk_spans(text, tag = 'document')

    assert spans == {4: (17, 30)}
    assert text[spans[4][0]:spans[4][1]] == 'Special info.'

def test_extract_chunk_spans_batched():
    texts = [
        '<chunk id="1">Alpha</chunk><chunk id="2">Beta</chunk>',
        '<chunk id="1">Gamma</chunk><chunk id="2">Delta</chunk>'
    ]
    spans = extract_chunk_spans(texts)
    assert len(spans) == 2
    assert texts[0][spans[0][1][0]:spans[0][1][1]] == 'Alpha'
    assert texts[1][spans[1][2][0]:spans[1][2][1]] == 'Delta'

def test_extract_chunk_spans_single_vs_batched_parity():
    text1 = '<chunk id="1">Hello</chunk><chunk id="2">World</chunk>'
    text2 = '<chunk id="1">Foo</chunk><chunk id="2">Bar</chunk>'

    spans1 = extract_chunk_spans(text1)
    spans2 = extract_chunk_spans(text2)

    batched = extract_chunk_spans([text1, text2])
    assert batched == [spans1, spans2]

    encode = lambda s: s.split()
    tok_spans1 = extract_chunk_spans(text1, tokenizer_encode = encode)
    tok_spans2 = extract_chunk_spans(text2, tokenizer_encode = encode)

    batched_tok = extract_chunk_spans([text1, text2], tokenizer_encode = encode)
    assert batched_tok == [tok_spans1, tok_spans2]

def test_derive_declarative_mask_single_vs_batched_parity():
    spans1 = [(2, 4), (4, 6)]
    spans2 = [(1, 3), (3, 5)]

    seq1 = torch.tensor([ord(c) for c in 'AB1122<local>XY'])
    seq2 = torch.tensor([ord(c) for c in 'A1122<focus 1>Z'])

    mask1 = derive_declarative_mask(seq1, spans1)
    mask2 = derive_declarative_mask(seq2, spans2)

    batched_seq = torch.stack([seq1, seq2])
    batched_mask = derive_declarative_mask(batched_seq, [spans1, spans2])

    assert batched_mask.shape == (2, 1, len(seq1), len(seq1))
    assert torch.equal(batched_mask[0, 0], mask1)
    assert torch.equal(batched_mask[1, 0], mask2)

# strict state machine violation detection

def test_state_machine_strict_detects_unclosed_tag_violation():
    da = DeclarativeAttention({1: (0, 10), 2: (10, 20)}, strict = True)

    da.step('<focus magic_chunks="1">')
    assert da.is_focus

    # opening <local> while <focus> is unclosed raises StateMachineViolationError
    with pytest.raises(StateMachineViolationError, match = "tag <local> opened before <focus> was closed"):
        da.step('<local>')

def test_state_machine_strict_false_allows_continuing():
    da = DeclarativeAttention({1: (0, 10), 2: (10, 20)}, strict = False)

    da.step('<focus magic_chunks="1">')
    assert da.is_focus

    # default allows continuing without error
    da.step('<local>')
    assert da.is_local

def test_state_machine_strict_unmatched_closing_tag():
    da = DeclarativeAttention({1: (0, 10)}, strict = True)

    with pytest.raises(StateMachineViolationError, match = "closing tag </local> detected but no tags are open"):
        da.step('</local>')

def test_state_machine_strict_mismatched_closing_tag():
    da = DeclarativeAttention({1: (0, 10)}, strict = True)

    da.step('<focus magic_chunks="1">')

    with pytest.raises(StateMachineViolationError, match = "closing tag </local> does not match open tag <focus>"):
        da.step('</local>')

# multi bare attributes

def test_focus_parses_multiple_bare_numbers():
    da = DeclarativeAttention({1: (0, 10), 2: (10, 20), 3: (20, 30)})

    da.step('<focus 1, 3>')
    assert da.active_chunks == {1, 3}

    da.step('<focus 1 2>')
    assert da.active_chunks == {1, 2}

# stack unwind

def test_stack_unwind_discards_enclosed_child_tags():
    da = DeclarativeAttention({1: (0, 10), 2: (10, 20), 3: (20, 30)})

    @da.on('outer')
    def handle_outer(m, attrs):
        m.active_chunks = {1}

    @da.on('inner')
    def handle_inner(m, attrs):
        m.active_chunks = {2}

    da.step('<outer>')
    assert da.active_chunks == {1}
    assert len(da.stack) == 1

    da.step('<inner>')
    assert da.active_chunks == {2}
    assert len(da.stack) == 2

    # closing outer directly must unwind both inner and outer
    da.step('</outer>')
    assert da.is_global
    assert len(da.stack) == 0

# reset and callable alias

def test_declarative_attention_reset():
    da = DeclarativeAttention({1: (0, 10), 2: (10, 20)})
    da.step('<focus 1>')
    assert da.is_focus

    da.reset()
    assert da.is_global
    assert da.active_chunks == {1, 2}
    assert len(da.stack) == 0

def test_declarative_attention_callable_alias():
    da = DeclarativeAttention({1: (0, 10)})
    assert torch.equal(da(32), da.get_mask(32))

def test_parse_chunk_ids_nested_and_compound():
    assert parse_chunk_ids([1, '2, 3', {4}]) == {1, 2, 3, 4}
    assert parse_chunk_ids(None) == set()
    assert parse_chunk_ids('') == set()
