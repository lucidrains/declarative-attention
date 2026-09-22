import pytest
import torch
from x_transformers import TransformerWrapper, Decoder

from declarative_attention import (
    DeclarativeAttention,
    DeclarativeAttentionWrapper,
    StateMachineViolationError,
    format_declarative_prompt,
    parse_chunk_ids
)

def decode_fn(token):
    return chr(token) if 0 <= token < 128 else ''

def make_wrapper(dim = 64, depth = 2, heads = 4, max_seq_len = 128):
    net = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = max_seq_len,
        attn_layers = Decoder(dim = dim, depth = depth, heads = heads)
    )
    return DeclarativeAttentionWrapper(net, tokenizer_decode = decode_fn)

def test_training_and_generation():
    wrapper = make_wrapper()
    chunk_spans = [(2, 4), (4, 6)]

    # 1. training forward and backward
    loss = wrapper(torch.randint(1, 10, (2, 16)), chunk_spans = chunk_spans)
    assert loss.ndim == 0 and not torch.isnan(loss)
    loss.backward()

    # 2. generation
    out = wrapper.generate(torch.tensor([[1, 1, 2, 2, 3, 3]]), seq_len = 8, chunk_spans = chunk_spans)
    assert out.shape == (1, 8)

def test_generation_streams_tag_pieces_to_the_machine():
    wrapper = make_wrapper()
    vocab = {10: '<lo', 11: 'cal', 12: '>'}
    script = iter([10, 11, 12])

    def scripted_logits(logits):
        logits = torch.full_like(logits, -1e9)
        logits[:, next(script, 12)] = 1e9
        return logits

    da = DeclarativeAttention([(2, 4)], tokenizer_decode = vocab.get)

    wrapper.generate(
        torch.tensor([[1, 2, 3]]),
        seq_len = 4,
        state_machine = da,
        filter_logits_fn = scripted_logits
    )

    # the tag was split across three sampled tokens and still detected
    assert da.is_local

def test_generation_dynamic_sampling_and_auto_restore():
    wrapper = make_wrapper()
    vocab = {10: '<verbatim chunk="1">', 11: 'x', 12: '</verbatim>', 13: 'y'}
    script = iter([10, 11, 12, 13])

    da = DeclarativeAttention([(0, 4)], tokenizer_decode = vocab.get)

    @da.on('verbatim')
    def verbatim(da, attrs):
        da.active_chunks = {int(attrs['chunk'])}
        da.state['temperature'] = 0.

    def scripted_logits(logits):
        logits = torch.full_like(logits, -1e9)
        logits[:, next(script, 13)] = 1e9
        return logits

    wrapper.generate(
        torch.tensor([[1, 2, 3]]),
        seq_len = 4,
        state_machine = da,
        logit_fn = scripted_logits
    )

    # after closing verbatim, da automatically restored to global and state['temperature'] was cleaned up
    assert da.is_global
    assert 'temperature' not in da.state

def test_batched_generation_with_mixed_temperatures():
    wrapper = make_wrapper()
    vocab = {10: '<verbatim chunk="1">', 11: 'x'}

    da1 = DeclarativeAttention([(0, 4)], tokenizer_decode = vocab.get)
    da2 = DeclarativeAttention([(0, 4)], tokenizer_decode = vocab.get)

    @da1.on('verbatim')
    def verbatim(da, attrs):
        da.active_chunks = {int(attrs['chunk'])}
        da.state['temperature'] = 0.

    da1.step('<verbatim chunk="1">') # seq 1 has temperature 0 (greedy)
    # seq 2 stays global with default temperature 1.0 (stochastic)

    out = wrapper.generate(
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        seq_len = 6,
        state_machine = [da1, da2]
    )

    assert out.shape == (2, 6)

    single_out = wrapper.generate(
        torch.tensor([[1, 2, 3]]),
        seq_len = 6,
        state_machine = da1
    )

    assert torch.equal(out[0], single_out[0])

def test_generation_two_vs_one_parity():
    torch.manual_seed(42)
    wrapper = make_wrapper(dim = 32, depth = 2, heads = 2)

    prompt1 = torch.tensor([[10, 11, 12, 13]])
    prompt2 = torch.tensor([[20, 21, 22, 23]])
    chunk_spans1 = [(0, 2)]
    chunk_spans2 = [(1, 3)]

    # 1. generate sequence 1 alone (batch size 1)
    out1 = wrapper.generate(
        prompt1,
        seq_len = 6,
        chunk_spans = chunk_spans1,
        temperature = 0.
    )

    # 2. generate sequence 2 alone (batch size 1)
    out2 = wrapper.generate(
        prompt2,
        seq_len = 6,
        chunk_spans = chunk_spans2,
        temperature = 0.
    )

    # 3. generate both together in a batch of 2
    out_batch = wrapper.generate(
        torch.cat([prompt1, prompt2], dim = 0),
        seq_len = 6,
        chunk_spans = [chunk_spans1, chunk_spans2],
        temperature = 0.
    )

    assert out_batch.shape == (2, 6)
    assert torch.equal(out_batch[0], out1[0]), "batch element 0 must match single generation of sequence 1"
    assert torch.equal(out_batch[1], out2[0]), "batch element 1 must match single generation of sequence 2"

def test_generation_dynamic_modes_two_vs_one_parity():
    vocab = {100: '<local>', 101: '</local>', 102: '<focus magic_chunks="1">'}
    decode_fn = lambda t: vocab.get(t, chr(t) if 0 <= t < 128 else '')

    torch.manual_seed(42)
    wrapper = make_wrapper(dim = 32, depth = 2, heads = 2)
    wrapper.tokenizer_decode = decode_fn

    # prompt 1 will emit <local>
    prompt1 = torch.tensor([[10, 11, 100]])
    # prompt 2 will emit <focus magic_chunks="1">
    prompt2 = torch.tensor([[20, 21, 102]])

    chunk_spans1 = [(0, 2)]
    chunk_spans2 = [(1, 2)]

    out1 = wrapper.generate(prompt1, seq_len = 5, chunk_spans = chunk_spans1, temperature = 0.)
    out2 = wrapper.generate(prompt2, seq_len = 5, chunk_spans = chunk_spans2, temperature = 0.)

    out_batch = wrapper.generate(
        torch.cat([prompt1, prompt2], dim = 0),
        seq_len = 5,
        chunk_spans = [chunk_spans1, chunk_spans2],
        temperature = 0.
    )

    assert torch.equal(out_batch[0], out1[0])
    assert torch.equal(out_batch[1], out2[0])

def test_custom_state_machine_instructing_prompt_e2e():
    torch.manual_seed(42)
    encode_fn = lambda s: [ord(c) for c in s]
    decode_fn = lambda t: chr(t) if 0 <= t < 128 else ''

    net = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 512,
        attn_layers = Decoder(dim = 32, depth = 2, heads = 2)
    )

    wrapper = DeclarativeAttentionWrapper(
        net,
        tokenizer_decode = decode_fn,
        tokenizer_encode = encode_fn
    )

    custom_instructions = """Answer using the magic chunks.
- <compare chunks="K,M">: compare chunks
- <brainstorm>: brainstorm locally"""

    prompt, chunk_spans = format_declarative_prompt(
        question = "What is the answer?",
        context = "Acme was founded in 2003.",
        instructions = custom_instructions,
        tokenizer_encode = encode_fn
    )

    da = DeclarativeAttention(chunk_spans, tokenizer_decode = decode_fn)

    @da.on('compare')
    def compare(machine, attrs):
        machine.active_chunks = parse_chunk_ids(attrs['chunks'])

    @da.on('brainstorm')
    def brainstorm(machine, attrs):
        machine.active_chunks = set()
        machine.state['temperature'] = 1.5

    # pass both prompt (list of token ints) and custom state machine to wrapper.generate
    out = wrapper.generate(prompt, seq_len = 16, state_machine = da)
    assert out.shape == (1, 16)

def test_generation_strict_during_generate_raises():
    torch.manual_seed(42)
    vocab = {100: '<focus magic_chunks="1">', 101: '<local>'}
    decode_fn = lambda t: vocab.get(t, chr(t) if 0 <= t < 128 else '')

    wrapper = make_wrapper(dim = 32, depth = 2, heads = 2)
    wrapper.tokenizer_decode = decode_fn

    call_count = 0
    def mock_logit_fn(logits):
        nonlocal call_count
        call_count += 1
        logits = torch.full_like(logits, -float('Inf'))
        token = 100 if call_count == 1 else 101
        logits[:, token] = 10.
        return logits

    da = DeclarativeAttention({1: (0, 1)}, tokenizer_decode = decode_fn, strict = True)

    with pytest.raises(StateMachineViolationError, match = "tag <local> opened before <focus> was closed"):
        wrapper.generate(
            torch.tensor([[1, 2]]),
            seq_len = 5,
            state_machine = da,
            logit_fn = mock_logit_fn,
            temperature = 0.
        )

def test_generation_strict_false_during_generate_continues():
    torch.manual_seed(42)
    vocab = {100: '<focus magic_chunks="1">', 101: '<local>'}
    decode_fn = lambda t: vocab.get(t, chr(t) if 0 <= t < 128 else '')

    wrapper = make_wrapper(dim = 32, depth = 2, heads = 2)
    wrapper.tokenizer_decode = decode_fn

    call_count = 0
    def mock_logit_fn(logits):
        nonlocal call_count
        call_count += 1
        logits = torch.full_like(logits, -float('Inf'))
        token = 100 if call_count == 1 else 101
        logits[:, token] = 10.
        return logits

    da = DeclarativeAttention({1: (0, 1)}, tokenizer_decode = decode_fn, strict = False)

    # strict=False allows continuing past unclosed tag
    out = wrapper.generate(
        torch.tensor([[1, 2]]),
        seq_len = 3,
        state_machine = da,
        logit_fn = mock_logit_fn,
        temperature = 0.
    )
    assert out.shape == (1, 3)

def test_forward_1d_tensor():
    wrapper = make_wrapper()
    chunk_spans = [(2, 4)]
    x_1d = torch.randint(1, 10, (16,))
    loss = wrapper(x_1d, chunk_spans = chunk_spans)
    assert loss.ndim == 0 and not torch.isnan(loss)
