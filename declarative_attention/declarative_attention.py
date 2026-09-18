from __future__ import annotations

import re
import inspect
from html.parser import HTMLParser
from typing import Callable, Sequence

import jinja2

import torch
from torch import Tensor
from einops import rearrange, reduce

from torch_einops_utils import (
    pad_right_at_dim,
    tree_flatten_with_inverse,
    tree_map_tensor
)

# helpers

def exists(val):
    return val is not None

def default(val, fallback):
    return val if exists(val) else fallback

def default_decode_fn(token):
    return chr(token) if 0 <= token < 128 else ''

def is_span(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) for v in value)

def is_batched_spans(chunk_spans):
    if not isinstance(chunk_spans, (list, tuple)) or len(chunk_spans) == 0:
        return False

    return all(isinstance(spans, dict) or not is_span(spans) for spans in chunk_spans)

def parse_chunk_ids(value):
    if isinstance(value, int):
        return {value}

    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}

    return {int(part) for part in re.findall(r'\d+', str(value))}

def accepts_is_closing(fn: Callable) -> bool:
    try:
        sig = inspect.signature(fn)
        return len(sig.parameters) >= 3 or any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    except (ValueError, TypeError):
        return False

# types

Span = tuple[int, int]
ChunkSpans = dict[int, Span] | Sequence[Span]
TokenizerDecode = Callable[[int], str]
TokenizerEncode = Callable[[str], list[int]]

# constants

CHUNK_ATTRS = ('magic_chunks', 'chunks', 'chunk', 'ids', 'id')

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

DEFAULT_INSTRUCTIONS = """Answer the question above using the magic chunks.
Reason through the magic chunks using three modes:
- <global> (default): all magic chunks are visible. Use to plan which magic chunk to examine next.
- <focus magic_chunks="K">: only magic chunk K is visible. Use to extract verbatim facts.
- <local>: no magic chunks are visible. Use to plan and synthesize facts already extracted.
Use at least one <focus> block, end with a <local> block committing to the final answer,
and wrap the final answer in <answer>...</answer>."""

# streaming tag parser

class TagParser(HTMLParser):
    # streaming html/xml parser, emits (is_closing, tag, attrs) once a tag completes

    def __init__(self, on_tag: Callable):
        super().__init__()
        self.on_tag = on_tag

    def _parse_attrs(self, attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        attrs_dict = dict()

        for k, v in attrs:
            attrs_dict[k] = default(v, k)
            if not exists(v) and not any(a in attrs_dict for a in CHUNK_ATTRS):
                attrs_dict['chunks'] = k

        return attrs_dict

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.on_tag(False, tag.lower(), self._parse_attrs(attrs))

    def handle_endtag(self, tag: str):
        self.on_tag(True, tag.lower(), dict())

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.handle_starttag(tag, attrs)

# state machine

class TagStateMachine:
    # streaming state machine over generated text - completed tags dispatch to
    # handlers registered with `on`, which may mutate the machine in any way
    #
    #     @machine.on('window')
    #     def window(machine, attrs):
    #         machine.state['window_size'] = int(attrs['size'])
    #
    # `state` holds arbitrary per-step modifications for an inference engine,
    # and active chunks are snapshotted on tag open, restored on tag close

    def __init__(self, tokenizer_decode: TokenizerDecode | None = None):
        self.tokenizer_decode = default(tokenizer_decode, default_decode_fn)
        self.handlers = dict()
        self.state = dict()
        self.stack = []
        self.active_chunks = None
        self.parser = TagParser(on_tag = self._on_tag)

    def _snapshot(self):
        chunks = self.active_chunks
        return (
            dict(self.state),
            set(chunks) if exists(chunks) else None
        )

    def _restore(self, snapshot):
        self.state, self.active_chunks = snapshot

    def step(self, token: int | Tensor | str | Sequence):
        tokens, _ = tree_flatten_with_inverse(tree_map_tensor(
            lambda t: t.tolist() if t.numel() > 1 else t.item(),
            token
        ))

        for one_token in tokens:
            decoded = self.tokenizer_decode(one_token) if isinstance(one_token, int) else str(one_token)

            if exists(decoded):
                self.parser.feed(decoded)

    def on(self, tag: str, fn: Callable | None = None):
        # handler signature - fn(machine, attrs) or fn(machine, attrs, is_closing)

        def register(fn):
            self.handlers[tag.lower()] = fn
            return fn

        return register(fn) if exists(fn) else register

    def _on_tag(self, is_closing, tag, attrs):
        handler = self.handlers.get(tag)

        if is_closing:
            # unwind to the snapshot saved when the matching tag opened

            for i in reversed(range(len(self.stack))):
                saved_tag, snapshot = self.stack[i]

                if saved_tag != tag:
                    continue

                self._restore(snapshot)
                self.stack.pop(i)
                break

            if exists(handler) and accepts_is_closing(handler):
                handler(self, attrs, is_closing)

            return

        self.stack.append((tag, self._snapshot()))

        if not exists(handler):
            return

        if accepts_is_closing(handler):
            handler(self, attrs, False)
        else:
            handler(self, attrs)

# declarative attention mode handlers

def handle_global(machine, attrs):
    machine.active_chunks = set(machine.all_chunks)

def handle_focus(machine, attrs):
    value = next((attrs[name] for name in CHUNK_ATTRS if name in attrs), '')
    machine.active_chunks = parse_chunk_ids(value) & machine.all_chunks

def handle_local(machine, attrs):
    machine.active_chunks = set()

# main class

class DeclarativeAttention(TagStateMachine):
    # declarative attention - Ho et al., 2026
    #
    # the model declares which context chunks it needs in its output stream and
    # the state machine derives the attention mask per decode step, so the
    # engine reads fewer KV blocks while the full KV cache stays resident
    #
    #     - <global>                    all chunks visible (default)
    #     - <focus magic_chunks="K,M">  only the named chunks visible
    #     - <local>                     no chunks visible
    #
    # only chunks are masked, so the scaffold and the response so far stay
    # attended in every mode, provided the chunk spans exclude the scaffold
    #
    # custom protocols subclass and override `instructions`, then register
    # their own tags with `on`

    instructions = DEFAULT_INSTRUCTIONS

    def __init__(
        self,
        chunk_spans: ChunkSpans,
        tokenizer_decode: TokenizerDecode | None = None,
        block_size: int = 16
    ):
        super().__init__(tokenizer_decode)

        if isinstance(chunk_spans, (list, tuple)):
            chunk_spans = {i + 1: tuple(span) for i, span in enumerate(chunk_spans)}

        self.chunk_spans = chunk_spans
        self.all_chunks = set(chunk_spans)
        self.block_size = block_size
        self.active_chunks = set(self.all_chunks)

        self.on('global', handle_global)
        self.on('focus', handle_focus)
        self.on('local', handle_local)

    @classmethod
    def format_prompt(
        cls,
        question: str,
        context: str | Sequence[str],
        tokenizer_encode: TokenizerEncode | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        template: str | jinja2.Template | None = None
    ):
        # builds a prompt instructing the model to use this machine's protocol

        return format_declarative_prompt(
            question,
            context,
            tokenizer_encode = tokenizer_encode,
            system_prompt = system_prompt,
            instructions = cls.instructions,
            template = template
        )

    @property
    def mode(self):
        if self.active_chunks == self.all_chunks:
            return 'global'

        return 'focus' if self.active_chunks else 'local'

    @property
    def is_global(self):
        return self.mode == 'global'

    @property
    def is_focus(self):
        return self.mode == 'focus'

    @property
    def is_local(self):
        return self.mode == 'local'

    def get_mask(self, total_len: int, device = None) -> Tensor:
        # 1d boolean mask over KV positions - True = attended

        mask = torch.ones(total_len, dtype = torch.bool, device = device)

        if self.is_global:
            return mask

        for chunk_id, (start, end) in self.chunk_spans.items():
            if chunk_id not in self.active_chunks:
                mask[start:end] = False

        return mask

    def get_block_mask(self, total_len: int, device = None) -> Tensor:
        # block-aligned mask - a block is kept if any of its tokens is attended,
        # rounding kept spans outward so no declared token is ever dropped

        mask = self.get_mask(total_len, device = device)

        pad = -total_len % self.block_size
        mask = pad_right_at_dim(mask, pad, value = False)

        return reduce(mask, '(b n) -> b', 'any', n = self.block_size)

    def get_kept_blocks(self, total_len: int) -> list[int]:
        return self.get_block_mask(total_len).nonzero(as_tuple = True)[0].tolist()

# full sequence mask for training

def derive_declarative_mask(
    tokens: Tensor,
    chunk_spans: ChunkSpans | Sequence[ChunkSpans],
    prompt_len: int | None = None,
    tokenizer_decode: TokenizerDecode | None = None,
    device = None
) -> Tensor:
    # derives the 2d causal declarative mask from a full sequence by replaying
    # the tokens through the state machine - returns (seq, seq), or
    # (batch, 1, seq, seq) if batched
    #
    # only generated tokens drive the machine, so pass `prompt_len` when the
    # prompt itself mentions tags

    device = default(device, tokens.device)
    batched = tokens.ndim == 2

    if not batched:
        tokens = tokens[None]

    batch, seq_len = tokens.shape

    if is_batched_spans(chunk_spans):
        assert len(chunk_spans) == batch, 'one set of chunk spans is needed per sequence'
        batch_spans = list(chunk_spans)
    else:
        batch_spans = [chunk_spans] * batch

    masks = []

    for b, spans in enumerate(batch_spans):
        machine = DeclarativeAttention(spans, tokenizer_decode = tokenizer_decode)
        mask = torch.zeros((seq_len, seq_len), dtype = torch.bool, device = device)

        for i in range(seq_len):
            mask[i] = machine.get_mask(seq_len, device = device)
            mask[i, i + 1:] = False

            if not exists(prompt_len) or i >= prompt_len:
                machine.step(tokens[b, i])

        masks.append(mask)

    out = torch.stack(masks, dim = 0)

    if not batched:
        return out[0]

    return rearrange(out, 'b i j -> b 1 i j')

# context segmentation (Ho et al., 2026, Appendix F)

SPLIT_PATTERNS = (
    r'\n\n+',
    r'\n',
    r'[.!?]\s+',
    r'[;:,]\s+',
    r'\s+'
)

def count_tokens(text, tokenizer_encode = None):
    if exists(tokenizer_encode):
        return len(tokenizer_encode(text))

    return max(1, len(text) // 4)

def split_after(text, pattern):
    ends = [match.end() for match in re.finditer(pattern, text)]
    bounds = [0, *ends, len(text)]

    return [text[start:end] for start, end in zip(bounds, bounds[1:]) if start < end]

def split_unit(text, tokenizer_encode, max_tokens):
    if count_tokens(text, tokenizer_encode) <= max_tokens:
        return [text]

    for pattern in SPLIT_PATTERNS:
        pieces = split_after(text, pattern)

        if len(pieces) > 1:
            return [unit for piece in pieces for unit in split_unit(piece, tokenizer_encode, max_tokens)]

    return [text]

def segment_context(text, tokenizer_encode = None, target_tokens = 2048, max_tokens = 2560):
    # splits a context into addressable segments - a unit is split only if it
    # exceeds `max_tokens`, at the coarsest available boundary, and adjacent
    # units are merged up to `target_tokens`
    #
    # segments form a lossless partition - concatenating them reproduces `text`

    if len(text.strip()) == 0:
        return ['<empty_context>']

    segments = []
    current, current_tokens = '', 0

    for unit in split_unit(text, tokenizer_encode, max_tokens):
        unit_tokens = count_tokens(unit, tokenizer_encode)

        if current and current_tokens + unit_tokens > target_tokens:
            segments.append(current)
            current, current_tokens = unit, unit_tokens
            continue

        current += unit
        current_tokens += unit_tokens

    if current:
        segments.append(current)

    return segments

# prompt construction (Ho et al., 2026, Section 2.1)

DEFAULT_PROMPT_TEMPLATE = """{{ system_prompt.strip() }}

{% for chunk in chunks %}
Magic Chunk {{ loop.index }}:
{{ marker_start }}{{ chunk }}{{ marker_end }}

{% endfor -%}
Question:
{{ question.strip() }}

Instructions:
{{ instructions.strip() }}

Response:
"""

MARKER_START = '\x00'
MARKER_END = '\x01'

def format_declarative_prompt(
    question: str,
    context: str | Sequence[str],
    tokenizer_encode: TokenizerEncode | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    instructions: str = DEFAULT_INSTRUCTIONS,
    template: str | jinja2.Template | None = None
):
    # builds the declarative attention prompt, delivering the context as
    # numbered magic chunks, and returns the chunk spans the state machine needs
    #
    # `context` may be a raw string, segmented with `segment_context`, or a
    # sequence of pre-made chunks - with a `tokenizer_encode`, the prompt is
    # token ids and spans are token indices, otherwise text with char offsets

    chunks = segment_context(context, tokenizer_encode) if isinstance(context, str) else list(context)

    def encode(text):
        if not exists(tokenizer_encode):
            return text

        tokens = tokenizer_encode(text)
        return tokens.tolist() if isinstance(tokens, Tensor) else list(tokens)

    prompt_template = default(template, DEFAULT_PROMPT_TEMPLATE)

    if isinstance(prompt_template, str):
        prompt_template = jinja2.Template(prompt_template, trim_blocks = True, lstrip_blocks = True)

    rendered = prompt_template.render(
        system_prompt = system_prompt,
        chunks = chunks,
        question = question,
        instructions = instructions,
        marker_start = MARKER_START,
        marker_end = MARKER_END
    )

    prompt = [] if exists(tokenizer_encode) else ''
    chunk_spans = dict()

    for part in rendered.split(MARKER_START):
        if MARKER_END not in part:
            prompt += encode(part)
            continue

        chunk_content, rest = part.split(MARKER_END, 1)

        start = len(prompt)
        prompt += encode(chunk_content)
        chunk_spans[len(chunk_spans) + 1] = (start, len(prompt))

        prompt += encode(rest)

    return prompt, chunk_spans

# extracting chunk spans from pre-formatted text

DEFAULT_CHUNK_PATTERNS = (
    r'<(?:magic_chunk|chunk)(?:\s+(?:id=)?["\']?(\d+)["\']?)?\s*>(.*?)</(?:magic_chunk|chunk)>',
    r'Magic Chunk\s+(\d+):\s*\n(.*?)(?=(?:\n\s*Magic Chunk\s+\d+:|\n\s*(?:Question|Instructions|Response):|\Z))'
)

def extract_chunk_spans(
    text: str | Sequence[str],
    pattern: str | re.Pattern | None = None,
    tag: str | None = None,
    tokenizer_encode: TokenizerEncode | None = None,
    strip: bool = True
) -> dict[int, Span] | list[dict[int, Span]]:
    # extracts chunk spans (char or token offsets) from text with delimited chunks
    #
    # supports:
    # - a single string, or a list of strings for batched prompts
    # - xml tags - <chunk id="1">...</chunk> or <magic_chunk id="1">...</magic_chunk>
    # - paper format - Magic Chunk 1:\n...
    # - custom tag - tag="document" -> <document id="1">...</document>
    # - custom regex pattern with id and content groups

    if isinstance(text, (list, tuple)):
        return [
            extract_chunk_spans(
                t,
                pattern = pattern,
                tag = tag,
                tokenizer_encode = tokenizer_encode,
                strip = strip
            )
            for t in text
        ]

    if exists(tag):
        patterns = [rf'<{tag}(?:\s+(?:id=)?["\']?(\d+)["\']?)?\s*>(.*?)</{tag}>']
    elif exists(pattern):
        patterns = [pattern]
    else:
        patterns = DEFAULT_CHUNK_PATTERNS

    spans = dict()
    auto_id = 1

    def count(s):
        if not exists(tokenizer_encode):
            return len(s)

        res = tokenizer_encode(s)
        return res.shape[0] if isinstance(res, Tensor) else len(res)

    for pat in patterns:
        matches = list(re.finditer(pat, text, re.DOTALL | re.IGNORECASE))

        if not matches:
            continue

        for m in matches:
            groups = m.groups()
            chunk_id = int(groups[0]) if len(groups) > 1 and groups[0] and groups[0].isdigit() else auto_id
            auto_id = max(auto_id, chunk_id + 1)

            char_start, char_end = m.span(len(groups))

            if strip:
                raw = text[char_start:char_end]
                char_start += len(raw) - len(raw.lstrip())
                char_end -= len(raw) - len(raw.rstrip())

            start = count(text[:char_start])
            end = count(text[:char_end])

            spans[chunk_id] = (start, end)

        break

    return spans
