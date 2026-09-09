from __future__ import annotations
from html.parser import HTMLParser
from typing import Callable, Sequence

import torch
from torch import Tensor
from statemachine import StateMachine, State
from torch_einops_utils import maybe

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def default_decode_fn(token: int) -> str:
    return chr(token) if 0 <= token < 128 else ""

# types

ChunkId = int
StartPos = int
EndPos = int
Span = tuple[StartPos, EndPos]

# chunk spans can be:
# 1. dict mapping 1-indexed chunk id to token span:
#    {1: (16, 32), 2: (32, 48)}
# 2. sequence of spans (auto-indexed from 1):
#    [(16, 32), (32, 48)]

ChunkSpans = (
    dict[ChunkId, Span] |
    Sequence[Span]
)

TokenizerDecode = Callable[[int], str]

# general streaming tag parser

class StreamingTagParser(HTMLParser):
    def __init__(self, on_tag: Callable[[bool, str, dict[str, str]], None]):
        super().__init__()
        self.on_tag = on_tag

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.on_tag(False, tag, dict(attrs))

    def handle_endtag(self, tag: str):
        self.on_tag(True, tag, {})

# declarative attention state machine (Ho et al., 2026)

class DeclarativeStateMachine(StateMachine):
    """
    Declarative Attention State Machine

    - <global>              : attends to all context chunks (default)
    - <focus chunks="1,2">  : attends only to specified chunk(s)
    - <local>               : attends to 0 context chunks
    """

    global_mode = State(initial = True)
    focus_mode  = State()
    local_mode  = State()

    focus  = focus_mode.from_(global_mode, local_mode)
    local  = local_mode.from_(global_mode, focus_mode)
    revert = global_mode.from_(focus_mode, local_mode)

    def __init__(
        self,
        chunk_spans: ChunkSpans,
        tokenizer_decode: TokenizerDecode | None = None
    ):
        if isinstance(chunk_spans, (list, tuple)):
            chunk_spans = {i + 1: span for i, span in enumerate(chunk_spans)}

        self.chunk_spans = chunk_spans
        self.active_chunks = set(chunk_spans.keys())
        self.parser = StreamingTagParser(self._handle_tag)
        self.tokenizer_decode = default(tokenizer_decode, default_decode_fn)
        self.handlers: dict[str, Callable] = dict()

        super().__init__()

    # register custom tag handlers for arbitrary researcher logic

    def on(self, tag: str, fn: Callable | None = None):
        def decorator(handler: Callable):
            self.handlers[tag.lower()] = handler
            return handler

        return maybe(decorator, default = decorator)(fn)

    # dispatch streaming tags to transitions or custom handlers

    def _handle_tag(self, is_closing: bool, tag: str, attrs: dict[str, str]):
        tag = tag.lower()

        if is_closing:
            if hasattr(self, f"revert_{tag}"):
                self.send(f"revert_{tag}")
            elif not self.is_global:
                try: self.revert()
                except Exception: pass
            return

        # opening tag: try transition event first, then custom handler

        if tag in [e.id for e in self.events]:
            try:
                self.send(tag, **attrs)
                return
            except Exception:
                pass

        if tag in self.handlers:
            self.handlers[tag](self, **attrs)

    # transition callbacks

    @focus.on
    def _on_focus(
        self,
        chunks: str | Sequence[int] | None = None,
        chunk: str | int | None = None,
        magic_chunks: str | None = None
    ):
        target = default(chunks, default(chunk, magic_chunks))

        if isinstance(target, str):
            self.active_chunks = {int(c) for c in target.split(',') if c.strip().isdigit()}
        elif isinstance(target, int):
            self.active_chunks = {target}
        elif exists(target):
            self.active_chunks = set(target)
        else:
            self.active_chunks = set()

    @local.on
    def _on_local(self):
        self.active_chunks = set()

    @revert.on
    def _on_revert(self):
        self.active_chunks = set(self.chunk_spans.keys())

    # properties

    @property
    def is_global(self):
        return self.global_mode.is_active

    @property
    def is_focus(self):
        return self.focus_mode.is_active

    @property
    def is_local(self):
        return self.local_mode.is_active

    @property
    def mode(self):
        return "global" if self.is_global else ("focus" if self.is_focus else "local")

    @property
    def is_streaming_tag(self) -> bool:
        """True if the parser is currently mid-tag (e.g. between '<' and '>')"""
        return "<" in self.parser.rawdata

    # token streaming (accepts int, torch LongTensor, or str)

    def step(self, token: int | Tensor | str):
        if isinstance(token, Tensor):
            token = token.item()

        text = self.tokenizer_decode(token) if isinstance(token, int) else str(token)
        self.parser.feed(text)

    # attention mask generation

    def get_mask(self, total_len: int, device = None) -> Tensor:
        mask = torch.ones(total_len, dtype = torch.bool, device = device)

        if self.is_global:
            return mask

        for chunk_id, (start, end) in self.chunk_spans.items():
            if chunk_id not in self.active_chunks:
                mask[start:end] = False

        return mask

    def __call__(self, total_len: int, device = None) -> Tensor:
        return self.get_mask(total_len, device = device)
