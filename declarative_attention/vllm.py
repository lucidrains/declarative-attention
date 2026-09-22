from __future__ import annotations
from typing import Sequence

from torch import Tensor

from declarative_attention.declarative_attention import (
    DeclarativeAttention,
    ChunkSpans,
    TokenizerDecode,
    exists,
    default
)

class DeclarativeVLLMHook:
    # applies declarative attention to vLLM's paged KV cache (Ho et al., 2026,
    # Section 2.3 and Appendix B)
    #
    # vLLM stores the KV cache in fixed-size blocks and its attention kernels
    # read whole blocks, so masking saves time only if entire blocks are skipped
    # - the token mask is therefore rounded outward to block boundaries and each
    # request's block table is rewritten to contain only the kept blocks, so the
    # kernel simply reads less, with no kernel modifications
    #
    # hook `__call__` onto the attention metadata builder at each decode step

    def __init__(
        self,
        block_size: int = 16,
        tokenizer_decode: TokenizerDecode | None = None,
        strict: bool = False
    ):
        self.block_size = block_size
        self.tokenizer_decode = tokenizer_decode
        self.strict = strict
        self.machines: dict[str, DeclarativeAttention] = dict()

    def register_request(
        self,
        request_id: str,
        chunk_spans: ChunkSpans,
        strict: bool | None = None
    ) -> DeclarativeAttention:
        machine = DeclarativeAttention(
            chunk_spans,
            tokenizer_decode = self.tokenizer_decode,
            strict = default(strict, self.strict),
            block_size = self.block_size
        )
        self.machines[request_id] = machine
        return machine

    def unregister_request(self, request_id: str):
        self.machines.pop(request_id, None)

    register = register_request
    unregister = unregister_request

    def step(self, request_id: str, token: int | Tensor | str):
        # feeds a generated token to a request's state machine

        machine = self.machines.get(request_id)

        if exists(machine):
            machine.step(token)

    def __call__(
        self,
        block_tables: Tensor,
        seq_lens: Tensor | Sequence[int] | int,
        request_ids: Sequence[str] | str
    ) -> tuple[Tensor, Tensor | Sequence[int] | int]:
        # rewrites a batch of block tables and sequence lengths so the attention
        # kernel reads only the blocks kept by each request's state machine
        # `block_tables` has shape (batch, max_blocks), or (max_blocks,) for a single request

        is_single = isinstance(request_ids, str)
        request_ids = [request_ids] if is_single else list(request_ids)

        if block_tables.ndim == 1:
            block_tables = block_tables[None]

        is_seq_lens_tensor = isinstance(seq_lens, Tensor)

        if is_seq_lens_tensor:
            seq_lens_list = seq_lens.flatten().tolist()
        elif isinstance(seq_lens, int):
            seq_lens_list = [seq_lens]
        else:
            seq_lens_list = list(seq_lens)

        new_block_tables = block_tables.clone()
        new_seq_lens = list(seq_lens_list)

        for i, request_id in enumerate(request_ids):
            machine = self.machines.get(request_id)

            if not exists(machine):
                continue

            seq_len = int(seq_lens_list[i])
            kept = machine.get_kept_blocks(seq_len)

            new_block_tables[i] = 0
            new_block_tables[i, :len(kept)] = block_tables[i, kept]

            last_block = (seq_len - 1) // self.block_size
            last_block_len = (seq_len - 1) % self.block_size + 1

            # if the last (partial) block survived, the compacted length ends
            # inside it - otherwise every kept block is full

            new_seq_lens[i] = (
                (len(kept) - 1) * self.block_size + last_block_len
                if last_block in kept
                else len(kept) * self.block_size
            )

        if is_single:
            return new_block_tables[0], new_seq_lens[0]

        if is_seq_lens_tensor:
            new_seq_lens = seq_lens.new_tensor(new_seq_lens)

        return new_block_tables, new_seq_lens
