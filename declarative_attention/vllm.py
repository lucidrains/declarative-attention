from __future__ import annotations
from typing import Sequence

from torch import Tensor
from torch_einops_utils import pack_with_inverse

from declarative_attention.declarative_attention import (
    DeclarativeAttention,
    ChunkSpans,
    TokenizerDecode,
    exists
)

class DeclarativeVLLMHook:
    """
    Applies Declarative Attention to vLLM's paged KV cache (Ho et al., 2026,
    Section 2.3 and Appendix B).

    vLLM stores the KV cache in fixed-size blocks and its attention kernels read
    whole blocks, so masking saves time only if entire blocks are skipped. The
    state machine's token mask is therefore rounded outward to block boundaries
    and each request's block table is rewritten to contain only the kept blocks.
    The kernel then simply reads less, with no kernel modifications.

    Hook `__call__` onto the attention metadata builder at each decode step.
    """

    def __init__(
        self,
        block_size: int = 16,
        tokenizer_decode: TokenizerDecode | None = None
    ):
        self.block_size = block_size
        self.tokenizer_decode = tokenizer_decode
        self.machines: dict[str, DeclarativeAttention] = dict()

    def register_request(self, request_id: str, chunk_spans: ChunkSpans) -> DeclarativeAttention:
        machine = DeclarativeAttention(
            chunk_spans,
            tokenizer_decode = self.tokenizer_decode,
            block_size = self.block_size
        )
        self.machines[request_id] = machine
        return machine

    def unregister_request(self, request_id: str):
        self.machines.pop(request_id, None)

    def step(self, request_id: str, token: int | Tensor | str):
        """Feeds a generated token to a request's state machine."""
        machine = self.machines.get(request_id)
        if exists(machine):
            machine.step(token)

    def __call__(
        self,
        block_tables: Tensor,
        seq_lens: Tensor | Sequence[int] | int,
        request_ids: Sequence[str] | str
    ) -> tuple[Tensor, Tensor | Sequence[int] | int]:
        """
        Rewrites a batch of block tables and sequence lengths so the attention
        kernel reads only the blocks kept by each request's state machine.
        `block_tables` has shape (batch, max_blocks), or (max_blocks,) for a single request.
        """
        block_tables, inverse_pack_blocks = pack_with_inverse([block_tables], '* n')

        is_seq_lens_tensor = isinstance(seq_lens, Tensor)
        if is_seq_lens_tensor:
            seq_lens, inverse_pack_seq_lens = pack_with_inverse([seq_lens], '*')

        is_single = isinstance(request_ids, str)
        if is_single:
            request_ids = [request_ids]
            seq_lens = [seq_lens]

        new_block_tables = block_tables.clone()
        new_seq_lens = seq_lens.clone() if isinstance(seq_lens, Tensor) else list(seq_lens)

        for i, request_id in enumerate(request_ids):
            machine = self.machines.get(request_id)
            if not exists(machine):
                continue

            kept = machine.get_kept_blocks(int(seq_lens[i]))

            new_block_tables[i] = 0
            new_block_tables[i, :len(kept)] = block_tables[i, kept]
            new_seq_lens[i] = len(kept) * self.block_size

        new_block_tables, = inverse_pack_blocks(new_block_tables)

        if is_single:
            return new_block_tables, new_seq_lens[0]

        if is_seq_lens_tensor:
            new_seq_lens, = inverse_pack_seq_lens(new_seq_lens)

        return new_block_tables, new_seq_lens
