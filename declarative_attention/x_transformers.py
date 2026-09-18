from __future__ import annotations
from typing import Callable, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from einops import rearrange, pack, unpack

from torch_einops_utils import (
    pad_sequence,
    tree_map_tensor_to_device
)

from declarative_attention.declarative_attention import (
    DeclarativeAttention,
    ChunkSpans,
    TokenizerDecode,
    derive_declarative_mask,
    is_batched_spans,
    exists,
    default
)

# helper sampling function

def top_k(logits, thres = 0.9):
    k = max(int((1 - thres) * logits.shape[-1]), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -float('Inf'))
    probs.scatter_(-1, ind, val)
    return probs

# autoregressive wrapper for declarative attention

class DeclarativeAttentionWrapper(nn.Module):
    """
    simple autoregressive wrapper for declarative attention

    training derives the 2d causal DA mask from the full sequence; generation
    streams sampled tokens through the state machine and passes the per-step KV
    mask to the model

    any machine exposing `step`, `get_mask` and `state` can be driven this way
    """

    def __init__(
        self,
        net: nn.Module,
        pad_value: int = 0,
        ignore_index: int = -100,
        tokenizer_decode: TokenizerDecode | None = None
    ):
        super().__init__()
        self.net = net
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.tokenizer_decode = tokenizer_decode
        self.max_seq_len = net.max_seq_len

    def forward(
        self,
        x: Tensor,
        chunk_spans: ChunkSpans | Sequence[ChunkSpans] | None = None,
        prompt_len: int | None = None,
        attn_mask: Tensor | None = None,
        **kwargs
    ):
        inp, target = x[:, :-1], x[:, 1:]

        if exists(chunk_spans) and not exists(attn_mask):
            attn_mask = derive_declarative_mask(
                inp,
                chunk_spans,
                prompt_len = prompt_len,
                tokenizer_decode = self.tokenizer_decode,
                device = x.device
            )

        logits = self.net(inp, attn_mask = attn_mask, **kwargs)
        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = self.ignore_index
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        prompts: list[Tensor] | Tensor,
        seq_len: int,
        chunk_spans: ChunkSpans | Sequence[ChunkSpans] | None = None,
        state_machine: DeclarativeAttention | Sequence[DeclarativeAttention] | None = None,
        temperature: float = 1.,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        logit_fn: Callable | None = None,
        cache_kv: bool = True,
        eos_token: int | None = None,
        **kwargs
    ):
        batch = len(prompts) if isinstance(prompts, (list, tuple)) else (prompts.shape[0] if prompts.ndim > 1 else 1)

        machines = state_machine
        if not exists(machines) and exists(chunk_spans):
            spans = list(chunk_spans) if is_batched_spans(chunk_spans) else [chunk_spans] * batch
            machines = [DeclarativeAttention(s, tokenizer_decode = self.tokenizer_decode) for s in spans]
        elif exists(machines) and not isinstance(machines, (list, tuple)):
            machines = [machines] * batch

        if isinstance(prompts, list):
            prompts = pad_sequence(prompts)

        prompts, ps = pack([prompts], '* n')
        out = prompts
        t = prompts.shape[-1]
        cache = None
        sample = None

        for _ in range(seq_len):
            if exists(machines) and exists(sample):
                for machine, token in zip(machines, rearrange(sample, 'b 1 -> b')):
                    machine.step(token)

            step_kwargs = dict()

            if exists(machines):
                masks = [machine.get_mask(out.shape[-1], device = out.device) for machine in machines]
                step_kwargs['self_attn_kv_mask'] = torch.stack(masks)

                temps = [machine.state.get('temperature', temperature) for machine in machines]

                for machine in machines:
                    step_kwargs.update(machine.state)

                step_kwargs = tree_map_tensor_to_device(step_kwargs, out.device)
            else:
                temps = [temperature] * batch

            if step_kwargs.pop('stop', False):
                break

            step_kwargs.pop('temperature', None)
            step_filter = step_kwargs.pop('filter_logits_fn', filter_logits_fn)
            step_filter_kw = step_kwargs.pop('filter_kwargs', filter_kwargs)
            step_logit_fn = step_kwargs.pop('logit_fn', logit_fn)

            can_cache = cache_kv and self.net.can_cache_kv
            x = out[:, -1:] if (can_cache and exists(cache)) else out

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                **kwargs,
                **step_kwargs
            )

            if can_cache:
                cache = new_cache

            logits = logits[:, -1]

            if exists(step_logit_fn):
                logits = step_logit_fn(logits)

            filtered_logits = step_filter(logits, **step_filter_kw)
            temps = torch.tensor(temps, device = out.device, dtype = logits.dtype)

            if (temps == 0.).all():
                sample = logits.argmax(dim = -1, keepdim = True)
            elif not (temps == 0.).any():
                probs = F.softmax(filtered_logits / rearrange(temps, 'b -> b 1'), dim = -1)
                sample = torch.multinomial(probs, 1)
            else:
                greedy_sample = logits.argmax(dim = -1, keepdim = True)
                probs = F.softmax(filtered_logits / rearrange(temps.clamp(min = 1e-5), 'b -> b 1'), dim = -1)
                stochastic_sample = torch.multinomial(probs, 1)
                sample = torch.where(rearrange(temps == 0., 'b -> b 1'), greedy_sample, stochastic_sample)

            out = torch.cat((out, sample), dim = -1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)
                if is_eos_tokens.any(dim = -1).all():
                    break

        if exists(eos_token):
            shifted = F.pad(is_eos_tokens, (1, -1))
            mask = shifted.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        out = out[:, t:]
        out, = unpack(out, ps, '* n')
        return out
