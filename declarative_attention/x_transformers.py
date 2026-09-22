import copy
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
    TokenizerEncode,
    derive_declarative_mask,
    is_batched_spans,
    exists,
    default
)

# helper sampling function

def log(t, eps = None):
    eps = default(eps, torch.finfo(t.dtype).eps)
    return torch.log(t.clamp(min = eps, max = 1. - eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(logits, temperature = 1., dim = -1, keepdim = True):
    if not isinstance(temperature, Tensor):
        temperature = torch.tensor(temperature, device = logits.device, dtype = logits.dtype)

    if temperature.ndim == 1:
        temperature = rearrange(temperature, 'b -> b 1')

    if (temperature == 0.).all():
        return logits.argmax(dim = dim, keepdim = keepdim)

    noise = gumbel_noise(logits)

    return (logits + noise * temperature).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = max(int((1 - thres) * logits.shape[-1]), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, -float('Inf'))
    probs.scatter_(-1, ind, val)
    return probs

# autoregressive wrapper for declarative attention

class DeclarativeAttentionWrapper(nn.Module):
    # simple autoregressive wrapper for declarative attention
    #
    # training derives the 2d causal DA mask from the full sequence, generation
    # streams sampled tokens through the state machine and passes the per-step
    # KV mask to the model
    #
    # any machine exposing `step`, `get_mask` and `state` can be driven this way

    def __init__(
        self,
        net: nn.Module,
        pad_value: int = 0,
        ignore_index: int = -100,
        tokenizer_decode: TokenizerDecode | None = None,
        tokenizer_encode: TokenizerEncode | None = None,
        strict: bool = False
    ):
        super().__init__()
        self.net = net
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.tokenizer_decode = tokenizer_decode
        self.tokenizer_encode = tokenizer_encode
        self.strict = strict
        self.max_seq_len = net.max_seq_len

    def forward(
        self,
        x: Tensor,
        chunk_spans: ChunkSpans | Sequence[ChunkSpans] | None = None,
        prompt_len: int | None = None,
        attn_mask: Tensor | None = None,
        **kwargs
    ):
        x = rearrange(x, 'n -> 1 n') if x.ndim == 1 else x
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
        prompts: Tensor | list[Tensor] | str | list[str] | list[int] | list[list[int]],
        seq_len: int,
        chunk_spans: ChunkSpans | Sequence[ChunkSpans] | None = None,
        state_machine: DeclarativeAttention | Sequence[DeclarativeAttention] | None = None,
        strict: bool | None = None,
        temperature: float = 1.,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        logit_fn: Callable | None = None,
        cache_kv: bool = True,
        eos_token: int | None = None,
        **kwargs
    ):
        if isinstance(prompts, str):
            assert exists(self.tokenizer_encode), 'tokenizer_encode must be passed to DeclarativeAttentionWrapper to pass string prompts'
            tokens = self.tokenizer_encode(prompts)
            prompts = torch.as_tensor(tokens)[None]

        elif isinstance(prompts, list):
            if len(prompts) == 0:
                prompts = torch.empty((0, 0), dtype = torch.long)
            elif all(isinstance(x, int) for x in prompts):
                prompts = torch.tensor([prompts])
            elif all(isinstance(x, str) for x in prompts):
                assert exists(self.tokenizer_encode), 'tokenizer_encode must be passed to DeclarativeAttentionWrapper to pass string prompts'
                prompts = [torch.as_tensor(self.tokenizer_encode(p)) for p in prompts]
                prompts = pad_sequence(prompts, pad_value = self.pad_value)
            elif all(isinstance(x, (list, tuple)) for x in prompts):
                prompts = [torch.as_tensor(p) for p in prompts]
                prompts = pad_sequence(prompts, pad_value = self.pad_value)
            elif all(isinstance(x, Tensor) for x in prompts):
                prompts = pad_sequence(prompts, pad_value = self.pad_value)

        elif isinstance(prompts, Tensor) and prompts.ndim == 1:
            prompts = prompts[None]

        prompts, ps = pack([prompts], '* n')
        batch = prompts.shape[0]

        machines = state_machine

        if exists(machines):
            if isinstance(machines, (list, tuple)):
                assert len(machines) == batch
                machines = list(machines)
            else:
                machines = [copy.deepcopy(machines) for _ in range(batch)] if batch > 1 else [machines]

            if exists(strict):
                for machine in machines:
                    machine.strict = strict

        elif exists(chunk_spans):
            spans = list(chunk_spans) if is_batched_spans(chunk_spans) else [chunk_spans] * batch
            machines = [
                DeclarativeAttention(
                    s,
                    tokenizer_decode = self.tokenizer_decode,
                    strict = default(strict, self.strict)
                )
                for s in spans
            ]

        out = prompts
        t = prompts.shape[-1]
        cache = None
        sample = None
        is_eos_tokens = None

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

            if step_kwargs.pop('stop', False) and all(m.state.get('stop', False) for m in machines):
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
            sample = gumbel_sample(filtered_logits, temperature = temps)

            out = torch.cat((out, sample), dim = -1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    break

        if exists(eos_token) and exists(is_eos_tokens):
            shifted = F.pad(is_eos_tokens, (1, -1))
            mask = shifted.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        out = out[:, t:]
        out, = unpack(out, ps, '* n')
        return out
