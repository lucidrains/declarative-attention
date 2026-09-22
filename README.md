<img src="./fig1.png" width="400px"></img>

## Declarative Attention - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2609.02737">Language Models Can Control Their Own Attention</a>, from Namgyu Ho et al. of KAIST AI

Normally, a language model attends to the entire prompt KV cache at every decoding step, even when the reasoning only needs a fraction of the text.

**Declarative Attention** lets the model declare what it needs to attend to in its output stream:

- `<global>` - all chunks visible (default)
- `<focus magic_chunks="1, 3">` - only the selected chunks visible
- `<local>` - no chunks visible; the model reasons using only its own thoughts and instructions

The full KV cache stays resident in memory, so masking is completely reversible, with zero cache eviction.

## Install

```bash
$ pip install declarative-attention
```

## Quickstart

Run directly in a fresh Python session (character offsets if tokenizer is omitted, or pass your tokenizer):

```python
import torch
from declarative_attention import DeclarativeAttention, extract_chunk_spans

prompt = """
Question: Where did the meeting take place?

<chunk id="1">
Acme Corp was founded in 2003 in Seattle.
</chunk>
<chunk id="2">
The annual shareholder meeting was held in Chicago in 2011.
</chunk>
"""

# 1. extract chunk spans (omit tokenizer for char offsets, or pass tokenizer_encode = tokenizer.encode)

chunk_spans = extract_chunk_spans(prompt)

declare_attn = DeclarativeAttention(chunk_spans)
assert declare_attn.mode == 'global' # all chunks visible by default

# 2. focus on specific chunks while reading facts

declare_attn.step('<focus magic_chunks="2">')

assert declare_attn.mode == 'focus'
assert declare_attn.active_chunks == {2}

mask = declare_attn(total_len = 8192)        # 1d boolean mask over the KV cache (True = attended)
kept = declare_attn.get_kept_blocks(total_len = 8192) # block indices for paged attention (vLLM)

# 3. switch to local mode to reason without reading any context chunks

declare_attn.step('<local>')

assert declare_attn.mode == 'local'
assert declare_attn.active_chunks == set()

# in local mode, all context chunks are masked out
# only the instructions and the model's own response so far stay attended

mask = declare_attn(total_len = 8192)
kept = declare_attn.get_kept_blocks(total_len = 8192)

# 4. closing the tag restores the previous state

declare_attn.step('</local>')

assert declare_attn.mode == 'focus'
assert declare_attn.active_chunks == {2}

# closing focus, or switching to <global>, returns to all chunks visible

declare_attn.step('</focus>')

assert declare_attn.mode == 'global'
```

The state machine tracks active chunks and unwinds nested tags on close. The `mode` (`global` / `focus` / `local`) is derived automatically from the active chunks.

> **Flexible Tag Attributes:** Models don't need to be rigid. `<focus magic_chunks="1, 3">`, `<focus chunks="1, 3">`, `<focus chunk="2">`, or even bare `<focus 1, 3>` all work - attributes are normalized and chunk ids extracted automatically.

## Minimal PyTorch Drop-in

To use declarative attention in any existing autoregressive generation loop:

```python
declare_attn = DeclarativeAttention(chunk_spans, tokenizer_decode = tokenizer.decode)

for _ in range(max_new_tokens):
    # derive 1d boolean mask over KV cache (True = attended)
    kv_mask = declare_attn(kv_cache_len)

    # apply to attention scores:
    # attn_scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    # attn_scores = attn_scores.masked_fill(~kv_mask, -torch.finfo(attn_scores.dtype).max)

    # step the machine with the sampled token id (or string)
    declare_attn.step(sampled_token_id)
```

## Prompt Construction

The context is split into `~2048`-token chunks ("magic chunks") and presented to the model with that label. The segmenter splits only text that exceeds the limit, breaking cleanly (paragraphs, sentences, words) without losing any characters:

```python
from declarative_attention import segment_context, format_declarative_prompt

segments = segment_context(context, tokenizer_encode = tokenizer.encode)

prompt, chunk_spans = format_declarative_prompt(
    question = "How long after its founding did Acme Corp go public?",
    context = context,
    tokenizer_encode = tokenizer.encode
)
```

`format_declarative_prompt` returns token ids and token-index spans when given a `tokenizer_encode`, or text and character offsets when omitted.

## Custom State Machines

Any tag can modify any part of the decode step. Active chunks and `declare_attn.state` are automatically saved when a tag opens and restored when it closes:

```python
from declarative_attention import DeclarativeAttention, parse_chunk_ids

declare_attn = DeclarativeAttention(chunk_spans, tokenizer_decode = tokenizer.decode)

# 1. compare multiple chunks together

@declare_attn.on('compare')
def compare(declare_attn, attrs):
    declare_attn.active_chunks = parse_chunk_ids(attrs['chunks'])

# 2. transcribe verbatim - isolate one chunk and decode greedily

@declare_attn.on('verbatim')
def verbatim(declare_attn, attrs):
    declare_attn.active_chunks = {int(attrs['chunk'])}
    declare_attn.state['temperature'] = 0.

# 3. brainstorm - think locally without reading context, with higher temperature

@declare_attn.on('brainstorm')
def brainstorm(declare_attn, attrs):
    declare_attn.active_chunks = set()
    declare_attn.state['temperature'] = 1.5

# 4. force MoE routing to specific experts

@declare_attn.on('expert')
def expert(declare_attn, attrs):
    declare_attn.state['force_expert_ids'] = parse_chunk_ids(attrs['ids'])
```

The model can now declare:

```xml
<compare chunks="1, 3">
Contrasting filing 1 with filing 3...
</compare>

<verbatim chunk="2">
"Exact legal text transcribed without hallucinations."
</verbatim>

<brainstorm>
Synthesizing new hypotheses without context distractions...
</brainstorm>
```

Tell the model about these tags by passing matching `instructions` when building the prompt:

```python
prompt, chunk_spans = format_declarative_prompt(
    question,
    context,
    instructions = "Answer using the magic chunks. Use <compare chunks=\"K,M\"> to view several at once, <verbatim chunk=\"K\"> to transcribe one exactly, and <brainstorm> to reason with no chunks visible.",
    tokenizer_encode = tokenizer.encode
)
```

A reusable protocol can instead subclass `DeclarativeAttention` and override `instructions`, so `MyMachine.format_prompt(...)` builds the matching prompt.

For a state machine that is not about attention at all, subclass `TagStateMachine` and implement `get_mask` however you like. The only contract the wrappers rely on is `step`, `get_mask`, and `state`.

## vLLM

vLLM reads whole KV blocks, so the token mask is rounded outward to block boundaries and each request's block table is rewritten to contain only the kept blocks. The attention kernel simply reads less, with zero kernel modifications:

```python
from declarative_attention import DeclarativeVLLMHook

hook = DeclarativeVLLMHook(block_size = 16, tokenizer_decode = tokenizer.decode)
hook.register(request_id, chunk_spans)

# on every sampled token
hook.step(request_id, token)

# hook onto the attention metadata builder
block_tables, seq_lens = hook(block_tables, seq_lens, request_ids)
```

## With `x-transformers`

Training derives the 2d causal DA mask from the full sequence. Generation streams sampled tokens through the state machine and passes the per-step KV mask to the model:

```python
import torch
from x_transformers import TransformerWrapper, Decoder
from declarative_attention import DeclarativeAttentionWrapper

net = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = 1024,
    attn_layers = Decoder(dim = 64, depth = 2, heads = 4)
)

wrapper = DeclarativeAttentionWrapper(net)

# 1. training, with the 2d causal declarative attention mask

chunk_spans = [(64, 128), (128, 192)]
x = torch.randint(0, 256, (2, 512))

loss = wrapper(x, chunk_spans = chunk_spans, prompt_len = 64)
loss.backward()

# 2. generation, with dynamic per-step KV masking

prompt = torch.randint(0, 256, (1, 64))
out = wrapper.generate(prompt, seq_len = 256, chunk_spans = chunk_spans)
```

Only generated tokens drive the state machine. The DA instructions themselves mention tags, so pass `prompt_len` during training to keep the prompt in global mode.

## Citations

```bibtex
@misc{ho2026languagemodelscontrolattention,
    title   = {Language Models Can Control Their Own Attention}, 
    author  = {Namgyu Ho and Huzama Ahmad and Woosung Koh and Se-Young Yun and Tal Schuster and Cicero Nogueira dos Santos},
    year    = {2026},
    eprint  = {2609.02737},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/2609.02737}
}
```
