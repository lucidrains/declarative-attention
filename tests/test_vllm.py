import torch

from declarative_attention import DeclarativeVLLMHook

def test_block_table_rewritten_per_mode():
    hook = DeclarativeVLLMHook(block_size = 16)
    hook.register_request('req', [(16, 48), (48, 80), (80, 112), (112, 144)])

    # 170 tokens = 11 blocks, logical block -> physical block
    block_tables = torch.tensor([[42, 55, 61, 73, 84, 92, 101, 115, 128, 137, 201]])
    seq_lens = torch.tensor([170])

    # global: every block stays visible
    new_block_tables, _ = hook(block_tables, seq_lens, ['req'])
    assert new_block_tables[0].tolist() == block_tables[0].tolist()

    # focus on chunk 2: sink, chunk 2, question, response
    hook.step('req', '<focus magic_chunks="2">')
    new_block_tables, new_seq_lens = hook(block_tables, seq_lens, ['req'])
    assert new_block_tables[0, :5].tolist() == [42, 73, 84, 137, 201]
    assert new_seq_lens[0] == 5 * 16

    # local: no chunks
    hook.step('req', '</focus><local>')
    new_block_tables, _ = hook(block_tables, seq_lens, ['req'])
    assert new_block_tables[0, :3].tolist() == [42, 137, 201]

def test_hook_streams_tag_pieces():
    vocab = {10: '<fo', 11: 'cus magic_chunks="1"', 12: '>'}
    hook = DeclarativeVLLMHook(block_size = 16, tokenizer_decode = vocab.get)
    hook.register_request('req', [(16, 48), (48, 80)])

    block_tables = torch.tensor([[3, 7, 12, 19, 21, 25]])
    seq_lens = torch.tensor([96])

    def kept_blocks():
        return hook(block_tables, seq_lens, ['req'])[0][0, :4].tolist()

    # until the tag closes, the full cache stays visible
    hook.step('req', 10)
    hook.step('req', 11)
    assert kept_blocks() == [3, 7, 12, 19]

    hook.step('req', 12)
    assert kept_blocks() == [3, 7, 12, 25]

def test_untracked_requests_pass_through():
    hook = DeclarativeVLLMHook(block_size = 16)
    block_tables = torch.tensor([[1, 2, 3, 4]])
    seq_lens = torch.tensor([64])

    new_block_tables, new_seq_lens = hook(block_tables, seq_lens, ['unknown'])
    assert new_block_tables.tolist() == block_tables.tolist()
    assert new_seq_lens.tolist() == seq_lens.tolist()

def test_vllm_two_vs_one_parity():
    hook = DeclarativeVLLMHook(block_size = 16)
    hook.register_request('req1', [(16, 48), (48, 80)])
    hook.register_request('req2', [(32, 64)])

    hook.step('req1', '<focus magic_chunks="1">')
    hook.step('req2', '<local>')

    block_tables = torch.tensor([
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23, 24, 25]
    ])
    seq_lens = torch.tensor([96, 96])

    # single request 1
    bt1, sl1 = hook(block_tables[0], seq_lens[0], 'req1')

    # single request 2
    bt2, sl2 = hook(block_tables[1], seq_lens[1], 'req2')

    # batch of 2
    bt_batch, sl_batch = hook(block_tables, seq_lens, ['req1', 'req2'])

    assert torch.equal(bt_batch[0], bt1)
    assert torch.equal(bt_batch[1], bt2)
    assert sl_batch[0] == sl1
    assert sl_batch[1] == sl2
