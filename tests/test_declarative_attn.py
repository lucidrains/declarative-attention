import torch
from declarative_attention import DeclarativeStateMachine

def test_declarative_state_machine():
    state_machine = DeclarativeStateMachine({1: (16, 32), 2: (32, 48)})
    assert state_machine.get_mask(64).all()

    state_machine.step('<focus chunks="1">')
    assert state_machine.is_focus and state_machine.active_chunks == {1}
    assert not state_machine.get_mask(64)[32:48].any()

    state_machine.step('</focus>')
    assert state_machine.is_global and state_machine.get_mask(64).all()

    state_machine.step('<local>')
    assert state_machine.is_local and not state_machine.get_mask(64)[16:48].any()

    state_machine.step('</local>')
    assert state_machine.is_global and state_machine.get_mask(64).all()

def test_subword_token_streaming():
    state_machine = DeclarativeStateMachine([(16, 32), (32, 48)])
    assert state_machine.is_global and not state_machine.is_streaming_tag

    # Stream partial subword tokens: <f, o, c, us chunks="1", >
    subwords = ['<f', 'o', 'c', 'us chunks="1"', '>']

    for token in subwords[:-1]:
        state_machine.step(token)
        assert state_machine.is_global, "Should stay in global until tag completes on '>'"
        assert state_machine.is_streaming_tag, "Should track that a tag is mid-stream"

    # Closing '>' completes the tag and executes the transition
    state_machine.step(subwords[-1])
    assert state_machine.is_focus and state_machine.active_chunks == {1}
    assert not state_machine.is_streaming_tag

def test_custom_tokenizer_decode_with_torch_long():
    vocab = {10: '<local>', 11: '</local>'}
    state_machine = DeclarativeStateMachine([(16, 32)], tokenizer_decode = lambda t: vocab.get(t, ''))

    # Feed step as torch long scalar
    state_machine.step(torch.tensor(10, dtype = torch.long))
    assert state_machine.is_local

    state_machine.step(torch.tensor(11, dtype = torch.long))
    assert state_machine.is_global

def test_custom_researcher_logic():
    state_machine = DeclarativeStateMachine([(16, 32), (32, 48)])

    @state_machine.on('exact')
    def handle_exact(machine):
        machine.temperature = 0.0

    state_machine.step('<exact>')
    assert state_machine.temperature == 0.0

def test_dynamic_temperature_and_verification():
    state_machine = DeclarativeStateMachine([(16, 32), (32, 48)])
    state_machine.temperature = 0.7

    @state_machine.on('explore')
    def on_explore(machine, temp = 1.2):
        machine.temperature = float(temp)

    @state_machine.on('verify')
    def on_verify(machine, temp = 0.0):
        machine.temperature = float(temp)

    # 1. High temperature exploration
    state_machine.step('<explore temp="1.5">')
    assert state_machine.temperature == 1.5

    # 2. Switch to zero-temperature verification while focusing on Chunk 1
    state_machine.step('</explore><focus chunks="1"><verify temp="0.0">')
    assert state_machine.temperature == 0.0
    assert state_machine.is_focus and state_machine.active_chunks == {1}
    assert not state_machine.get_mask(64)[32:48].any()
