import pathlib 

import pytest

from yaar.models import Agent, Model, Session, LoggingOutputToStd
from yaar.agent import start_agent_with_session

@pytest.mark.asyncio
async def test():
    main_agent = Agent(
        name='Generic-test-ai',
        model=Model.TEST,
        system_prompt='the system prompt',
        description='generic llm ai',
        toolsets=[],
        api_key='test key'
    )


    session = Session.create_main_session(
        session_name='test-agent',
        path = pathlib.Path('./.session'),
        logging_factory=lambda session: LoggingOutputToStd(session)
    )

    response = await start_agent_with_session(
        prompt='prompt',
        agent=main_agent,
        sub_agents=[],
        session=session,
        previous_session=None
    )

    assert response.text == 'success (no tool calls)'

    with session.debug_log.open('rt') as f:
        assert len(f.readlines()) == 7

    with session.session_log.open('rt') as f:
        assert len(f.readlines()) == 4

    with (session.path / 'messages.json').open('rt') as f:
        assert len(f.readlines()) == 1

    with (session.path / 'prompt').open('rt') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert lines[0] == 'the system prompt\n'
        assert lines[1] == 'prompt'

    with (session.path / 'result').open('rt') as f: 
        assert f.read() == 'success (no tool calls)'


@pytest.mark.asyncio
async def test_continue_of_the_session():
    main_agent = Agent(
        name='Generic-test-ai',
        model=Model.TEST,
        system_prompt='the system prompt',
        description='generic llm ai',
        toolsets=[],
        api_key='test key'
    )


    session = Session.create_main_session(
        session_name='test-agent',
        path = pathlib.Path('./.session/feature-x'),
        logging_factory=lambda session: LoggingOutputToStd(session)
    )

    response = await start_agent_with_session(
        prompt='prompt',
        agent=main_agent,
        sub_agents=[],
        session=session,
        previous_session=None
    )

    assert response.text == 'success (no tool calls)'
    with session.debug_log.open('rt') as f:
        assert len(f.readlines()) == 7
    with session.session_log.open('rt') as f:
        assert len(f.readlines()) == 4
    with (session.path / 'messages.json').open('rt') as f:
        assert len(f.readlines()) == 1
    with (session.path / 'prompt').open('rt') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert lines[0] == 'the system prompt\n'
        assert lines[1] == 'prompt'
    with (session.path / 'result').open('rt') as f: 
        assert f.read() == 'success (no tool calls)'
