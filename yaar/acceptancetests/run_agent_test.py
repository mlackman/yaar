import pathlib 
import shutil 

import pytest

from yaar.models import Agent, Model
from yaar import create_session, load_session
from yaar.session import Session, start_agent_with_session
from yaar.logging import LoggingOutputToStd

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

    with session.log_destinations.debug_log.open('rt') as f:
        assert len(f.readlines()) == 7

    with session.log_destinations.session_log.open('rt') as f:
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
    test_root = pathlib.Path('./.yaar-test')
    shutil.rmtree(test_root, ignore_errors=True)

    main_agent = Agent(
        name='Generic-test-ai',
        model=Model.TEST,
        system_prompt='the system prompt',
        description='generic llm ai',
        toolsets=[],
        api_key='test key'
    )

    session_root_path = test_root / 'session'

    session = create_session(
        name='feature-x',
        root_path = session_root_path,
        logging_factory=lambda dests: LoggingOutputToStd(dests),
        chat_name_provider=lambda: 'exec_1_1'
    )
    session_path = session_root_path / 'feature-x'

    response = await session.run_agent(
        prompt='prompt',
        agent=main_agent,
        sub_agents=[]
    )
    
    assert response.text == 'success (no tool calls)'
    assert_session_logs(session_path / 'exec_1_1')

    # Continue the session

    session = load_session(
        name='feature-x',
        root_path=session_root_path,
        logging_factory=lambda dests: LoggingOutputToStd(dests),
        chat_name_provider=lambda: 'exec_1_2'
    )
    response = await session.run_agent(
        prompt='prompt',
        agent=main_agent,
        sub_agents=[]
    )
    assert response.text == 'success (no tool calls)'
    assert_session_logs(session_path / 'exec_1_2')




def assert_session_logs(exec_path: pathlib.Path):
    with (exec_path / 'debug.log').open('rt') as f:
        assert len(f.readlines()) == 7
    with (exec_path / 'session.log').open('rt') as f:
        assert len(f.readlines()) == 4
    with (exec_path / 'messages.json').open('rt') as f:
        assert len(f.readlines()) == 1
    with (exec_path / 'prompt').open('rt') as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert lines[0] == 'the system prompt\n'
        assert lines[1] == 'prompt'
    with (exec_path / 'result').open('rt') as f: 
        assert f.read() == 'success (no tool calls)'



# TODO: Can see conversations from session, which agent etc.
# TODO: Can attach to conversation and continue from there. Same agent or some other agent
