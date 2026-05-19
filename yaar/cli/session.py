import asyncio
import os
import pathlib
import sys

from yaar.models import Session, LoggingOutputToStd, Logging, Agent, Model
from yaar.prompts import load_prompt
from yaar.tools import all_tools 
from yaar.agent import start_agent_with_session, create_mcps
from .subagents import all_sub_agents


def run_agent(prompt_filename: str, agent_name: str, description: str):
    agent_system_prompt = load_prompt(prompt_filename)

    api_key = os.getenv('API_KEY')
    assert api_key is not None, 'API_KEY environment variable not found'
    prompt_filename = sys.argv[1]
    with open(prompt_filename, 'rt') as f:
        prompt = f.read()

    previous_session = sys.argv[2] if len(sys.argv) > 2 else None 

    system_prompt = f'''
    # Project
        - root {os.getcwd()}
    '''
    main_agent = Agent(
        name=agent_name, 
        model=Model.GPT_54_THINKING, 
        system_prompt=f'{system_prompt}\n{agent_system_prompt}',
        description=description,
        toolsets=[*create_mcps(), *all_tools()],        
        api_key=api_key
    )

    cli_session = session(session_name=main_agent.name, previous_session=previous_session)

    asyncio.run(
        start_agent_with_session(
            session=cli_session,
            prompt=prompt, 
            agent=main_agent,
            sub_agents=all_sub_agents(system_prompt=system_prompt, session=cli_session, api_key=api_key),
            previous_session=None
        )
    )


def session(session_name: str, previous_session: str | None) -> Session:

    def create_main_agent_loggin(session: Session) -> Logging:
        return LoggingOutputToStd(session)
    return Session.create_main_session(session_name, pathlib.Path('./.session/'), create_main_agent_loggin, previous_session)
