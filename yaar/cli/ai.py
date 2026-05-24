import sys 
import os
import asyncio

from yaar.models import Agent, Model 
from yaar.tools import all_tools
from .session import session
from .subagents import all_sub_agents
from yaar.session import start_agent_with_session


def main():
    api_key = os.getenv('API_KEY')
    assert api_key is not None, 'API_KEY environment variable not found'
    prompt_filename = sys.argv[1] if len(sys.argv) > 1 else None

    previous_session = sys.argv[2] if len(sys.argv) > 2 else None 

    if prompt_filename:
        with open(prompt_filename, 'rt') as f:
            prompt = f.read()
    else:
        prompt = input('Prompt: ')

    system_prompt = '''
# Project
    - root {os.getcwd()}
    '''

    main_agent = Agent(
        name='Generic-ai', 
        model=Model.GEMINI_31_FLASH_LITE,
        system_prompt=system_prompt,
        description='generic llm ai',
        toolsets=[*all_tools()],
        api_key=api_key
    )


    cli_session = session(session_name=main_agent.name, previous_session=previous_session)
    asyncio.run(
        start_agent_with_session(
            session=cli_session,
            prompt=prompt, 
            agent=main_agent,
            sub_agents=all_sub_agents(system_prompt=system_prompt, session=cli_session, api_key=api_key),
            previous_session=previous_session
        )
    )


if __name__ == "__main__":
    main()
