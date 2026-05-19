from yaar.models import Agent, Model, Session
from yaar.tools import all_tools
from yaar.prompts import load_prompt
from yaar.agent import create_mcps


def all_sub_agents(system_prompt: str, session: Session, api_key: str) -> list[Agent]:
    return [
        _create_sub_agent(
            agent_name='codebase-analyzer',
            system_prompt=system_prompt,
            session=session,
            api_key=api_key
        ),
        _create_sub_agent(
            agent_name='codebase-locator',
            system_prompt=system_prompt,
            session=session,
            api_key=api_key
        ),
        _create_sub_agent(
            agent_name='codebase-pattern-finder',
            system_prompt=system_prompt,
            session=session,
            api_key=api_key
        )
    ]



def _create_sub_agent(agent_name: str, system_prompt: str, session: Session, api_key: str) -> Agent:
    try:
        agent_prompt = load_prompt(f'agents/{agent_name}.md')
    except FileNotFoundError:
        session.logging.output(f'\n***** SUB-AGENT "{agent_name}" prompt not found *****')
        agent_prompt = system_prompt

                
    tools = [*create_mcps(), *all_tools()]
    return Agent(
        name=agent_name,
        model=Model.GPT_54_THINKING, 
        system_prompt=f'{system_prompt}\n{agent_prompt}',
        description='Code research agent',
        toolsets=tools,        
        api_key=api_key
    )
