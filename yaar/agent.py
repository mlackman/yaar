from typing import Callable, Any
import pathlib
import json

from pydantic_ai.mcp import MCPServerStdio 
from pydantic_ai.messages import ModelMessagesTypeAdapter

from .runner import Response, run_agent
from .models import Session, Agent, Deps


type MCPFactory = Callable[[], list[MCPServerStdio]]


def create_mcps() -> list[MCPServerStdio]:
    desktop_commander = MCPServerStdio(
        timeout=10,
        command="npx",
        args=["-y", "@wonderwhy-er/desktop-commander@latest", "--no-onboarding"],
        tool_prefix="desktop_commander",
        allow_sampling=False,
    )
    return [desktop_commander]


async def start_agent_with_session(
    prompt: str,
    agent: Agent,
    sub_agents: list[Agent],
    session: Session, 
    previous_session: str | None = None
) -> Response:

    message_history = _load_message_history(previous_session) if previous_session is not None else None

    pydantic_agent = agent.create() 
    session.save_prompt(agent.system_prompt, prompt)

    with session.logging as output:
        result = await run_agent(
            pydantic_agent,
            prompt,
            output,
            message_history=message_history,
            deps=Deps(session=session, sub_agents=sub_agents) 
        )

        # Write the conversation history. This can be used for continuing the session
        session.save_response(result)
        output.output(f'\nSession id: {session.session_id}')
        return result


def _load_message_history(previous_session: str) -> Any:
    previous_messages = pathlib.Path(f'./.session/{previous_session}/messages.json') 

    with(previous_messages.open('rt')) as f:
        return ModelMessagesTypeAdapter.validate_python(json.load(f))
