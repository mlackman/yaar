import dataclasses
import datetime
import json
import random

from pathlib import Path
from typing import Callable, Any

from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelMessagesTypeAdapter

from .logging import LoggingFactory, LogDestinations, Logging
from .models import Agent
from .runner import run_agent, Response

type ChatSessionNameProvider = Callable[[], str]

def _chat_name_generator() -> str:
    return f'exec_{datetime.datetime.now().isoformat()}_{random.randint(1000, 100000)}'

def create_session(
    name: str, 
    root_path: Path | str, 
    logging_factory: LoggingFactory, 
    chat_name_provider: ChatSessionNameProvider | None = None
) -> 'Session':
    """
    Conversations session.
    """
    session_root_path = Path(root_path) / name
    exec_id = chat_name_provider() if chat_name_provider else _chat_name_generator()
    session_path = Path(session_root_path / exec_id)
    session_path.mkdir(parents=True)
    return Session(session_name=name, path=session_path, session_id=exec_id, logging_factory=logging_factory, previous_session=None)   


def load_session(
    name: str, 
    root_path: Path | str, 
    logging_factory: LoggingFactory, 
    chat_name_provider: ChatSessionNameProvider | None = None
) -> 'Session':
    session_root_path = Path(root_path) / name
    exec_id = chat_name_provider() if chat_name_provider else _chat_name_generator()
    session_path = Path(session_root_path / exec_id)
    session_path.mkdir()
    return Session(session_name=name, path=session_path, session_id=exec_id, logging_factory=logging_factory, previous_session=None)


class Session:

    @staticmethod
    def create_main_session(session_name: str, path: Path, logging_factory: LoggingFactory, previous_session: str | None = None) -> 'Session':
        session_id = f'{session_name}_{datetime.datetime.now().isoformat()}_{random.randint(1000, 100000)}'
        session_path = Path(path / session_id)
        session_path.mkdir(parents=True)
        return Session(session_name, session_path, session_id, logging_factory, previous_session)

    def __init__(self, session_name: str, path: Path, session_id: str, logging_factory: LoggingFactory, previous_session: str | None = None):
        self.session_name = session_name 
        self.session_id = session_id
        self.path = path
        self.log_destinations = LogDestinations(
            debug_log = self.path / 'debug.log',
            session_log = self.path / 'session.log' 
        )
        self.logging = logging_factory(self.log_destinations)
        self.previous_session = previous_session

    async def run_agent(self, prompt: str, agent: Agent, sub_agents: list[Agent]) -> Response:
        return await start_agent_with_session(
            prompt=prompt,
            agent=agent, 
            sub_agents=sub_agents,
            session=self,
            previous_session=None
        )

    def sub_session(self, session_name: str) -> 'Session':
        path = self.path / f'{session_name}'
        path.mkdir(parents=True)

        return Session(session_name, path, self.session_id, lambda dests: self._sub_session_logging_factory(dests))

    def _sub_session_logging_factory(self, log_destinations: LogDestinations):
        return Logging(log_destinations)

    def save_prompt(self, system_prompt: str, prompt: str) -> None:
        with open(self.path / 'prompt', 'wt') as f:
            f.write(system_prompt)
            f.write('\n')
            f.write(prompt)

    def save_response(self, response: Response) -> None:
        with open(self.path / 'messages.json', 'wt') as f:
            json.dump(to_jsonable_python(response.response.all_messages()), f)
        with open(self.path / 'result', 'wt') as f:
            f.write(response.text)


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
    previous_messages = Path(f'./.session/{previous_session}/messages.json') 

    with(previous_messages.open('rt')) as f:
        return ModelMessagesTypeAdapter.validate_python(json.load(f))

@dataclasses.dataclass(frozen=True)
class Deps:
    session: Session
    sub_agents: list[Agent]
