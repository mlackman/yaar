from typing import Sequence, Callable
import dataclasses
import datetime
import enum
import io
import random
import pathlib
import json

import pydantic_ai
from pydantic_core import to_jsonable_python
from pydantic_ai import AbstractToolset, BaseToolCallPart, RunUsage, AgentRunResult
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings 
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.test import TestModel

type LoggingFactory = Callable[['Session'], Logging]

@dataclasses.dataclass(frozen=True)
class Response:
    text: str  # the textual response of the agent
    response: AgentRunResult


class Model(enum.Enum):
    TEST=0
    GPT_54_THINKING=1
    GPT_55=2
    GEMINI_31_FLASH_LITE=3


    @classmethod
    def create(cls, model: 'Model', api_key: str) -> pydantic_ai.models.Model:
        openai_settings = OpenAIResponsesModelSettings(
            openai_reasoning_effort='high',
            openai_builtin_tools=[],
            openai_text_verbosity='medium',
            openai_reasoning_generate_summary='detailed',
        )

        if model == Model.TEST:
            return TestModel()

        if model == Model.GPT_54_THINKING:
            return OpenAIResponsesModel(
                model_name='gpt-5.4',
                provider=OpenAIProvider(api_key=api_key),
                settings=openai_settings
            )
        elif model == Model.GPT_55:
            return OpenAIResponsesModel(
                model_name='gpt-5.5',
                provider=OpenAIProvider(api_key=api_key),
                settings=openai_settings
            )
        elif model == Model.GEMINI_31_FLASH_LITE:
            return GoogleModel(
                model_name='gemini-3.1-flash-lite',
                provider=GoogleProvider(api_key=api_key),
            )


@dataclasses.dataclass
class Agent:
    name: str
    system_prompt: str
    toolsets: Sequence[AbstractToolset]
    model: Model
    description: str
    api_key: str

    def create(self) -> pydantic_ai.Agent:
        return pydantic_ai.Agent(
            Model.create(self.model, api_key=self.api_key),
            name = self.name,
            toolsets=self.toolsets,
            system_prompt=self.system_prompt,
            end_strategy='exhaustive'
        )
        
class Logging:

    def __init__(self, session: 'Session'):
        self._session = session
        self._debug_file: io.TextIOWrapper | None = None
        self._session_file: io.TextIOWrapper | None = None

    def __enter__(self) -> 'Logging':
        self._debug_file = self._session.debug_log.open('wt')
        self._session_file = self._session.session_log.open('wt')
        return self

    def __exit__(self, *args, **kwargs) -> None:
        assert self._debug_file is not None
        assert self._session_file is not None
        self._debug_file.close()
        self._debug_file.close()

    def debug(self, s: str) -> None:
        print(s, file=self._debug_file, flush=True)

    def output(self, s: str) -> None:
        print(s, file=self._session_file, end='', flush=True)

    def tool_usage(self, tool: BaseToolCallPart) -> None:
        self.output(f'\nUSING TOOL: {tool.tool_name}, args {tool.args_as_dict()}\n')

    def usage(self, usage: RunUsage) -> None:
        self.output(f'\nTOKEN USAGE: {usage}\n')


class LoggingOutputToStd(Logging):
    def output(self, s: str) -> None:
        super().output(s)
        print(s, end='\n', flush=True)


class Session:

    @staticmethod
    def create_main_session(session_name: str, path: pathlib.Path, logging_factory: LoggingFactory, previous_session: str | None = None) -> 'Session':
        session_id = f'{session_name}_{datetime.datetime.now().isoformat()}_{random.randint(1000, 100000)}'
        session_path = pathlib.Path(path / session_id)
        session_path.mkdir(parents=True)
        return Session(session_name, session_path, session_id, logging_factory, previous_session)

    def __init__(self, session_name: str, path: pathlib.Path, session_id: str, logging_factory: LoggingFactory, previous_session: str | None = None):
        self.session_name = session_name 
        self.session_id = session_id
        self.path = path
        self.session_log = self.path / 'session.log' 
        self.debug_log = self.path / 'debug.log'
        self.logging = logging_factory(self)
        self.previous_session = previous_session

    def sub_session(self, session_name: str) -> 'Session':
        path = self.path / f'{session_name}'
        path.mkdir(parents=True)
        return Session(session_name, path, self.session_id, lambda session: self._sub_session_logging_factory(session))

    def _sub_session_logging_factory(self, session: 'Session'):
        return Logging(session)

    def save_prompt(self, system_prompt: str, prompt: str) -> None:
        with open(self.path / f'{self.session_name}.prompt', 'wt') as f:
            f.write(system_prompt)
            f.write('\n')
            f.write(prompt)

    def save_response(self, response: Response) -> None:
        with open(self.path / 'messages.json', 'wt') as f:
            json.dump(to_jsonable_python(response.response.all_messages()), f)
        with open(self.path / 'result', 'wt') as f:
            f.write(response.text)


@dataclasses.dataclass(frozen=True)
class Deps:
    session: Session
    sub_agents: list[Agent]
