from typing import Callable

import dataclasses
import io
import pathlib

from pydantic_ai import BaseToolCallPart, RunUsage


@dataclasses.dataclass(frozen=True)
class LogDestinations:
    debug_log: pathlib.Path
    session_log: pathlib.Path

type LoggingFactory = Callable[[LogDestinations], Logging]


class Logging:

    def __init__(self, log_destinations: LogDestinations):
        self._log_destinations = log_destinations 
        self._debug_file: io.TextIOWrapper | None = None
        self._session_file: io.TextIOWrapper | None = None

    def __enter__(self) -> 'Logging':
        self._debug_file = self._log_destinations.debug_log.open('wt')
        self._session_file = self._log_destinations.session_log.open('wt')
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
