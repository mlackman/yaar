from typing import Sequence
import os
import pathlib
import subprocess

from pydantic_ai import RunContext, AbstractToolset, FunctionToolset, Tool

from ..session import Deps
from ..runner import run_agent


def all_tools() -> Sequence[AbstractToolset[Deps|None]]:
    return [
        FunctionToolset(
            tools=[
                Tool(call_sub_agent, takes_ctx=True),
                Tool(write_to_file, takes_ctx=False),
                Tool(insert_to_file, takes_ctx=False),
                Tool(read_file, takes_ctx=False),
                Tool(read_lines, takes_ctx=False),
                Tool(ask_question, takes_ctx=False),
                Tool(shell, takes_ctx=False),
                Tool(gradle, takes_ctx=False)
            ]
        )
    ]

async def shell(cmd: list[str], cwd: str | None=None) -> tuple[str, str, int]:
    """
    Run cmd in shell.
    Example: stdout_str, stderr_str, error_code = shell(['ls', 'ls'])
    Args:
        cmd - list[str], command to be run
        cwd - optional str, working directory where to run the command
    Returns:
        stdout, stderr, exit code
        tuple[str, str, int]
    """
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    return r.stdout, r.stderr, r.returncode

async def gradle(gradle_cmd: str, cwd: str | None=None) -> tuple[int, str]:
    """
    Run gradle command.
    Example: gradle('build')
    Args:
        gradle_cmd: str, gradle command like build, test
        cwd - optional str, directory where to run the command
    Returns:
        exitcode, error text
        tuple[int, str]
    """
    _, error, error_code = await shell(cmd=['./gradlew', gradle_cmd], cwd=cwd)
    breakpoint()
    return error_code, error



id = 0
async def call_sub_agent(ctx: RunContext[Deps], name: str, prompt: str):
    """
    Call sub-agent for executing a task 
    Args:
        name: Sub-agent name
        prompt: Prompt for sub agent
    """
    global id
    id += 1

    the_id = int(id)

    main_session = ctx.deps.session
    sub_agents = ctx.deps.sub_agents

    the_agent = [sa for sa in sub_agents if sa.name == name]
    if the_agent == []:
        raise RuntimeError(f'No sub agent found with name "{name}"')
    the_agent = the_agent[0]

    main_session.logging.output((f'\n***** STARTING SUB-AGENT: {name}, id = {the_id} *****'))
    agent_name = f'{the_id}_{name}'

    session = main_session.sub_session(agent_name)
    session.save_prompt(the_agent.system_prompt, prompt)

    agent = the_agent.create()
        
    with session.logging as output:
        r = await run_agent(
            agent,
            prompt,
            output,
            message_history=None,
            deps=ctx.deps
        )
        session.save_response(r)
        output.output(f'Session id: {session.session_id}')

        main_session.logging.output((f'\n**** {r.response.usage()} *****'))
        main_session.logging.output((f'\n**** SUB-AGENT DONE: {name}, id = {the_id} *****'))

    return r.text


async def write_to_file(filename: str, data: str, mode: str='at'):
    """
    Writes to a file. If mode not given then by default appends to the file.

    Args:
        filename - must be full filename, string type
        data - Data to be written to file, string type
        mode - Default 'at'. file mode, like 'wt', 'at'
"""
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode=mode) as f:
        f.write(data)


async def insert_to_file(filename: str, line_no: int, data: str) -> None:
    """
    Inserts data to a file. If the line already has content then data will be inserted before the existing data in line. 
    Adding line break to the data allows inserting lines.

    Args:
        filename - string, must be absolute filename
        line_no - integer, line number, where data is inserted to. If line_no is -1 then data is appended to file. Line_no 1 is first line
        data - string, data to be written to the file
    """
    assert line_no >= -1, 'line_no parameter must be -1 or larger'
    line_no = line_no if line_no == -1 else line_no - 1
    path = pathlib.Path(filename)
    with path.open('rt') as f:
        content = f.readlines()
        assert line_no < len(content), f'File has only {len(content)} lines, trying to insert data to line number {line_no}' 
        if line_no != -1:
            content[line_no] = data + content[line_no]
        else:
            content[-1] = content[-1] + data

        with path.open('wt') as wf:
            wf.write(''.join(content))


async def read_file(filename: str, include_line_numbers: bool = False) -> str | None:
    """
    Reads the whole file and returns the content of the file
    Args:
        filename - must be full filename
        include_line_numbers - boolean, default false. If set to true, line number is appended to each line like `0:`
    Returns:
        Content of the file or None if the file does not exists
    """
    try:
        with open(filename, 'rt') as f:
            if include_line_numbers:
                return ''.join(_read_lines_with_line_numbers(f))
            else:
                return f.read()
    except FileNotFoundError:
        return None


async def read_lines(filename: str, start: int, end: int, include_line_numbers: bool =  False) -> str | None:
    """
    Reads the lines from file and returns the lines
    Args:
        filename - must be full filename
        start: int, read from this line. Line 1 is the first line
        end: int, read to this line, exclusive
        include_line_numbers - boolean, default false. If set to true, line number is appended to each line like `0:`
    Returns:
        Lines read or None if the file does not exists
    """
    try:
        with open(filename, 'rt') as f:
            if include_line_numbers:
                lines = _read_lines_with_line_numbers(f)
            else:
                lines = f.readlines()
            return ''.join(lines[start-1:end-1])
    except FileNotFoundError:
        return None


def _read_lines_with_line_numbers(f) -> list[str]:
    return [f'{i+1}:{line}' for i, line in enumerate(f.readlines())]


Answer = str
Question = str

def ask_question(questions: list[str]) -> list[tuple[Question, Answer]]:
    """
    Asks questions from user
    Args:
        questions - list of questions (str)
    Returns:
        List of question and answers tuples (list[(question, answer)])
    """
    prompt = f'{os.linesep}***** QUESTION ****{os.linesep}'
    return [(question, input(f'{prompt}{question}')) for question in questions]
