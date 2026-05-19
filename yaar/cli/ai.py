import sys 
import os
import asyncio

from yaar.models import Agent, Model 
from yaar.tools import all_tools
from .session import session
from .subagents import all_sub_agents
from yaar.agent import start_agent_with_session


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

    system_prompt = f'''
<!-- codebase-memory-mcp:start -->
# Codebase Knowledge Graph (codebase-memory-mcp)

This project uses codebase-memory-mcp to maintain a knowledge graph of the codebase.
ALWAYS prefer MCP graph tools over grep/glob/file-search for code discovery.

## Priority Order
1. `search_graph` — find functions, classes, routes, variables by pattern
2. `trace_path` — trace who calls a function or what it calls
3. `get_code_snippet` — read specific function/class source code
4. `query_graph` — run Cypher queries for complex patterns
5. `get_architecture` — high-level project summary

## When to fall back to grep/glob
- Searching for string literals, error messages, config values
- Searching non-code files (Dockerfiles, shell scripts, configs)
- When MCP tools return insufficient results

## Examples
- Find a handler: `search_graph(name_pattern=".*OrderHandler.*")`
- Who calls it: `trace_path(function_name="OrderHandler", direction="inbound")`
- Read source: `get_code_snippet(qualified_name="pkg/orders.OrderHandler")`
<!-- codebase-memory-mcp:end -->
    '''
    system_prompt = '''
# Project
    - root {os.getcwd()}
    '''
    """
    code_memory = MCPServerStdio(
        timeout=10,
        command="/Users/mlackman/.local/bin/codebase-memory-mcp",
        args=[],
        tool_prefix="",
        allow_sampling=False,
    )
    """
 
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
