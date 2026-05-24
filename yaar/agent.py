from typing import Callable

from pydantic_ai.mcp import MCPServerStdio 

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
