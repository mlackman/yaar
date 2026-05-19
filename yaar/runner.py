from typing import Callable, Protocol, Sequence

from pydantic_ai import Agent, BaseToolCallPart, PartEndEvent, RunUsage, ThinkingPart, UsageLimits
from pydantic_ai import (
    FunctionToolCallEvent,
    FinalResultEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    ModelMessage
)

from .models import Deps, Response 

type Log = Callable[[str], None]
type ToolUsageLog = Callable[[BaseToolCallPart], None] 
type UsageLog = Callable[[RunUsage], None]


class AgentLog(Protocol):
    debug: Log
    output: Log
    tool_usage: ToolUsageLog 
    usage: UsageLog


async def run_agent(
    agent: Agent,
    prompt: str,
    log: AgentLog,
    deps: Deps,
    message_history: Sequence[ModelMessage] | None = None
) -> Response:
    final_response: Response | None = None

    async with agent.iter(prompt, usage_limits=UsageLimits(request_limit=None), message_history=message_history, deps=deps) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # A user prompt node => The user has provided input
                log.debug('=== UserPromptNode ===')
            elif Agent.is_model_request_node(node):
                # A model request node => We can stream tokens from the model's request
                log.debug('=== ModelRequestNode: streaming partial request tokens ===')
                log.debug(f'=== Requests: {run.usage().requests} ===')

                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            log.debug(f'[Request] Starting part {event.index}: {event.part!r}')
                        elif isinstance(event, PartDeltaEvent):
                            pass
                            """
                            if isinstance(event.delta, TextPartDelta):
                                log(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
                            elif isinstance(event.delta, ThinkingPartDelta):
                                log(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
                            elif isinstance(event.delta, ToolCallPartDelta):
                                log(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
                            """
                        elif isinstance(event, PartEndEvent):
                            if (isinstance(event.part, ThinkingPart)):
                                log.output(event.part.content)
                        elif isinstance(event, FinalResultEvent):
                            log.debug(f'[Result] The model started producing a final result (tool_name={event.tool_name})')
                            final_result_found = True
                            break

                    if final_result_found:
                        # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                        # A similar `AgentStream.stream_output()` method is available to stream structured output.
                        async for output in request_stream.stream_text(delta=True):
                            log.output(output)

            elif Agent.is_call_tools_node(node):
                # A handle-response node => The model returned some data, potentially calls a tool
                log.debug('=== CallToolsNode: streaming partial response & tool usage ===')
                log.debug(f'***** USAGE: {node.model_response.usage} *****')
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            log.tool_usage(event.part) 
                            log.debug(f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})')
                        elif isinstance(event, FunctionToolResultEvent):
                            log.debug(f'[Tools] Tool call {event.tool_call_id!r} returned')
            elif Agent.is_end_node(node):
                # Once an End node is reached, the agent run is complete
                assert run.result is not None
                assert run.result.output == node.data.output
                log.usage(run.usage())
                final_response = Response(run.result.output, run.result)

    assert final_response is not None, 'Final result was not collected'
    return final_response
