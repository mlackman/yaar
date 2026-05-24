from typing import Sequence
import dataclasses
import enum

import pydantic_ai
from pydantic_ai import AbstractToolset
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings 
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.test import TestModel

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
