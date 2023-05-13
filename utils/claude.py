# (C) Yoshi Sato <satyoshi.com>

import anthropic
import os
import json

from utils.utils import *


with open("utils/claude/anthropic.json", "r") as f:
    g_anthropic_config: dict = json.load(f)


class Claude:
    def __init__(self, config: dict, arglist):
        # https://console.anthropic.com/docs/api/reference
        self.api_key = None
        if "ANTHROPIC_API_KEY" in os.environ:
            self.api_key: str = os.environ["ANTHROPIC_API_KEY"]
        else:
            self.api_key: str = config["access_token"]
        self.client = anthropic.Client(self.api_key)
        self.model: str = config["model"]
        self.temperature = float(config["temperature"])
        self.top_k = int(config["top_k"])
        self.top_p = float(config["top_p"])
        self.max_tokens = int(config["max_tokens_to_sample"])

        self.messages: str = ''

        instruction, example = None, None
        self.num_agents: int = arglist.num_agents
        if self.num_agents == 1:
            with open("utils/prompts/single_agent_instruction.txt", "r") as f:
                instruction = f.read()
            with open("utils/prompts/single_agent_example.txt", "r") as f:
                example = f.read()
        elif self.num_agents == 2:
            with open("utils/prompts/multi_agent_instruction.txt", "r") as f:
                instruction = f.read()
            with open("utils/prompts/multi_agent_example.txt", "r") as f:
                example = f.read()
        else:
            assert False, f"num_agents must be 1 or 2: {self.num_agents}"

        self.messages += f"{anthropic.HUMAN_PROMPT}You are a Python programmer. Help me write code in Python."
        # one-shot learning
        self.messages += f"{anthropic.HUMAN_PROMPT}{instruction}"
        self.messages += f"{anthropic.AI_PROMPT}{example}"

    def __call__(self, message):
        self.messages += f"{anthropic.HUMAN_PROMPT}{message}{anthropic.AI_PROMPT}"
        result: str = self.__execute()
        self.messages += result
        return result

    def __execute(self) -> str:
        try:
            completion = self.client.completion_stream(
                prompt=self.messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model=self.model,
                temperature=self.temperature,
                max_tokens_to_sample=self.max_tokens,
                stream=True,
            )
        except Exception as e:
            print(e)
            assert False, colors.RED + f"ERROR: {e}" + colors.ENDC
        result = ''
        for c in completion:
            if "completion" in c.keys():
                chunk: str = c["completion"]
                if len(result) > 0:
                    chunk = chunk[chunk.find(result)+len(result):]
                print(colors.YELLOW + chunk + colors.ENDC, end="", flush=True)
                result += chunk
            if "stop" in c.keys():
                if c["stop"] == anthropic.HUMAN_PROMPT:
                    break
        print(colors.YELLOW + result + colors.ENDC)
        return result
