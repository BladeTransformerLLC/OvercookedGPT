# (C) Yoshi Sato <satyoshi.com>

import openai
import json
import os

from utils.utils import *


with open("utils/chatgpt/openai.json", "r") as f:
    g_openai_config: dict = json.load(f)


class ChatGPT:
    def __init__(self, config: dict, arglist):
        self.api_key = None
        if "OPENAI_API_KEY" in os.environ:
            self.api_key: str = os.environ["OPENAI_API_KEY"]
        else:
            self.api_key: str = config["access_token"]
        openai.api_key = self.api_key
        self.model: str = config["model"]
        self.temperature = float(config["temperature"])
        self.top_p = float(config["top_p"])
        self.max_tokens = int(config["max_tokens"])

        self.messages: list = []

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

        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        self.messages.append({"role": "system", "content": "You are a Python programmer. Help me write code in Python."})
        self.messages.append({"role": "user", "content": instruction})

        # one-shot learning
        self.messages.append({
            "role": "system",
            "name": "example_user",
            "content": "Make a lettuce salad."
        })
        self.messages.append({"role": "system", "name": "example_assistant", "content": example})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result: str = self.__execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def __execute(self) -> str:
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                #top_p=self.top_p,
                max_tokens=self.max_tokens,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True  # use SSE (Server-Sent Events)
            )
        except Exception as e:
            print(e)
            assert False, colors.RED + f"ERROR: {e}" + colors.ENDC
        result = []
        for chunk in completion:
            chunk: dict = json.loads(str(chunk["choices"][0]["delta"]))
            if "content" in chunk.keys():
                chunk = str(chunk["content"])
                print(colors.YELLOW + chunk + colors.ENDC, end="", flush=True)
                result.append(chunk)
        return ''.join(result)
