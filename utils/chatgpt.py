# (C) Yoshi Sato <satyoshi.com>

import time
#import signal
from multiprocessing.connection import Listener
from pynput.keyboard import Key, Controller
import numpy as np
from typing import List, Tuple
from copy import deepcopy
import openai
import json
import re
import sys
import copy
import os

from utils.astar import *
from utils.environment import *
from utils.utils import *


__code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
def __extract_python_code(content: str) -> str:
    global __code_block_regex
    code_blocks: list = __code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n"
        for block in code_blocks:
            if block.startswith("python"):
                full_code += block[7:] + "\n"
            elif block.startswith(" python"):
                full_code += block[8:] + "\n"
            else:
                #pass
                full_code += block[0:] + "\n"
        print(colors.GREEN + "\n=========== execution =============")
        print(full_code)
        print("===================================" + colors.ENDC)
        return full_code
    else:
        return None


with open("utils/chatgpt/openai.json", "r") as f:
    g_openai_config: dict = json.load(f)


class ChatBot:
    def __init__(self, config: dict, arglist):
        if not("OPENAI_API_KEY" in os.environ):
            openai.api_key = config["access_token"]
        self.model: str = config["model"]
        self.messages: list = []

        instruction, example = None, None
        self.num_agents: int = arglist.num_agents
        if self.num_agents == 1:
            with open("utils/chatgpt/single_agent_instruction.txt", "r") as f:
                instruction = f.read()
            with open("utils/chatgpt/single_agent_example.txt", "r") as f:
                example = f.read()
        elif self.num_agents == 2:
            with open("utils/chatgpt/multi_agent_instruction.txt", "r") as f:
                instruction = f.read()
            with open("utils/chatgpt/multi_agent_example.txt", "r") as f:
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
        result: str = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self) -> str:
        try:
            completion = openai.ChatCompletion.create(model=self.model, messages=self.messages)
            #print(completion.usage) # number of tokens consumed
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            return colors.RED + f"ERROR: {e}" + colors.ENDC

g_chatbot = None
g_keyboard = Controller()

class GPTAgent:
    def __init__(self, id: int, arglist):
        assert 0 <= id <= 4
        self.id = id
        self.location = None
        self.on_hand = None
        self.level = None
        self.history = []
        self.prev_state = None
        if arglist.level == "open-divider_salad":
            self.level = OPEN_DIVIDER_SALAD
        elif arglist.level == "partial-divider_salad":
            self.level = PARTIAL_DEVIDER_SALAD
        elif arglist.level == "full-divider_salad":
            self.level = FULL_DIVIDER_SALAD
        else:
            assert False, f"unknown level: {arglist.level}"

    def set_state(self, location: Tuple[int, int], action_str: str, action_loc: Tuple[int, int]):
        """ set the latest game state
        Args:
            location (Tuple[int, int]): agent's current location
            action_str (str): action taken by the agent
            action_loc (Tuple[int, int]): location where the action was taken
        """
        self.location = location
        if action_str is None:
            return
        if self.prev_state is not None:
            # discard duplicate state
            if (self.prev_state[0] == location) and (self.prev_state[1] == action_str) and (self.prev_state[2] == action_loc):
                return
        description = action_str
        items: List[str] = identify_items_at(action_loc)
        if len(items) > 0:
            # remove duplicated items
            if ("sliced" in description) or ("picked" in description):
                if "tomato" in description:
                    items.remove("tomato")
                if "lettuce" in description:
                    items.remove("lettuce")
                if ("picked" in description) and (len(items) > 0):
                    description += " from"
            # change description for merged plate
            elif ("merged plate" in description) and (self.on_hand is not None):
                description = "put sliced " + ", ".join(self.on_hand) + " onto"
            description += ' ' + ", ".join(items)
            print(colors.GREEN + f"agent{self.id}.set_state(): " + description + colors.ENDC)
        self.history.append(description)
        if "picked" in description:
            # identify what item was picked up
            for item in ITEM_LOCATIONS.keys():
                if (item in description) and (item in MOVABLES):
                    if self.on_hand is None:
                        self.on_hand = [item]
                    else:
                        self.on_hand.append(item)
        elif ("put" in description) or ("merged" in description):
            if self.on_hand is not None:
                # update the location of the item
                for item in MOVABLES:
                    for obj in self.on_hand:
                        if item in obj:
                            ITEM_LOCATIONS[item] = action_loc
                self.on_hand = None
        if self.on_hand is not None:
            print(colors.YELLOW + f"agent{self.id}.on_hand = {self.on_hand}" + colors.ENDC)
        self.prev_state = (location, action_str, action_loc)

    def reset_state(self, reset_on_hand: bool=False):
        """ reset the game state of the agent
        Args:
            reset_on_hand (bool, optional): reset the on_hand variable. Defaults to False.
        """
        self.location = None
        self.action_str = None
        self.action_loc = None
        if reset_on_hand:
            self.on_hand = None

    def move_to(self, destination: Tuple[int, int]) -> bool:
        """ move to the specified destination
        Args:
            destination (Tuple[int, int]): 2D coordinate of the destination
        Returns:
            bool: True when the agent has reached the destination
        """
        if not isinstance(destination, tuple):
            print(colors.RED + f"ERROR: destination is not a tuple: {destination}" + colors.ENDC)
            return False
        if self.__has_reached(destination):
            print(colors.YELLOW + f"agent{self.id}.move_to(): reached destination" + colors.ENDC)
            return True
        dx = destination[0] - self.location[0]
        dy = destination[1] - self.location[1]
        print(colors.YELLOW + f"agent{self.id}.move_to(): source={self.location}, destination={destination}, (dx, dy) = ({dx}, {dy})" + colors.ENDC)
        global g_keyboard
        if dx < 0:
            g_keyboard.press(Key.left)
            g_keyboard.release(Key.left)
            return False
        elif dx > 0:
            g_keyboard.press(Key.right)
            g_keyboard.release(Key.right)
            return False
        if dy < 0:
            g_keyboard.press(Key.up)
            g_keyboard.release(Key.up)
            return False
        elif dy > 0:
            g_keyboard.press(Key.down)
            g_keyboard.release(Key.down)
            return False

    def fetch(self, item: str) -> bool:
        """ move to the item's location and pick it up
        Args:
            item (str): item to be picked up
        Returns:
            bool: success or failure
        """
        if self.on_hand is not None:
            for obj in self.on_hand:
                if item in obj:
                    return True  # item is already in hand
        for key in ITEM_LOCATIONS.keys():
            if item == key:
                destination, level = get_dst_tuple(item, self.level)
                path: List[Tuple[int, int]] = find_path(self.location, destination, level)
                print(colors.YELLOW + f"agent{self.id}.fetch(): path={path}" + colors.ENDC)
                self.move_to(path[1])
                break
        return False

    def put_onto(self, item) -> bool:
        """ place the object in hand onto the specified item
        Args:
            item (str or Tuple[int, int]): where to put the object
        Returns:
            bool: True if the task is closed
        """
        if self.on_hand is None:
            #print(colors.RED + f"GPTAgent.put_onto(): nothing in hand to put" + colors.ENDC)
            return True
        destination, level = None, None
        if isinstance(item, str):
            if not(item in ITEM_LOCATIONS.keys()):
                print(colors.RED + f"agent{self.id}.put_onto(): invalid item: {item}" + colors.ENDC)
                return True
            destination, level = get_dst_tuple(item, self.level)
        elif isinstance(item, tuple):
            pass #TODO: also accept 2D coordinate
        else:
            assert False, f"item must be str or Tuple[int, int]: {type(item)}"
        path: List[Tuple[int, int]] = find_path(self.location, destination, level)
        print(colors.YELLOW + f"agent{self.id}.put_onto(): path={path}" + colors.ENDC)
        self.move_to(path[1])
        return False

    def slice_on(self, item: str) -> bool:
        """ slice food at the specified item's location
        Args:
            item (str): the name of the item to chop on (must be a cutboard)
        Returns:
            bool: True if the task is closed
        """
        if not(item in ITEM_LOCATIONS.keys()):
            print(colors.RED + f"agent{self.id}.slice_on(): invalid item: {item}" + colors.ENDC)
            return True
        if not("cutboard" in item):
            print(colors.RED + f"agent{self.id}.slice_on(): cannot slice on {item}" + colors.ENDC)
            return True
        destination: Tuple[int, int] = ITEM_LOCATIONS[item]
        for description in self.history[::-1]:
            if ("put" in description) and (item in description):
                self.move_to(destination)
                break
            elif "sliced" in description:
                return True
        return False

    def deliver(self, dummy=None) -> bool:
        """ deliver the food to the goal destination (i.e., "star")
        Args:
            dummy (_type_, optional): ignored
        Returns:
            bool: True if the task is closed
        """
        destination = list(ITEM_LOCATIONS["star"])
        destination[0] += 1
        if self.move_to(tuple(destination)):
            # reached the destination
            global g_keyboard
            g_keyboard.press(Key.left)
            return True
        return False

    def __has_reached(self, destination) -> bool:
        return (self.location[0] == destination[0]) and (self.location[1] == destination[1])


def gpt_proc(arglist):
    listener = Listener(("localhost", 6000)) # family is deduced to be 'AF_INET'
    #print(colors.GREEN + f"GPT listener.address: {listener.address}" + colors.ENDC)
    connection = None
    #listener.close() #todo

    agent1 = GPTAgent(1, arglist)
    agent2 = GPTAgent(2, arglist)

    global g_chatbot, g_openai_config
    g_chatbot = ChatBot(g_openai_config, arglist)

    ##########################################################
    def __update_state() -> bool:
        """receive the latest state from the game
        Returns:
            bool: success or failure
        """
        nonlocal listener, connection, agent1, agent2
        if listener.last_accepted is None:
            try:
                connection = listener.accept() # blocking
                print(colors.GREEN + f"listener.last_accepted: {listener.last_accepted}" + colors.ENDC)
            except Exception as e:
                print(colors.RED + f"Exception in listener.accept(): {e}" + colors.ENDC)
                return False
        if listener.last_accepted is not None:
            try:
                if connection.poll(): # non-blocking, returns True if new data is found
                    msg = connection.recv()
                    while connection.poll(): # retrieve the last data, discard the previous ones
                        msg = connection.recv()

                    # state: [game frame, agent id, agent location, agent action string, agent action location]
                    print(f"received the current game state: {msg}")
                    agent_id: int = msg[1]
                    if agent_id == 1:
                        agent1.set_state(location=msg[2], action_str=msg[3], action_loc=msg[4])
                    elif agent_id == 2:
                        agent2.set_state(location=msg[2], action_str=msg[3], action_loc=msg[4])

                #else:
                #    print("WARNING: no new data from client")
            except Exception as e:
                    print(colors.RED + f"Exception in __update_state(): {e}" + colors.ENDC)
                    return False
            return True
        return False
    ##########################################################

    if arglist.debug:
        task_queue = []
        if arglist.num_agents == 1:
            task_queue.append((agent1.fetch, "tomato"))
            task_queue.append((agent1.put_onto, "cutboard0"))
            task_queue.append((agent1.slice_on, "cutboard0"))
            task_queue.append((agent1.fetch, "tomato"))
            task_queue.append((agent1.put_onto, "plate0"))
            task_queue.append((agent1.fetch, "lettuce"))
            task_queue.append((agent1.put_onto, "cutboard0"))
            task_queue.append((agent1.slice_on, "cutboard0"))
            task_queue.append((agent1.fetch, "lettuce"))
            task_queue.append((agent1.put_onto, "plate0"))
            task_queue.append((agent1.fetch, "lettuce"))
            task_queue.append((agent1.deliver, None))
        elif arglist.num_agents == 2:
            task_queue.append((agent2.fetch, "tomato"))
            task_queue.append((agent2.put_onto, "counter0"))
            task_queue.append((agent1.fetch, "tomato"))
            task_queue.append((agent1.put_onto, "cutboard0"))
            task_queue.append((agent1.slice_on, "cutboard0"))
            task_queue.append((agent2.fetch, "lettuce"))
            task_queue.append((agent2.put_onto, "counter0"))
            task_queue.append((agent1.fetch, "lettuce"))
            task_queue.append((agent1.put_onto, "cutboard1"))
            task_queue.append((agent1.slice_on, "cutboard1"))
            task_queue.append((agent2.fetch, "plate0"))
            task_queue.append((agent2.put_onto, "counter0"))
            task_queue.append((agent1.fetch, "tomato"))
            task_queue.append((agent1.put_onto, "counter0"))
            task_queue.append((agent1.fetch, "lettuce"))
            task_queue.append((agent1.put_onto, "counter0"))
            task_queue.append((agent1.fetch, "lettuce"))
            task_queue.append((agent1.deliver, None))
        else:
            assert False, f"arglist.num_agents must be 1 or 2: {arglist.num_agents}"
    else:
        time.sleep(2)
        sys.stdin = open(0)  # input() does not work with multiprocessing without this line
        question = input(colors.GREEN + "Enter a task: " + colors.ENDC)
        print(colors.YELLOW + "ChatGPT: Thinking...please wait..." + colors.ENDC)
        num_retries = 0
        max_retries = 5
        while num_retries < max_retries:
            response: str = g_chatbot(question)
            print("\n-------------------------- response --------------------------")
            print(colors.YELLOW + "ChatGPT: " + colors.ENDC + response)
            code: str = __extract_python_code(response)
            if code is None:
                print(colors.RED + "ERROR: no python code found in the response. Retrying..." + colors.ENDC)
                num_retries += 1
                question = "You must generate valid Python code. Please try again."
                continue
            else:
                if len(code) == 0:
                    print(colors.RED + "ERROR: python code is empty. Retrying..." + colors.ENDC)
                    num_retries += 1
                    question = "You must generate valid Python code. Please try again."
                    continue
                else:
                    print("\nPlease wait while I execute the above code...")
                    try:
                        # existing local vars must be given explicitly as a dict
                        ldict = {"agent1": agent1, "agent2": agent2}
                        exec(code, globals(), ldict)#locals())
                        task_queue = ldict["task_queue"]
                        print("Done executing code.")
                        break
                    except Exception as e:
                        print(colors.RED + "ERROR: could not execute the code: {}\nRetrying...".format(e) + colors.ENDC)
                        num_retries += 1
                        question = "While executing your code I've encountered the following error: {}\nPlease fix the error and show me valid code.".format(e)
                        continue
        print("Excecuting the task queue in the simulator...")
        time.sleep(2)

    i, j = -1, 0
    done = False
    while True:
        time.sleep(0.5)
        if done or arglist.manual:
            continue

        i += 1
        print("--------------------------------------------------------")
        print(f"i={i}")

        if not(done):
            f = task_queue[j][0]
            arg = task_queue[j][1]

            global g_keyboard
            if str(agent1) in str(f):
                print("agent1 is in the task")
                agent1.reset_state()
                g_keyboard.press('1')
                g_keyboard.release('1')
                while agent1.location is None:
                    if not(__update_state()):
                        pass#time.sleep(1)
            elif str(agent2) in str(f):
                print("agent2 is in the task")
                agent2.reset_state()
                g_keyboard.press('2')
                g_keyboard.release('2')
                #time.sleep(0.5)
                while agent2.location is None:
                    if not(__update_state()):
                        pass#time.sleep(1)

            if f(arg):
                print(colors.GREEN + f"task complete: {str(f)}({str(arg)})" + colors.ENDC)
                j += 1
                if j == len(task_queue):
                    print(colors.GREEN + f"ALL TASKS COMPLETE: score={i} (lower the better)" + colors.ENDC)
                    done = True





