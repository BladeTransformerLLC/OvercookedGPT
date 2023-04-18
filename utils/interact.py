from utils.core import *
import numpy as np
from typing import Tuple


def __extract_object_names(s: str) -> str:
    result = []
    if "Tomato" in s:
        result.append("tomato")
    if "Lettuce" in s:
        result.append("lettuce")
    if "Plate" in s:
        result.append("plate")
    return ", ".join(result)



def interact(agent, world) -> Tuple[str, Tuple[int, int]]:
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    action_str = None
    action_loc = None

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return action_str, action_loc

    action_loc = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    gs = world.get_gridsquare_at(action_loc)
    gs_list = world.get_gridsquare_list_at(action_loc)
    #print(f"gs_list = {gs_list}")

    # if floor in front --> move to that square
    if isinstance(gs, Floor): #and gs.holding is None:
        action_str = "moved to"
        agent.move_to(gs.location)

    # if holding something
    elif agent.holding is not None:
        # not None only when agent puts foods on cutboard or plate, or delivers

        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            obj = agent.holding
            #print(f"holding && delivering: obj.contents = {obj.contents}")

            if obj.is_deliverable():
                action_str = f"delivered {__extract_object_names(str(obj.contents))} at"
                gs.acquire(obj)
                agent.release()
                print('\nDelivered {}!'.format(obj.full_name))

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            #print(f"holding && occupied: obj.contents = {obj.contents}")

            if mergeable(agent.holding, obj):
                action_str = f"merged {__extract_object_names(str(obj.contents))} with"
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                # if playable version, merge onto counter first
                if world.arglist.gpt:
                    # --gpt
                    gs.acquire(agent.holding)
                    agent.release()


        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            #print(f"holding && not(occupied): obj.contents = {obj.contents}")

            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.gpt:
                # normally chop, but if in playable game mode then put down first
                obj.chop()
            else:
                # --gpt
                action_str = f"put {__extract_object_names(str(obj.contents))} onto"
                gs.acquire(obj) # obj is put onto gridsquare
                agent.release()
                assert world.get_object_at(gs.location, obj, find_held_objects =\
                    False).is_held == False, "Verifying put down works"

    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            #print(f"not(holding) && occupied: obj.contents = {obj.contents}")

            # if in playable game mode, then chop raw items on cutting board
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.gpt:
                # --gpt
                action_str = f"sliced {__extract_object_names(str(obj.contents))} on"
                obj.chop()
            else:
                action_str = f"picked up {__extract_object_names(str(obj.contents))}"
                gs.release()
                agent.acquire(obj)

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass

    return action_str, action_loc
