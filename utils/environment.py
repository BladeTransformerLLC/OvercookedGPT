# (C) Yoshi Sato <satyoshi.com>

import numpy as np
from typing import Tuple, List
import copy

# all levels are 7x7 grids
OPEN_DIVIDER_SALAD =   [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]]
OPEN_DIVIDER_SALAD = np.transpose(OPEN_DIVIDER_SALAD)

PARTIAL_DEVIDER_SALAD =[[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]]
PARTIAL_DEVIDER_SALAD = np.transpose(PARTIAL_DEVIDER_SALAD)

# TODO: for multi-agent sim
FULL_DIVIDER_SALAD =   [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]]
FULL_DIVIDER_SALAD = np.transpose(FULL_DIVIDER_SALAD)


# x, y
ITEM_LOCATIONS = {
    "tomato": (5, 0),
    "lettuce": (6, 1),
    #"cutboard0": (0, 1),
    #"cutboard1": (0, 2),
    "cutboard": (0, 2),
    #"plate0": (5, 6),
    #"plate1": (6, 5),
    "plate": (5, 6),
    "star": (0, 3)
}
MOVABLES = ["tomato", "lettuce", "plate"]


def identify_items_at(location: Tuple[int, int]) -> List[str]:
    global ITEM_LOCATIONS
    result = []
    for item, loc in ITEM_LOCATIONS.items():
        if (loc[0] == location[0]) and (loc[1] == location[1]):
            result.append(item)
    return result


def get_dst_tuple(item: str, level: list) -> Tuple[Tuple[int, int], list]:
    destination: Tuple[int, int] = ITEM_LOCATIONS[item]
    level: list = copy.deepcopy(level)
    level[destination[0]][destination[1]] = 0
    return destination, level
