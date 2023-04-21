# (C) Yoshi Sato <satyoshi.com>

import numpy as np
from typing import Tuple, List
import copy

# normal levels are 7x7 grids
# large levels are 14x11 grids

OPEN_DIVIDER_SALAD =   [[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]]
OPEN_DIVIDER_SALAD = np.transpose(OPEN_DIVIDER_SALAD)

OPEN_DIVIDER_SALAD_L =  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
OPEN_DIVIDER_SALAD_L = np.transpose(OPEN_DIVIDER_SALAD_L)

PARTIAL_DEVIDER_SALAD =[[1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1]]
PARTIAL_DEVIDER_SALAD = np.transpose(PARTIAL_DEVIDER_SALAD)

PARTIAL_DEVIDER_SALAD_L =  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
PARTIAL_DEVIDER_SALAD_L = np.transpose(PARTIAL_DEVIDER_SALAD_L)

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
    "cutboard0": (0, 1),
    "cutboard1": (0, 2),
    "plate0": (5, 6),
    "plate1": (6, 5),

    "counter0": (3, 1),
    "counter1": (3, 2),
    "counter2": (3, 3),
    "counter3": (3, 4),

    "star": (0, 3)
}
ITEM_LOCATIONS_L = {
    "tomato": (12, 0),
    "lettuce": (13, 1),
    "cutboard0": (0, 1),
    "cutboard1": (0, 2),
    "plate0": (12, 10),
    "plate1": (13, 9),

    "counter0": (6, 6),
    "counter1": (6, 7),
    "counter2": (6, 8),
    "counter3": (6, 9),

    "star": (0, 9)
}
MOVABLES = ["tomato", "lettuce", "plate0", "plate1"]


def identify_items_at(location: Tuple[int, int], item_locations: dict) -> List[str]:
    result = []
    for item, loc in item_locations.items():
        if (loc[0] == location[0]) and (loc[1] == location[1]):
            result.append(item)
    return result


def get_dst_tuple(item: str, level: list, item_locations: dict) -> Tuple[Tuple[int, int], list]:
    destination: Tuple[int, int] = item_locations[item]
    level: list = copy.deepcopy(level)
    level[destination[0]][destination[1]] = 0
    return destination, level
