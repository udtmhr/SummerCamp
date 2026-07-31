from __future__ import annotations

WORKER_ACTIONS = ("stay", "move", "build_city", "transfer")
CART_ACTIONS = ("stay", "move", "pillage", "transfer")
CITY_ACTIONS = ("no_action", "build_worker", "build_cart", "research")
DIRECTIONS = ("n", "e", "s", "w")
RESOURCES = ("wood", "coal", "uranium")
TARGET_NAMES = (
    "worker_type",
    "worker_move",
    "worker_transfer_dir",
    "worker_resource",
    "cart_type",
    "cart_move",
    "cart_transfer_dir",
    "cart_resource",
    "city",
)
ACTION_SCHEMA = {
    "worker_type": WORKER_ACTIONS,
    "worker_move": DIRECTIONS,
    "worker_transfer_dir": DIRECTIONS,
    "worker_resource": RESOURCES,
    "cart_type": CART_ACTIONS,
    "cart_move": DIRECTIONS,
    "cart_transfer_dir": DIRECTIONS,
    "cart_resource": RESOURCES,
    "city": CITY_ACTIONS,
}

# Exact action ordering used by Isaiah Pressman's first-place Lux AI 2021
# policy.  Keeping this separate from ACTION_SCHEMA preserves compatibility
# with the existing factorized behavior-cloning checkpoints.
FIRST_PLACE_DIRECTIONS = DIRECTIONS
FIRST_PLACE_WORKER_ACTIONS = (
    "no_action",
    *(f"move_{direction}" for direction in FIRST_PLACE_DIRECTIONS),
    *(f"transfer_{resource}_{direction}" for resource in RESOURCES for direction in FIRST_PLACE_DIRECTIONS),
    "pillage",
    "build_city",
)
FIRST_PLACE_CART_ACTIONS = FIRST_PLACE_WORKER_ACTIONS[:17]
FIRST_PLACE_CITY_ACTIONS = CITY_ACTIONS
FIRST_PLACE_ACTION_SCHEMA = {
    "worker": FIRST_PLACE_WORKER_ACTIONS,
    "cart": FIRST_PLACE_CART_ACTIONS,
    "city_tile": FIRST_PLACE_CITY_ACTIONS,
}


def first_place_action_remap(
    entity: str,
    rotations: int,
    *,
    horizontal_flip: bool = False,
) -> dict[int, int]:
    actions = FIRST_PLACE_ACTION_SCHEMA[entity]
    direction_to_delta = {"n": (-1, 0), "e": (0, 1), "s": (1, 0), "w": (0, -1)}
    delta_to_direction = {value: key for key, value in direction_to_delta.items()}
    direction_remap = {}
    for direction, delta in direction_to_delta.items():
        dy, dx = delta
        for _ in range(rotations % 4):
            dy, dx = -dx, dy
        if horizontal_flip:
            dx = -dx
        direction_remap[direction] = delta_to_direction[(dy, dx)]
    result = {}
    for index, action in enumerate(actions):
        transformed = action
        for direction, mapped in direction_remap.items():
            if action.endswith(f"_{direction}"):
                transformed = f"{action[:-1]}{mapped}"
                break
        result[index] = actions.index(transformed)
    return result
