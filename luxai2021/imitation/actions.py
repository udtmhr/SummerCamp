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
