from dataclasses import dataclass
from typing import Tuple, Any

import jax.random
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0

    def __str__(self):
        return (
            f"Inventory(wood={self.wood}, stone={self.stone}, coal={self.coal}, "
            f"iron={self.iron}, diamond={self.diamond}, sapling={self.sapling}, "
            f"wood_pickaxe={self.wood_pickaxe}, stone_pickaxe={self.stone_pickaxe}, "
            f"iron_pickaxe={self.iron_pickaxe}, wood_sword={self.wood_sword}, "
            f"stone_sword={self.stone_sword}, iron_sword={self.iron_sword})"
        )


@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: int
    mask: bool
    attack_cooldown: int


@struct.dataclass
class EnvState:
    map: jnp.ndarray
    mob_map: jnp.ndarray

    player_position: jnp.ndarray
    player_direction: int

    # Intrinsics
    player_health: int
    player_food: int
    player_drink: int
    player_energy: int
    is_sleeping: bool

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float

    inventory: Inventory

    zombies: Mobs
    cows: Mobs
    skeletons: Mobs
    arrows: Mobs
    arrow_directions: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    light_level: float

    achievements: jnp.ndarray

    state_rng: Any

    timestep: int

    closest_blocks: jnp.ndarray
    player_state: jnp.ndarray
    inventory_diff: Inventory
    intrinsics_diff: jnp.ndarray
    achievements_diff: jnp.ndarray
    closest_blocks_prev: jnp.ndarray
    seen_blocks: jnp.ndarray
    task_done: bool

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 1500
    day_length: int = 300

    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3

    mob_despawn_distance: int = 14

    spawn_cow_chance: float = 0.1
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (64, 64)

    # Mobs
    max_zombies: int = 3
    max_cows: int = 3
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3
