from craftax.craftax.util.code_parser import task_and_reward_funcs
import jax
from jax import lax
from gymnax.environments import spaces
from typing import Tuple, Optional
import chex

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import *
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.craftax.renderer_fast import render_craftax_pixels_fast
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world


def get_map_obs_shape():
    num_mob_classes = 5
    num_mob_types = 8
    num_blocks = len(BlockType)
    num_items = len(ItemType)

    return (
        OBS_DIM[0],
        OBS_DIM[1],
        num_blocks + num_items + num_mob_classes * num_mob_types + 1,
    )


def get_flat_map_obs_shape():
    map_obs_shape = get_map_obs_shape()
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    return 51


def get_symbolic_obs_shape():
    """Compute the symbolic observation dimension."""
    num_blocks = len(BlockType)
    num_items = len(ItemType)
    mob_types_per_class = 8
    num_mob_classes = 5

    # Map view one-hot + items one-hot + mobs + light map
    map_features = num_blocks + num_items + (num_mob_classes * mob_types_per_class) + 1
    map_flat = OBS_DIM[0] * OBS_DIM[1] * map_features

    # Inventory (16) + potions (6) + intrinsics (9) + direction (4) + armour (4) + armour_enchantments (4) + special (8)
    flat_features = 16 + 6 + 9 + 4 + 4 + 4 + 8

    # Closest blocks: (num_blocks + 3 ladders) * 2 directions
    num_closest = num_blocks + 3  # blocks + ladder_down, ladder_up, ladder_down_blocked
    closest_flat = num_closest * 2

    return map_flat + flat_features + closest_flat


class CraftaxPixelsEnvNoAutoReset(EnvironmentNoAutoReset):
    """Flow-RL Craftax env with pixel observations and symbolic task logic."""

    def __init__(
        self,
        static_env_params: Optional[StaticEnvParams] = None,
        module_dict=None,
    ):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

        if module_dict is None:
            raise ValueError("module_dict must not be None for Flow-RL env")

        (
            self.check_task_completion,
            self.check_task_reward,
            self.num_tasks,
            common_heads,
            num_common_heads,
            self.task_to_skill_index,
        ) = task_and_reward_funcs(module_dict)
        self.heads_info = [
            common_heads,
            num_common_heads,
        ]

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        prev_monsters_killed = state.monsters_killed
        prev_player_level = state.player_level
        state, env_reward = craftax_step(rng, state, action, params, self.static_env_params)
        player_intrinsics = jnp.array(
            [
                state.player_health,
                state.player_food,
                state.player_drink,
                state.player_energy,
            ]
        )
        killed = self.is_terminal(state, params)
        in_lava = (
            state.map[state.player_level, state.player_position[0], state.player_position[1]]
            == BlockType.LAVA.value
        )
        in_lava_pen = -state.player_health * in_lava
        monsters_killed_diff = state.monsters_killed - prev_monsters_killed
        player_level_diff = state.player_level - prev_player_level

        reward = self.check_task_reward(
            state.player_state,
            state.intrinsics_diff,
            state.inventory_diff,
            state.closest_blocks,
            state.closest_blocks_prev,
            player_level_diff,
            monsters_killed_diff,
            env_reward,
            in_lava_pen,
            state.achievements_diff,
        )

        new_player_state, task_done = self.check_task_completion(
            state.player_state,
            state.closest_blocks,
            state.closest_blocks_prev,
            state.player_level,
            state.monsters_killed,
            state.inventory,
            state.inventory_diff,
            player_intrinsics,
            state.intrinsics_diff,
            state.achievements,
        )
        player_state_diff = new_player_state - state.player_state
        task_done = jnp.logical_or(task_done, killed)
        done = jnp.logical_or(killed, state.player_state == self.num_tasks)

        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        state = state.replace(player_state=new_player_state)
        state = state.replace(task_done=task_done)
        state = state.replace(player_state_diff=player_state_diff)
        info["task_done"] = task_done
        info["closest_blocks"] = state.closest_blocks_prev
        info["reached_state"] = jnp.arange(self.num_tasks + 1) <= state.player_state
        info["env_reward"] = env_reward

        # Update symbolic obs in state for DAgger teacher queries
        state = state.replace(symbolic_obs=render_craftax_symbolic(state))

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        rng, _rng = jax.random.split(rng)
        state = generate_world(_rng, params, self.static_env_params)
        state = state.replace(symbolic_obs=render_craftax_symbolic(state))
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_pixels_fast(state, BLOCK_PIXEL_SIZE_AGENT) / 255.0
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-Pixels-NoAutoReset-Flow-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        height = (OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * BLOCK_PIXEL_SIZE_AGENT
        width = OBS_DIM[1] * BLOCK_PIXEL_SIZE_AGENT
        return spaces.Box(
            0.0,
            1.0,
            (height, width, 3),
            dtype=jnp.float32,
        )
