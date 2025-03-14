from jax import lax
from gymnax.environments import spaces, environment
from typing import Tuple, Optional
import chex

from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax_classic.envs.common import compute_score
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.game_logic import craftax_step, is_game_over
from craftax.craftax_classic.envs.craftax_state import (
    EnvState,
    EnvParams,
    StaticEnvParams,
)
from craftax.craftax_classic.renderer import render_craftax_symbolic
from craftax.craftax_classic.world_gen import generate_world

from Craftax.craftax.craftax_classic.util.code_parser import task_and_reward_funcs


def get_map_obs_shape():
    num_mobs = 4
    num_blocks = len(BlockType)

    return OBS_DIM[0], OBS_DIM[1], num_blocks + num_mobs


def get_flat_map_obs_shape():
    map_obs_shape = get_map_obs_shape()
    return map_obs_shape[0] * map_obs_shape[1] * map_obs_shape[2]


def get_inventory_obs_shape():
    inv_size = 12
    num_intrinsics = 4
    light_level = 1
    is_sleeping = 1
    direction = 4

    return inv_size + num_intrinsics + light_level + is_sleeping + direction


class CraftaxClassicSymbolicEnvNoAutoReset(EnvironmentNoAutoReset):
    def __init__(self, static_env_params: StaticEnvParams = None, module_dict=None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

        if module_dict is not None:

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
        else:
            assert 0, "Module dict is None"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
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
            state.map[state.player_position[0], state.player_position[1]]
            == BlockType.LAVA.value
        )
        in_lava_pen = -state.player_health * in_lava
        #done = self.is_terminal(state, params)
        reward = self.check_task_reward(
            state.player_state,
            state.intrinsics_diff,
            state.inventory_diff,
            state.closest_blocks,
            state.closest_blocks_prev,
            env_reward,
            in_lava_pen,
            state.achievements_diff,
        )

        new_player_state, task_done = self.check_task_completion(
            state.player_state,
            state.closest_blocks,
            state.closest_blocks_prev,
            state.inventory,
            state.inventory_diff,
            player_intrinsics,
            state.intrinsics_diff,
            state.achievements,
        )
        player_state_diff = new_player_state - state.player_state
        task_done = jnp.logical_or(task_done, killed)
        done = jnp.logical_or(killed, state.player_state == self.num_tasks)


        #done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

        state = state.replace(player_state=new_player_state)
        state = state.replace(task_done=task_done)
        state = state.replace(player_state_diff=player_state_diff)
        info["task_done"] = task_done
        info["closest_blocks"] = state.closest_blocks
        info["reached_state"] = jnp.arange(self.num_tasks + 1) <= state.player_state

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
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return 17

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(17)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )


class CraftaxClassicSymbolicEnv(environment.Environment):
    def __init__(self, static_env_params: StaticEnvParams = None):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)

        done = self.is_terminal(state, params)
        info = compute_score(state, done)
        info["discount"] = self.discount(state, params)

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
        state = generate_world(rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_symbolic(state)
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params)

    @property
    def name(self) -> str:
        return "Craftax-Classic-Symbolic-v1"

    @property
    def num_actions(self) -> int:
        return 17

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(17)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()

        obs_shape = flat_map_obs_shape + inventory_obs_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )
