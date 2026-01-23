"""
CraftaxSymbolicEnvNoAutoReset variant that prepares all inputs needed for external flow logic.

This environment runs the base game logic and saves all intermediate values
(intrinsics_diff, inventory_diff, closest_blocks, etc.) in the info dict,
allowing flow reward/completion to be computed externally after the step.

This avoids JAX recompilation when the flow module changes.
"""
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import spaces
from typing import Tuple, Optional
import chex

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import BlockType, Action, OBS_DIM, ItemType
from craftax.craftax.game_logic import craftax_step as craftax_step_default, is_game_over
from craftax.craftax.game_logic_warp import craftax_step as craftax_step_warp
from craftax.craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world as generate_world_default
from craftax.craftax.world_gen.world_gen_warp import generate_world as generate_world_warp


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


class CraftaxSymbolicEnvNoAutoResetFlowReady(EnvironmentNoAutoReset):
    """
    Craftax environment that prepares all inputs needed for external flow logic.

    Unlike the flow environment, this does NOT compute flow rewards or task completion.
    Instead, it saves all intermediate values in the info dict so they can be used
    externally to compute flow logic without JAX recompilation on module changes.
    """

    def __init__(
        self,
        static_env_params: Optional[StaticEnvParams] = None,
        include_relative_positions: bool = True,
        use_warp: bool = False,
    ):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        # Set include_relative_positions on static_env_params for game_logic
        static_env_params = static_env_params.replace(include_relative_positions=include_relative_positions)
        self.static_env_params = static_env_params
        self.include_relative_positions = include_relative_positions
        self.use_warp = use_warp

        # Select step function based on warp flag
        if use_warp:
            self._craftax_step = craftax_step_warp
            self._generate_world = generate_world_warp
            print(f"[CraftaxEnvFlowReady] Using optimized JAX warp step function")
        else:
            self._craftax_step = craftax_step_default
            self._generate_world = generate_world_default
            print(f"[CraftaxEnvFlowReady] Using default step function")

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def step_env(
        self, rng: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """
        Step the environment and prepare all values needed for external flow logic.

        Returns:
            obs: Observation after step
            state: New environment state
            reward: Base environment reward (game score, NOT flow reward)
            done: Whether agent died (NOT flow done which depends on player_state)
            info: Dict containing all values needed for external flow computation:
                - player_intrinsics: [health, food, drink, energy] after step
                - intrinsics_diff: Change in intrinsics this step
                - inventory_diff: Change in inventory this step
                - closest_blocks: Closest blocks after step
                - closest_blocks_prev: Closest blocks before step
                - player_level_diff: Change in dungeon level this step
                - monsters_killed_diff: Change in monsters killed this step
                - in_lava_pen: Lava penalty (negative health if in lava)
                - achievements_diff: Change in achievements this step
                - killed: Whether agent died this step
                - env_reward: Base environment reward
                - inventory: Current inventory (for task completion checks)
                - player_level: Current dungeon level
                - monsters_killed: Current monsters killed count
                - achievements: Current achievements
        """
        # Capture state BEFORE step for computing diffs
        prev_monsters_killed = state.monsters_killed
        prev_player_level = state.player_level

        # Run game logic
        state, env_reward = self._craftax_step(rng, state, action, params, self.static_env_params)

        # Build player intrinsics array
        player_intrinsics = jnp.array([
            state.player_health,
            state.player_food,
            state.player_drink,
            state.player_energy,
        ])

        # Check if dead
        killed = is_game_over(state, params, self.static_env_params)

        # Compute lava penalty
        in_lava = (
            state.map[state.player_level, state.player_position[0], state.player_position[1]]
            == BlockType.LAVA.value
        )
        in_lava_pen = -state.player_health * in_lava

        # Compute diffs (these are also stored in state by craftax_step, but we compute explicitly for clarity)
        monsters_killed_diff = state.monsters_killed - prev_monsters_killed
        player_level_diff = state.player_level - prev_player_level

        # Build info dict with base achievements logging
        info = log_achievements_to_info(state, killed)
        info["discount"] = self.discount(state, params)

        # Only save values that are COMPUTED here and not available in state
        # Everything else (intrinsics_diff, inventory_diff, closest_blocks, etc.)
        # can be read directly from state in the training loop
        info["player_level_diff"] = player_level_diff
        info["monsters_killed_diff"] = monsters_killed_diff
        info["in_lava_pen"] = in_lava_pen
        info["killed"] = killed
        info["env_reward"] = env_reward

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            env_reward,  # Return base env reward, NOT flow reward
            killed,  # Return killed signal, NOT flow done (which depends on external player_state)
            info,
        )

    def reset_env(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        rng, _rng = jax.random.split(rng)
        state = self._generate_world(_rng, params, self.static_env_params)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        pixels = render_craftax_symbolic(
            state, include_relative_positions=self.include_relative_positions
        )
        return pixels

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-Symbolic-NoAutoReset-FlowReady-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()
        # closest_blocks includes 3 extra ladder channels (down, up, down_blocked)
        relative_positions_shape = (
            2 * (len(BlockType) + 3) if self.include_relative_positions else 0
        )

        obs_shape = flat_map_obs_shape + inventory_obs_shape + relative_positions_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )


def compute_flow_reward_and_completion(
    player_state,
    inner_state,
    info,
    check_task_reward,
    check_task_completion,
    num_tasks,
):
    """
    Compute flow reward and task completion from state and info.

    Most values come from inner_state (the env state after step).
    Only computed values (diffs, in_lava_pen, killed, env_reward) come from info.

    Args:
        player_state: Current flow state (which skill the agent is on)
        inner_state: The inner env state after step (state_st.env_state)
        info: Info dict with computed values (player_level_diff, monsters_killed_diff, etc.)
        check_task_reward: Flow reward function from code_parser
        check_task_completion: Flow completion function from code_parser
        num_tasks: Total number of tasks in the flow

    Returns:
        reward: Flow-based reward
        new_player_state: Updated player state after task completion check
        task_done: Whether current task was completed (or agent died)
        flow_done: Whether flow is complete (agent died or reached final task)
    """
    # Build player intrinsics from state
    player_intrinsics = jnp.array([
        inner_state.player_health,
        inner_state.player_food,
        inner_state.player_drink,
        inner_state.player_energy,
    ])

    # Compute flow reward
    # Values from state: intrinsics_diff, inventory_diff, closest_blocks, closest_blocks_prev, achievements_diff
    # Values from info: player_level_diff, monsters_killed_diff, env_reward, in_lava_pen
    reward = check_task_reward(
        player_state,
        inner_state.intrinsics_diff,
        inner_state.inventory_diff,
        inner_state.closest_blocks,
        inner_state.closest_blocks_prev,
        info["player_level_diff"],
        info["monsters_killed_diff"],
        info["env_reward"],
        info["in_lava_pen"],
        inner_state.achievements_diff,
    )

    # Check task completion
    # Values from state: closest_blocks, closest_blocks_prev, player_level, monsters_killed,
    #                    inventory, inventory_diff, intrinsics_diff, achievements
    # Values computed here: player_intrinsics
    new_player_state, task_done = check_task_completion(
        player_state,
        inner_state.closest_blocks,
        inner_state.closest_blocks_prev,
        inner_state.player_level,
        inner_state.monsters_killed,
        inner_state.inventory,
        inner_state.inventory_diff,
        player_intrinsics,
        inner_state.intrinsics_diff,
        inner_state.achievements,
    )

    # Handle terminal states
    killed = info["killed"]
    task_done = jnp.logical_or(task_done, killed)
    flow_done = jnp.logical_or(killed, player_state == num_tasks)

    return reward, new_player_state, task_done, flow_done
