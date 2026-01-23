from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from craftax.craftax.constants import BlockType

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.craftax.renderer import render_craftax_symbolic
from craftax.craftax.world_gen.world_gen import generate_world
from craftax.craftax.craftax_state import EnvParams, EnvState, StaticEnvParams
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax.envs.craftax_flow_symbolic_env import (
    get_flat_map_obs_shape,
    get_inventory_obs_shape,
)


def _scan_options_from_module(module_dict: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Lightweight option function scanner.

    Expects functions with names: skill_{i}_name/initiation/termination/reward/network_number
    Returns lists aligned by increasing i for indices where all are present.
    """
    if module_dict is None:
        raise ValueError("module_dict must not be None for options env")

    def collect(pattern: str) -> Dict[int, Any]:
        out: Dict[int, Any] = {}
        rx = re.compile(pattern)
        for name, obj in module_dict.items():
            m = rx.match(name)
            if m is not None:
                idx = int(m.group(1))
                out[idx] = obj
        return out

    names = collect(r"^skill_(\d+)_name$")
    inits = collect(r"^skill_(\d+)_initiation$")
    terms = collect(r"^skill_(\d+)_termination$")
    rewards = collect(r"^skill_(\d+)_reward$")
    nets = collect(r"^skill_(\d+)_network_number$")

    indices = sorted(set(names) & set(inits) & set(terms) & set(rewards) & set(nets))
    if not indices:
        raise ValueError("No complete option definitions found in module_dict")

    ordered = {
        "names": [names[i] for i in indices],
        "inits": [inits[i] for i in indices],
        "terms": [terms[i] for i in indices],
        "rewards": [rewards[i] for i in indices],
        "nets": [nets[i] for i in indices],
    }
    return ordered


@chex.dataclass
class OptionContext:
    option_id: int
    start_inventory: Any
    start_level: int
    start_monsters_killed: Any
    target_diff: int = 1


class CraftaxOptionsSymbolicEnv(EnvironmentNoAutoReset):
    """Environment supporting hierarchical control with options.

    This env does not enforce option transitions by itself; instead, callers can
    supply an OptionContext (start snapshots) to step_env_with_option.
    """

    def __init__(self, static_env_params: Optional[StaticEnvParams] = None, module_dict=None):
        super().__init__()
        self.static_env_params = static_env_params or StaticEnvParams()

        # Parse option functions from the provided module.
        opts = _scan_options_from_module(module_dict)
        self._opt_names = opts["names"]
        self._opt_init = opts["inits"]
        self._opt_term = opts["terms"]
        self._opt_rew = opts["rewards"]
        self._opt_nets = opts["nets"]
        self.num_options = len(self._opt_names)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        # Same primitive action space as Craftax
        from craftax.craftax.constants import Action

        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        flat_map_obs_shape = get_flat_map_obs_shape()
        inventory_obs_shape = get_inventory_obs_shape()
        # closest_blocks includes 3 extra ladder channels (down, up, down_blocked)
        relative_positions_shape = 2 * (len(BlockType) + 3)

        obs_shape = flat_map_obs_shape + inventory_obs_shape + relative_positions_shape

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.float32,
        )

    def reset_env(self, rng: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        rng, _rng = jax.random.split(rng)
        state = generate_world(_rng, params, self.static_env_params)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        return render_craftax_symbolic(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # Standard Craftax step and info; no options logic unless using step_env_with_option
        prev_state = state
        state, env_reward = craftax_step(rng, state, action, params, self.static_env_params)
        done = self.is_terminal(state, params)
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        obs = self.get_obs(state)
        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state), env_reward, done, info

    def step_env_with_option(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
        opt: OptionContext,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # Step underlying environment
        prev_state = state
        state, env_reward = craftax_step(rng, state, action, params, self.static_env_params)
        done = self.is_terminal(state, params)

        # Build standard info
        info = log_achievements_to_info(state, done)
        info["discount"] = self.discount(state, params)

        # Compute option intrinsic reward and termination using start snapshots
        oid = int(opt.option_id)
        oid = max(0, min(oid, self.num_options - 1))

        # Initiation mask (optional): allow external to check before starting
        # Termination condition per our option definitions
        term_fn = self._opt_term[oid]
        rew_fn = self._opt_rew[oid]

        # Intrinsic reward uses per-step diffs; definitions accept health_penalty as last arg
        # Use 0.0 for simplicity; callers may incorporate penalties separately
        intrinsic_reward = rew_fn(
            state.inventory_diff,
            state.closest_blocks,
            prev_state.closest_blocks,
            state.intrinsics_diff,
            state.achievements_diff,
            0.0,
        )

        # Termination uses start snapshots
        option_done = term_fn(
            state.inventory,
            state.inventory_diff,
            state.closest_blocks,
            prev_state.closest_blocks,
            state.intrinsics,
            state.intrinsics_diff,
            state.achievements,
            state.player_state,  # pass current level as player_level
            opt.target_diff,
            opt.start_inventory,
            opt.start_level,
            opt.start_monsters_killed,
        )

        info["option_id"] = oid
        info["option_done"] = option_done
        info["intrinsic_reward"] = intrinsic_reward

        obs = self.get_obs(state)
        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(state), env_reward, done, info
