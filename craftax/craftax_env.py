def make_craftax_env_from_name(
    name: str, auto_reset: bool, include_relative_positions: bool = True,
    use_floor_aware_closest_blocks: bool = False,
):
    if auto_reset:
        if name == "Craftax-Symbolic-v1" or name == "Craftax-Symbolic-AutoReset-v1":
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
            from craftax.craftax.craftax_state import StaticEnvParams
            static_env_params = StaticEnvParams(
                use_floor_aware_closest_blocks=use_floor_aware_closest_blocks
            ) if use_floor_aware_closest_blocks else None
            return CraftaxSymbolicEnv(
                static_env_params=static_env_params,
                include_relative_positions=include_relative_positions
            )
        elif name == "Craftax-Pixels-v1" or name == "Craftax-Pixels-AutoReset-v1":
            from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

            return CraftaxPixelsEnv()
        if (
            name == "Craftax-Classic-Symbolic-v1"
            or name == "Craftax-Classic-Symbolic-AutoReset-v1"
        ):
            from craftax.craftax_classic.envs.craftax_symbolic_env import (
                CraftaxClassicSymbolicEnv,
            )

            return CraftaxClassicSymbolicEnv()
        elif (
            name == "Craftax-Classic-Pixels-v1"
            or name == "Craftax-Classic-Pixels-AutoReset-v1"
        ):
            from craftax.craftax_classic.envs.craftax_pixels_env import (
                CraftaxClassicPixelsEnv,
            )

            return CraftaxClassicPixelsEnv()
        # NEW FABRAX ENVIRONMENTS
        elif (
            name == "Fabrax-Symbolic-v1"
            or name == "Fabrax-Symbolic-AutoReset-v1"
        ):
            from craftax.fabrax.envs.craftax_symbolic_env import (
                FabraxSymbolicEnv,
            )

            return FabraxSymbolicEnv()
        elif (
            name == "Fabrax-Pixels-v1"
            or name == "Fabrax-Pixels-AutoReset-v1"
        ):
            from craftax.fabrax.envs.craftax_pixels_env import (
                FabraxPixelsEnv,
            )

            return FabraxPixelsEnv()
    else:
        if name == "Craftax-Symbolic-v1":
            from craftax.craftax.envs.craftax_symbolic_env import (
                CraftaxSymbolicEnvNoAutoReset,
            )
            from craftax.craftax.craftax_state import StaticEnvParams
            static_env_params = StaticEnvParams(
                use_floor_aware_closest_blocks=use_floor_aware_closest_blocks
            ) if use_floor_aware_closest_blocks else None
            return CraftaxSymbolicEnvNoAutoReset(
                static_env_params=static_env_params,
                include_relative_positions=include_relative_positions
            )
        elif name == "Craftax-Pixels-v1":
            from craftax.craftax.envs.craftax_pixels_env import (
                CraftaxPixelsEnvNoAutoReset,
            )

            return CraftaxPixelsEnvNoAutoReset()
        elif name == "Craftax-Classic-Symbolic-v1":
            from craftax.craftax_classic.envs.craftax_symbolic_env import (
                CraftaxClassicSymbolicEnvNoAutoReset,
            )

            return CraftaxClassicSymbolicEnvNoAutoReset()
        elif name == "Craftax-Classic-Pixels-v1":
            from craftax.craftax_classic.envs.craftax_pixels_env import (
                CraftaxClassicPixelsEnvNoAutoReset,
            )

            return CraftaxClassicPixelsEnvNoAutoReset()
        # NEW FABRAX ENVIRONMENTS (NO AUTO RESET)
        elif name == "Fabrax-Symbolic-v1":
            from craftax.fabrax.envs.craftax_symbolic_env import (
                FabraxSymbolicEnvNoAutoReset,
            )

            return FabraxSymbolicEnvNoAutoReset()
        elif name == "Fabrax-Pixels-v1":
            from craftax.fabrax.envs.craftax_pixels_env import (
                FabraxPixelsEnvNoAutoReset,
            )

            return FabraxPixelsEnvNoAutoReset()

    raise ValueError(f"Unknown craftax environment: {name}")



def make_craftax_flow_env_from_name(
    name: str,
    auto_reset: bool,
    module_dict,
    include_relative_positions: bool = True,
    use_floor_aware_closest_blocks: bool = False,
    use_warp: bool = False,
    use_true_warp: bool = False,
):
    if auto_reset:
        assert 0, "Not implemented"
        # if name == "Craftax-Symbolic-v1" or name == "Craftax-Symbolic-AutoReset-v1":
        #     from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        #     return CraftaxSymbolicEnv()
        # elif name == "Craftax-Pixels-v1" or name == "Craftax-Pixels-AutoReset-v1":
        #     from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

        #     return CraftaxPixelsEnv()
        # if (
        #     name == "Craftax-Classic-Symbolic-v1"
        #     or name == "Craftax-Classic-Symbolic-AutoReset-v1"
        # ):
        #     from craftax.craftax_classic.envs.craftax_symbolic_env import (
        #         CraftaxClassicSymbolicEnv,
        #     )

        #     return CraftaxClassicSymbolicEnv()
        # elif (
        #     name == "Craftax-Classic-Pixels-v1"
        #     or name == "Craftax-Classic-Pixels-AutoReset-v1"
        # ):
        #     from craftax.craftax_classic.envs.craftax_pixels_env import (
        #         CraftaxClassicPixelsEnv,
        #     )

        #     return CraftaxClassicPixelsEnv()
    else:
        if name == "Craftax-Symbolic-v1":
            from craftax.craftax.envs.craftax_flow_symbolic_env import (
                CraftaxSymbolicEnvNoAutoReset,
            )
            from craftax.craftax.craftax_state import StaticEnvParams
            static_env_params = StaticEnvParams(
                use_floor_aware_closest_blocks=use_floor_aware_closest_blocks
            ) if use_floor_aware_closest_blocks else None
            return CraftaxSymbolicEnvNoAutoReset(
                static_env_params=static_env_params,
                module_dict=module_dict,
                include_relative_positions=include_relative_positions,
                use_warp=use_warp,
                use_true_warp=use_true_warp,
            )
            # from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset
            # return CraftaxSymbolicEnvNoAutoReset()
        elif name == "Craftax-Pixels-v1":
            from craftax.craftax.envs.craftax_flow_pixels_env import (
                CraftaxPixelsEnvNoAutoReset,
            )

            return CraftaxPixelsEnvNoAutoReset(module_dict=module_dict)
        elif name == "Craftax-Classic-Symbolic-v1":
            # from craftax.craftax_classic.envs.craftax_symbolic_env import (
            #     CraftaxClassicSymbolicEnvNoAutoReset,
            # )

            # return CraftaxClassicSymbolicEnvNoAutoReset()
            from craftax.craftax_classic.envs.craftax_flow_symbolic_env import (
                CraftaxClassicSymbolicEnvNoAutoReset,
            )
            return CraftaxClassicSymbolicEnvNoAutoReset(module_dict=module_dict)
        elif name == "Fabrax-Symbolic-v1":
            from craftax.fabrax.envs.craftax_flow_symbolic_env import (
                FabraxFlowSymbolicEnvNoAutoReset,
            )
            return FabraxFlowSymbolicEnvNoAutoReset(module_dict=module_dict)
        elif name == "Craftax-Classic-Pixels-v1":
            from craftax.craftax_classic.envs.craftax_flow_pixels_env import (
                CraftaxClassicPixelsEnvNoAutoResetFlow,
            )

            return CraftaxClassicPixelsEnvNoAutoResetFlow(module_dict=module_dict)

    raise ValueError(f"Unknown craftax environment: {name}")


def make_craftax_env_from_params(classic: bool, symbolic: bool, auto_reset: bool):
    if classic:
        if symbolic:
            if auto_reset:
                from craftax.craftax_classic.envs.craftax_symbolic_env import (
                    CraftaxClassicSymbolicEnv,
                )

                return CraftaxClassicSymbolicEnv()
            else:
                from craftax.craftax_classic.envs.craftax_symbolic_env import (
                    CraftaxClassicSymbolicEnvNoAutoReset,
                )

                return CraftaxClassicSymbolicEnvNoAutoReset()
        else:
            if auto_reset:
                from craftax.craftax_classic.envs.craftax_pixels_env import (
                    CraftaxClassicPixelsEnv,
                )

                return CraftaxClassicPixelsEnv()
            else:
                from craftax.craftax_classic.envs.craftax_pixels_env import (
                    CraftaxClassicPixelsEnvNoAutoReset,
                )

                return CraftaxClassicPixelsEnvNoAutoReset()
    else:
        if symbolic:
            if auto_reset:
                from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

                return CraftaxSymbolicEnv()
            else:
                from craftax.craftax.envs.craftax_symbolic_env import (
                    CraftaxSymbolicEnvNoAutoReset,
                )

                return CraftaxSymbolicEnvNoAutoReset()
        else:
            if auto_reset:
                from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

                return CraftaxPixelsEnv()
            else:
                from craftax.craftax.envs.craftax_pixels_env import (
                    CraftaxPixelsEnvNoAutoReset,
                )

                return CraftaxPixelsEnvNoAutoReset()


def make_craftax_flow_ready_env_from_name(
    name: str,
    include_relative_positions: bool = True,
    use_warp: bool = False,
):
    """
    Create a Craftax environment that prepares all inputs for external flow logic.

    This environment does NOT compute flow rewards or task completion internally.
    Instead, it saves all intermediate values in the info dict so they can be
    used externally to compute flow logic without JAX recompilation on module changes.

    Args:
        name: Environment name (only "Craftax-Symbolic-v1" supported currently)
        include_relative_positions: Whether to include relative positions in obs
        use_warp: Whether to use the optimized warp step function

    Returns:
        CraftaxSymbolicEnvNoAutoResetFlowReady instance
    """
    if name == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env_flow_ready import (
            CraftaxSymbolicEnvNoAutoResetFlowReady,
        )
        return CraftaxSymbolicEnvNoAutoResetFlowReady(
            include_relative_positions=include_relative_positions,
            use_warp=use_warp,
        )
    else:
        raise ValueError(
            f"Flow-ready environment not supported for {name}. "
            f"Only 'Craftax-Symbolic-v1' is currently supported."
        )
