#!/usr/bin/env python3
"""
Test script to verify Fabrax environment registration works correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np

def test_fabrax_environments():
    """Test that all Fabrax environments can be created and initialized."""

    from craftax.craftax_env import make_craftax_env_from_name

    print("🧪 Testing Fabrax Environment Registration...")

    # Test environment names to verify
    test_envs = [
        "Fabrax-Symbolic-v1",
        "Fabrax-Symbolic-AutoReset-v1",
        "Fabrax-Pixels-v1",
        "Fabrax-Pixels-AutoReset-v1"
    ]

    for env_name in test_envs:
        try:
            print(f"  Testing {env_name}...")

            # Create environment
            env = make_craftax_env_from_name(env_name, auto_reset=True)
            env_params = env.default_params

            # Test reset
            rng = jax.random.PRNGKey(42)
            obs, state = env.reset(rng, env_params)

            # Test that we have the right number of achievements (54 total)
            assert len(state.achievements) == 54, f"Expected 54 achievements, got {len(state.achievements)}"

            # Test a basic action
            action = 0  # NOOP
            obs, state, reward, done, info = env.step(rng, state, action, env_params)

            print(f"    ✅ {env_name} - SUCCESS!")
            print(f"       - Total achievements: {len(state.achievements)}")
            print(f"       - Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

        except Exception as e:
            print(f"    ❌ {env_name} - FAILED: {str(e)}")
            return False

    print("🎉 All Fabrax environments registered and working correctly!")
    return True

def test_new_fabrax_features():
    """Test that new Fabrax-specific features are present."""

    from craftax.fabrax.constants import Achievement, Action, BlockType

    print("\\n🔧 Testing Fabrax Features...")

    # Test new achievements
    expected_new_achievements = [
        "PLACE_ANVIL", "MAKE_BELLOWS", "SMELT_IRON_BAR", "COLLECT_COPPER",
        "COLLECT_TIN", "MAKE_STEEL_BAR", "MAKE_BRONZE_BAR", "FORGE_STEEL_PICKAXE",
        "FORGE_BRONZE_PICKAXE", "HARDEN_EDGE", "FORGE_STEEL_SWORD", "FORGE_BRONZE_SWORD",
        "COLLECT_SAND", "PLACE_KILN", "SMELT_GLASS", "COLLECT_CLAY", "FIRE_BRICK",
        "MAKE_GLASS_BOTTLE", "MAKE_LENS", "COLLECT_LIMESTONE", "MAKE_LIME",
        "MAKE_TELESCOPE", "MAKE_MORTAR", "PLACE_WINDOW", "PLACE_WALL_MASONRY",
        "PLACE_COMPOSTER", "MAKE_TAR", "MAKE_FERTILIZER", "PLACE_ALCHEMY_BENCH",
        "BREW_TONIC", "BREW_STONE_SKIN", "MAKE_FLUX"
    ]

    # Verify new achievements exist
    for achievement_name in expected_new_achievements:
        try:
            achievement = getattr(Achievement, achievement_name)
            print(f"    ✅ Achievement {achievement_name} (ID: {achievement.value})")
        except AttributeError:
            print(f"    ❌ Missing achievement: {achievement_name}")
            return False

    # Test new blocks
    new_blocks = ["COPPER", "TIN", "CLAY", "LIMESTONE", "ANVIL", "KILN", "COMPOSTER", "ALCHEMY_BENCH"]
    for block_name in new_blocks:
        try:
            block = getattr(BlockType, block_name)
            print(f"    ✅ Block {block_name} (ID: {block.value})")
        except AttributeError:
            print(f"    ❌ Missing block: {block_name}")
            return False

    # Test new actions
    new_actions = ["PLACE_ANVIL", "SMELT_IRON_BAR", "FORGE_STEEL_PICKAXE", "SMELT_GLASS", "BREW_TONIC"]
    for action_name in new_actions:
        try:
            action = getattr(Action, action_name)
            print(f"    ✅ Action {action_name} (ID: {action.value})")
        except AttributeError:
            print(f"    ❌ Missing action: {action_name}")
            return False

    print(f"\\n🎯 Fabrax Features Summary:")
    print(f"   - Total Achievements: {len(Achievement)} (22 classic + 32 new)")
    print(f"   - Total Block Types: {len(BlockType)} (17 classic + 8 new)")
    print(f"   - Total Actions: {len(Action)} (17 classic + 27 new)")

    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 FABRAX ENVIRONMENT REGISTRATION TEST")
    print("=" * 60)

    # Test environment registration
    if not test_fabrax_environments():
        print("\\n❌ Environment registration test FAILED!")
        exit(1)

    # Test Fabrax features
    if not test_new_fabrax_features():
        print("\\n❌ Fabrax features test FAILED!")
        exit(1)

    print("\\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Fabrax is ready to play!")
    print("🎮 Run: python play_fabrax.py")
    print("=" * 60)