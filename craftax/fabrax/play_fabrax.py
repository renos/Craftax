import argparse
import sys

import pygame

import jax
import jax.numpy as jnp
import numpy as np

from craftax.fabrax.constants import (
    OBS_DIM,
    INVENTORY_OBS_HEIGHT,
    Action,
    Achievement,
    BLOCK_PIXEL_SIZE_HUMAN,
)
from craftax.fabrax.envs.craftax_symbolic_env import (
    FabraxSymbolicEnv as CraftaxEnv,
)
from craftax.fabrax.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

KEY_MAPPING = {
    # Basic controls
    pygame.K_q: Action.NOOP,
    pygame.K_w: Action.UP,
    pygame.K_d: Action.RIGHT,
    pygame.K_s: Action.DOWN,
    pygame.K_a: Action.LEFT,
    pygame.K_SPACE: Action.DO,
    pygame.K_TAB: Action.SLEEP,

    # Original Craftax placement/crafting
    pygame.K_t: Action.PLACE_TABLE,
    pygame.K_r: Action.PLACE_STONE,
    pygame.K_f: Action.PLACE_FURNACE,
    pygame.K_p: Action.PLACE_PLANT,
    pygame.K_1: Action.MAKE_WOOD_PICKAXE,
    pygame.K_2: Action.MAKE_STONE_PICKAXE,
    pygame.K_3: Action.MAKE_IRON_PICKAXE,
    pygame.K_4: Action.MAKE_WOOD_SWORD,
    pygame.K_5: Action.MAKE_STONE_SWORD,
    pygame.K_6: Action.MAKE_IRON_SWORD,

    # NEW FABRAX CONTROLS
    # Station placement
    pygame.K_u: Action.PLACE_ANVIL,      # U for anviL (close to A)
    pygame.K_k: Action.PLACE_KILN,       # K for Kiln
    pygame.K_c: Action.PLACE_COMPOSTER,  # C for Composter
    pygame.K_b: Action.PLACE_ALCHEMY_BENCH,  # B for alchemy Bench

    # Advanced crafting (Shift + number keys)
    pygame.K_7: Action.MAKE_BELLOWS,
    pygame.K_8: Action.SMELT_IRON_BAR,
    pygame.K_9: Action.MAKE_STEEL_BAR,
    pygame.K_0: Action.MAKE_BRONZE_BAR,

    # Forging (Ctrl + number keys - we'll use letters for simplicity)
    pygame.K_z: Action.FORGE_STEEL_PICKAXE,
    pygame.K_x: Action.FORGE_BRONZE_PICKAXE,
    pygame.K_v: Action.FORGE_STEEL_SWORD,
    pygame.K_n: Action.FORGE_BRONZE_SWORD,

    # Glass & ceramics (G prefix conceptually)
    pygame.K_g: Action.SMELT_GLASS,
    pygame.K_h: Action.FIRE_BRICK,
    pygame.K_j: Action.MAKE_GLASS_BOTTLE,
    pygame.K_l: Action.MAKE_LENS,
    pygame.K_o: Action.MAKE_TELESCOPE,
    pygame.K_m: Action.MAKE_LIME,
    pygame.K_COMMA: Action.MAKE_MORTAR,

    # Chemistry & alchemy
    pygame.K_y: Action.MAKE_TAR,
    pygame.K_i: Action.MAKE_FERTILIZER,
    pygame.K_LEFTBRACKET: Action.BREW_TONIC,
    pygame.K_RIGHTBRACKET: Action.BREW_STONE_SKIN,
    pygame.K_BACKSLASH: Action.MAKE_FLUX,

    # Special actions
    pygame.K_e: Action.HARDEN_EDGE,      # E for Edge hardening
    pygame.K_PERIOD: Action.PLACE_WINDOW,
    pygame.K_SLASH: Action.PLACE_WALL_MASONRY,
}


class CraftaxRenderer:
    def __init__(self, env: CraftaxEnv, env_params, pixel_render_size=4):
        self.env = env
        self.env_params = env_params
        self.pixel_render_size = pixel_render_size
        self.pygame_events = []

        self.screen_size = (
            OBS_DIM[1] * BLOCK_PIXEL_SIZE_HUMAN * pixel_render_size,
            (OBS_DIM[0] + INVENTORY_OBS_HEIGHT)
            * BLOCK_PIXEL_SIZE_HUMAN
            * pixel_render_size,
        )

        # Init rendering
        pygame.init()
        pygame.key.set_repeat(250, 75)

        self.screen_surface = pygame.display.set_mode(self.screen_size)

        self._render = jax.jit(render_craftax_pixels, static_argnums=(1,))

    def update(self):
        # Update pygame events
        self.pygame_events = list(pygame.event.get())

        # Update screen
        pygame.display.flip()
        # time.sleep(0.01)

    def render(self, env_state):
        # Clear
        self.screen_surface.fill((0, 0, 0))

        pixels = self._render(env_state, block_pixel_size=BLOCK_PIXEL_SIZE_HUMAN)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)

        surface = pygame.surfarray.make_surface(np.array(pixels).transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, 0))

    def is_quit_requested(self):
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                return True
        return False

    def get_action_from_keypress(self, state):
        if state.is_sleeping:
            return Action.NOOP.value
        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN:
                if event.key in KEY_MAPPING:
                    return KEY_MAPPING[event.key].value

        return None


def print_new_achievements(old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{Achievement(i).name} ({new_achievements.sum()}/{len(Achievement)})")


def main(args):
    env = make_craftax_env_from_name("Fabrax-Symbolic-v1", auto_reset=True)
    env_params = env.default_params
    print("\n🎉 Welcome to FABRAX! 🎉")
    print("Extended Craftax with Metallurgy, Glass & Ceramics, and Chemistry & Alchemy!")

    print("\n=== FABRAX CONTROLS ===")
    print("\nBasic Controls:")
    basic_controls = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_SPACE, pygame.K_TAB, pygame.K_q]
    for k in basic_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print("\nOriginal Craftax Placement:")
    placement_controls = [pygame.K_t, pygame.K_r, pygame.K_f, pygame.K_p]
    for k in placement_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print("\nOriginal Tool Crafting:")
    tool_controls = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]
    for k in tool_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print("\nNEW FABRAX STATIONS:")
    station_controls = [pygame.K_u, pygame.K_k, pygame.K_c, pygame.K_b]
    for k in station_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print("\nNEW METALLURGY:")
    metal_controls = [pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0, pygame.K_z, pygame.K_x, pygame.K_v, pygame.K_n, pygame.K_e]
    for k in metal_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print("\nNEW GLASS & CERAMICS:")
    glass_controls = [pygame.K_g, pygame.K_h, pygame.K_j, pygame.K_l, pygame.K_o, pygame.K_m, pygame.K_COMMA, pygame.K_PERIOD, pygame.K_SLASH]
    for k in glass_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print("\nNEW CHEMISTRY & ALCHEMY:")
    chem_controls = [pygame.K_y, pygame.K_i, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_BACKSLASH]
    for k in chem_controls:
        if k in KEY_MAPPING:
            print(f"  {pygame.key.name(k)}: {KEY_MAPPING[k].name.lower()}")

    print(f"\nTotal Achievements: {len(Achievement)} (22 original + 32 new Fabrax)")
    print("\n" + "="*50)

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(rng)
    _, env_state = env.reset(_rng, env_params)

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=pixel_render_size)
    renderer.render(env_state)

    step_fn = jax.jit(env.step)

    clock = pygame.time.Clock()

    while not renderer.is_quit_requested():
        action = renderer.get_action_from_keypress(env_state)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = step_fn(
                _rng, env_state, action, env_params
            )
            new_achievements = env_state.achievements
            print_new_achievements(old_achievements, new_achievements)

            if reward > 0.01 or reward < -0.01:
                print(f"Reward: {reward}\n")

            renderer.render(env_state)

        renderer.update()
        clock.tick(args.fps)


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fps", type=int, default=60)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)


if __name__ == "__main__":
    entry_point()
