import jax
import jax.numpy as jnp
from functools import partial

from craftax.craftax.constants import *
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable


def _tile_to_pixels(tile_rgba_or_rgb: jnp.ndarray, block_pixel_size: int) -> jnp.ndarray:
    """Convert [H, W, ts, ts, C] tiles to [H*ts, W*ts, C] pixels."""
    return tile_rgba_or_rgb.transpose(0, 2, 1, 3, 4).reshape(
        OBS_DIM[0] * block_pixel_size, OBS_DIM[1] * block_pixel_size, -1
    )


def _pad_small_rgb_to_tile(rgb_small: jnp.ndarray, block_pixel_size: int) -> jnp.ndarray:
    small = int(block_pixel_size * 0.8)
    left = (block_pixel_size - small) // 2
    right = block_pixel_size - small - left
    return jnp.pad(rgb_small, ((left, right), (left, right), (0, 0)))


def _pad_digit_to_tile(digit_rgb: jnp.ndarray, digit_alpha: jnp.ndarray, top: int, left: int, block_pixel_size: int):
    bottom = block_pixel_size - top - digit_rgb.shape[0]
    right = block_pixel_size - left - digit_rgb.shape[1]
    rgb = jnp.pad(digit_rgb, ((top, bottom), (left, right), (0, 0)))
    a = jnp.pad(digit_alpha, ((top, bottom), (left, right)))
    return rgb, a


def _blend_over(dst: jnp.ndarray, src_rgb_premul: jnp.ndarray, src_alpha: jnp.ndarray) -> jnp.ndarray:
    return dst * (1.0 - src_alpha[:, :, None]) + src_rgb_premul


def _blend_tile_layer(base_pixels: jnp.ndarray, tile_alpha: jnp.ndarray, tile_src: jnp.ndarray, block_pixel_size: int) -> jnp.ndarray:
    """Blend a per-tile source layer over a pixel image.

    Args:
        base_pixels: [H*b, W*b, 3]
        tile_alpha:  [H, W, b, b] alpha in [0,1]
        tile_src:    [H, W, b, b, 3] premultiplied source RGB (already includes alpha factor)
    """
    h, w = OBS_DIM
    b = block_pixel_size
    alpha_px = tile_alpha.transpose(0, 2, 1, 3).reshape(h * b, w * b)
    src_px = tile_src.transpose(0, 2, 1, 3, 4).reshape(h * b, w * b, 3)
    mask = alpha_px > 0
    return jnp.where(mask[:, :, None], src_px, base_pixels)


def _entity_tile_mask(local_pos: jnp.ndarray) -> jnp.ndarray:
    """One-hot tile mask for entity positions.

    local_pos: [N, 2] (y, x) in tile coordinates relative to obs window.
    Returns: [N, H, W] boolean.
    """
    ys = jnp.arange(OBS_DIM[0], dtype=jnp.int32)[None, :, None]
    xs = jnp.arange(OBS_DIM[1], dtype=jnp.int32)[None, None, :]
    y_eq = ys == local_pos[:, 0:1, None]
    x_eq = xs == local_pos[:, 1:2, None]
    return (y_eq & x_eq)


def _mobs_to_tile_layer(
    state: EnvState,
    mobs,
    mob_textures: jnp.ndarray,
    mob_texture_alphas: jnp.ndarray,
    obs_dim_array: jnp.ndarray,
    block_pixel_size: int,
):
    """Create a per-tile blended layer for a mob group.

    Uses a max-priority z-buffer per tile (last index wins) and blends exactly like the
    reference renderer for a single sprite per tile.
    """
    lvl = state.player_level
    positions = mobs.position[lvl]  # [N, 2]
    local_position = (
        positions
        - state.player_position[None, :]
        + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
    ).astype(jnp.int32)  # [N, 2]
    on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(axis=1)
    on_screen = on_screen & mobs.mask[lvl].astype(bool)

    n = mobs.mask.shape[1]
    tile_mask = _entity_tile_mask(local_position) & on_screen[:, None, None]

    # Per-tile z-buffer: last index wins.
    priorities = (jnp.arange(n, dtype=jnp.int32) + 1)[:, None, None]
    scored = tile_mask.astype(jnp.int32) * priorities  # [N, H, W]
    best = scored.max(axis=0)  # [H, W]
    best_idx = jnp.argmax(scored, axis=0).astype(jnp.int32)  # [H, W]
    sel_idx = jnp.where(best > 0, best_idx + 1, 0)  # [H, W] (0 = none)

    type_id = mobs.type_id[lvl].astype(jnp.int32)  # [N]
    tex = mob_textures[type_id].astype(jnp.uint8)  # [N, b, b, 3] (premultiplied)
    alpha_rgb = mob_texture_alphas[type_id].astype(jnp.uint8)  # [N, b, b, 3] (0/1)
    src = tex * alpha_rgb
    alpha = alpha_rgb[:, :, :, 0]  # [N, b, b]

    zeros_src = jnp.zeros((1, block_pixel_size, block_pixel_size, 3), dtype=jnp.uint8)
    zeros_a = jnp.zeros((1, block_pixel_size, block_pixel_size), dtype=jnp.uint8)
    src_pad = jnp.concatenate([zeros_src, src], axis=0)
    alpha_pad = jnp.concatenate([zeros_a, alpha], axis=0)

    tile_src = src_pad[sel_idx]  # [H, W, b, b, 3]
    tile_alpha = alpha_pad[sel_idx]  # [H, W, b, b]
    return tile_alpha, tile_src


def _projectiles_to_tile_layer(
    state: EnvState,
    projectiles,
    projectile_directions: jnp.ndarray,
    projectile_textures: jnp.ndarray,
    projectile_texture_alphas: jnp.ndarray,
    obs_dim_array: jnp.ndarray,
    block_pixel_size: int,
):
    """Create a per-tile blended layer for a projectile group (direction-aware)."""
    lvl = state.player_level
    positions = projectiles.position[lvl]  # [N, 2]
    local_position = (
        positions
        - state.player_position[None, :]
        + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
    ).astype(jnp.int32)  # [N, 2]
    on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all(axis=1)
    on_screen = on_screen & projectiles.mask[lvl].astype(bool)

    n = projectiles.mask.shape[1]
    tile_mask = _entity_tile_mask(local_position) & on_screen[:, None, None]

    priorities = (jnp.arange(n, dtype=jnp.int32) + 1)[:, None, None]
    scored = tile_mask.astype(jnp.int32) * priorities
    best = scored.max(axis=0)
    best_idx = jnp.argmax(scored, axis=0).astype(jnp.int32)
    sel_idx = jnp.where(best > 0, best_idx + 1, 0)

    type_id = projectiles.type_id[lvl].astype(jnp.int32)  # [N]
    tex = projectile_textures[type_id].astype(jnp.uint8)  # [N, b, b, 3] premultiplied
    alpha_rgb = projectile_texture_alphas[type_id].astype(jnp.uint8)  # [N, b, b, 3] (0/1)

    dirs = projectile_directions[lvl]  # [N, 2]
    flip = jnp.logical_or(dirs[:, 0] > 0, dirs[:, 1] > 0)
    transpose = dirs[:, 1] != 0

    tex_flipped = jnp.flip(tex, axis=1)
    alpha_flipped = jnp.flip(alpha_rgb, axis=1)
    tex = jnp.where(flip[:, None, None, None], tex_flipped, tex)
    alpha_rgb = jnp.where(flip[:, None, None, None], alpha_flipped, alpha_rgb)

    tex_t = jnp.transpose(tex, (0, 2, 1, 3))
    alpha_t = jnp.transpose(alpha_rgb, (0, 2, 1, 3))
    tex = jnp.where(transpose[:, None, None, None], tex_t, tex)
    alpha_rgb = jnp.where(transpose[:, None, None, None], alpha_t, alpha_rgb)

    src = tex * alpha_rgb
    alpha = alpha_rgb[:, :, :, 0]

    zeros_src = jnp.zeros((1, block_pixel_size, block_pixel_size, 3), dtype=jnp.uint8)
    zeros_a = jnp.zeros((1, block_pixel_size, block_pixel_size), dtype=jnp.uint8)
    src_pad = jnp.concatenate([zeros_src, src], axis=0)
    alpha_pad = jnp.concatenate([zeros_a, alpha], axis=0)

    tile_src = src_pad[sel_idx]
    tile_alpha = alpha_pad[sel_idx]
    return tile_alpha, tile_src


def _render_inventory_pixels_fast(state: EnvState, textures, block_pixel_size: int) -> jnp.ndarray:
    """Build inventory HUD via per-tile composition, avoiding many pixel-level scatters."""
    ts = int(block_pixel_size)
    number_size = int(block_pixel_size * 0.4)
    number_offset = ts - number_size
    number_double_offset = ts - 2 * number_size

    empty_tile = jnp.zeros((ts, ts, 3), dtype=jnp.uint8)

    def icon_tile(rgb_small: jnp.ndarray) -> jnp.ndarray:
        return _pad_small_rgb_to_tile(rgb_small.astype(jnp.uint8), ts)

    def overlay_center(tile: jnp.ndarray, rgba_small: jnp.ndarray) -> jnp.ndarray:
        rgb = rgba_small[:, :, :3].astype(jnp.uint8)
        a = rgba_small[:, :, 3].astype(jnp.uint8)
        rgb_p = _pad_small_rgb_to_tile(rgb, ts)
        a_p = _pad_small_rgb_to_tile(a[:, :, None], ts)[:, :, 0]
        return jnp.where(a_p[:, :, None] > 0, rgb_p, tile)

    def draw_digit(tile: jnp.ndarray, digit: jnp.ndarray) -> jnp.ndarray:
        digit_i = digit.astype(jnp.int32)
        tex = textures["number_textures"][digit_i].astype(jnp.uint8)
        a = textures["number_textures_alpha"][digit_i][:, :, 0].astype(jnp.uint8)
        tex_p, a_p = _pad_digit_to_tile(tex, a, number_offset, number_offset, ts)
        return jnp.where(a_p[:, :, None] > 0, tex_p, tile)

    def draw_two_digit(tile: jnp.ndarray, number: jnp.ndarray) -> jnp.ndarray:
        n = number.astype(jnp.int32)
        tens = n // 10
        ones = n % 10

        ones_tex_set = jax.lax.select(n == 0, textures["number_textures"], textures["number_textures_with_zero"])
        ones_alpha_set = jax.lax.select(
            n == 0, textures["number_textures_alpha"], textures["number_textures_alpha_with_zero"]
        )

        ones_tex = ones_tex_set[ones].astype(jnp.uint8)
        ones_a = ones_alpha_set[ones][:, :, 0].astype(jnp.uint8)
        ones_p, ones_ap = _pad_digit_to_tile(ones_tex, ones_a, number_offset, number_offset, ts)
        tile = jnp.where(ones_ap[:, :, None] > 0, ones_p, tile)

        tens_tex = textures["number_textures"][tens].astype(jnp.uint8)
        tens_a = textures["number_textures_alpha"][tens][:, :, 0].astype(jnp.uint8)
        tens_p, tens_ap = _pad_digit_to_tile(tens_tex, tens_a, number_offset, number_double_offset, ts)
        tile_t = jnp.where(tens_ap[:, :, None] > 0, tens_p, tile)
        return jnp.where(tens != 0, tile_t, tile)

    # --- Compute icon tiles and numbers for each HUD cell (4 rows x 11 cols) ---
    player_health = jnp.maximum(jnp.floor(state.player_health), 1).astype(jnp.int32)

    health_icon = jax.lax.select(player_health > 0, textures["health_texture"], textures["smaller_empty_texture"])
    hunger_icon = jax.lax.select(state.player_food > 0, textures["hunger_texture"], textures["smaller_empty_texture"])
    thirst_icon = jax.lax.select(state.player_drink > 0, textures["thirst_texture"], textures["smaller_empty_texture"])
    energy_icon = jax.lax.select(state.player_energy > 0, textures["energy_texture"], textures["smaller_empty_texture"])
    mana_icon = jax.lax.select(state.player_mana > 0, textures["mana_texture"], textures["smaller_empty_texture"])

    inv_wood_icon = jax.lax.select(
        state.inventory.wood > 0,
        textures["smaller_block_textures"][BlockType.WOOD.value],
        textures["smaller_empty_texture"],
    )
    inv_stone_icon = jax.lax.select(
        state.inventory.stone > 0,
        textures["smaller_block_textures"][BlockType.STONE.value],
        textures["smaller_empty_texture"],
    )
    inv_coal_icon = jax.lax.select(
        state.inventory.coal > 0,
        textures["smaller_block_textures"][BlockType.COAL.value],
        textures["smaller_empty_texture"],
    )
    inv_iron_icon = jax.lax.select(
        state.inventory.iron > 0,
        textures["smaller_block_textures"][BlockType.IRON.value],
        textures["smaller_empty_texture"],
    )
    inv_diamond_icon = jax.lax.select(
        state.inventory.diamond > 0,
        textures["smaller_block_textures"][BlockType.DIAMOND.value],
        textures["smaller_empty_texture"],
    )
    inv_sapphire_icon = jax.lax.select(
        state.inventory.sapphire > 0,
        textures["smaller_block_textures"][BlockType.SAPPHIRE.value],
        textures["smaller_empty_texture"],
    )
    inv_ruby_icon = jax.lax.select(
        state.inventory.ruby > 0,
        textures["smaller_block_textures"][BlockType.RUBY.value],
        textures["smaller_empty_texture"],
    )
    inv_sapling_icon = jax.lax.select(
        state.inventory.sapling > 0,
        textures["sapling_texture"],
        textures["smaller_empty_texture"],
    )

    pickaxe_icon = textures["pickaxe_textures"][state.inventory.pickaxe]
    sword_icon = textures["sword_textures"][state.inventory.sword]
    bow_icon = textures["bow_textures"][state.inventory.bow]

    torch_icon = jax.lax.select(
        state.inventory.torches > 0,
        textures["torch_inv_texture"],
        textures["smaller_empty_texture"],
    )
    arrow_icon = jax.lax.select(
        state.inventory.arrows > 0,
        textures["player_projectile_textures"][0],
        textures["smaller_empty_texture"],
    )
    book_icon = jax.lax.select(
        state.inventory.books > 0,
        textures["book_texture"],
        textures["smaller_empty_texture"],
    )
    fireball_icon = jax.lax.select(
        state.learned_spells[0],
        textures["fireball_inv_texture"],
        textures["smaller_empty_texture"],
    )
    iceball_icon = jax.lax.select(
        state.learned_spells[1],
        textures["iceball_inv_texture"],
        textures["smaller_empty_texture"],
    )

    xp_icon = jax.lax.select(
        state.player_xp > 0,
        textures["xp_texture"],
        textures["smaller_empty_texture"],
    )

    # Armour slots
    armour_tiles = [
        textures["armour_textures"][state.inventory.armour[i], i] for i in range(4)
    ]

    # Potions (row 3)
    potion_icons = [
        jax.lax.select(
            state.inventory.potions[i] > 0,
            textures["potion_textures"][i],
            textures["smaller_empty_texture"],
        )
        for i in range(6)
    ]

    # Row 0
    r0 = [empty_tile] * OBS_DIM[1]
    r0[0] = draw_two_digit(icon_tile(health_icon), player_health)
    r0[1] = draw_two_digit(icon_tile(hunger_icon), state.player_food)
    r0[2] = draw_two_digit(icon_tile(thirst_icon), state.player_drink)
    r0[3] = draw_two_digit(icon_tile(energy_icon), state.player_energy)
    r0[4] = draw_two_digit(icon_tile(mana_icon), state.player_mana)
    r0[6] = draw_digit(empty_tile, state.player_level)
    r0[8] = icon_tile(pickaxe_icon)
    r0[9] = draw_digit(icon_tile(xp_icon), state.player_xp)

    # Row 1
    r1 = [empty_tile] * OBS_DIM[1]
    r1[0] = draw_two_digit(icon_tile(inv_coal_icon), state.inventory.coal)
    r1[1] = draw_two_digit(icon_tile(inv_iron_icon), state.inventory.iron)
    r1[2] = draw_two_digit(icon_tile(inv_diamond_icon), state.inventory.diamond)
    r1[3] = draw_two_digit(icon_tile(inv_sapphire_icon), state.inventory.sapphire)
    r1[4] = draw_two_digit(icon_tile(inv_ruby_icon), state.inventory.ruby)
    r1[5] = draw_two_digit(icon_tile(inv_sapling_icon), state.inventory.sapling)
    r1[6] = icon_tile(bow_icon)
    r1[7] = icon_tile(armour_tiles[1])
    r1[8] = icon_tile(sword_icon)
    r1[9] = draw_digit(icon_tile(textures["dex_texture"]), state.player_dexterity)

    # Sword enchantment overlay (x=8,y=1)
    sword_ench = textures["sword_enchantment_textures"][state.sword_enchantment]
    r1[8] = overlay_center(r1[8], sword_ench)

    # Armour enchantment overlays (x=7, y=0..3)
    for i in range(4):
        ench = textures["armour_enchantment_textures"][state.armour_enchantments[i], i]
        if i == 0:
            r0[7] = overlay_center(icon_tile(armour_tiles[0]), ench)
        elif i == 1:
            r1[7] = overlay_center(r1[7], ench)
        elif i == 2:
            # Will be set in row 2; placeholder here.
            pass
        else:
            # Row 3
            pass

    # Row 2
    r2 = [empty_tile] * OBS_DIM[1]
    r2[0] = draw_two_digit(icon_tile(inv_wood_icon), state.inventory.wood)
    r2[1] = draw_two_digit(icon_tile(inv_stone_icon), state.inventory.stone)
    r2[2] = draw_two_digit(icon_tile(torch_icon), state.inventory.torches)
    r2[3] = draw_two_digit(icon_tile(book_icon), state.inventory.books)
    r2[4] = icon_tile(fireball_icon)
    r2[5] = icon_tile(iceball_icon)
    r2[6] = draw_two_digit(icon_tile(arrow_icon), state.inventory.arrows)
    r2[7] = icon_tile(armour_tiles[2])
    r2[9] = draw_digit(icon_tile(textures["str_texture"]), state.player_strength)

    # Arrow enchantment overlay (x=6,y=2)
    arrow_ench_level = state.bow_enchantment * (state.inventory.arrows > 0)
    arrow_ench = textures["arrow_enchantment_textures"][arrow_ench_level]
    r2[6] = overlay_center(r2[6], arrow_ench)

    # Armour enchantment overlay for slot 2
    ench2 = textures["armour_enchantment_textures"][state.armour_enchantments[2], 2]
    r2[7] = overlay_center(r2[7], ench2)

    # Row 3
    r3 = [empty_tile] * OBS_DIM[1]
    for i in range(6):
        r3[i] = draw_two_digit(icon_tile(potion_icons[i]), state.inventory.potions[i])
    r3[7] = icon_tile(armour_tiles[3])
    r3[9] = draw_digit(icon_tile(textures["int_texture"]), state.player_intelligence)

    # Armour enchantment overlay for slot 3
    ench3 = textures["armour_enchantment_textures"][state.armour_enchantments[3], 3]
    r3[7] = overlay_center(r3[7], ench3)

    inv_tiles = jnp.stack(
        [jnp.stack(r0, axis=0), jnp.stack(r1, axis=0), jnp.stack(r2, axis=0), jnp.stack(r3, axis=0)],
        axis=0,
    )  # [4, 11, ts, ts, 3]

    return inv_tiles.transpose(0, 2, 1, 3, 4).reshape(INVENTORY_OBS_HEIGHT * ts, OBS_DIM[1] * ts, 3)


@partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def render_craftax_pixels_fast(state: EnvState, block_pixel_size: int, do_night_noise: bool = True):
    """TPU-friendlier pixel renderer.

    Produces the same output image as `craftax.craftax.renderer.render_craftax_pixels`,
    but avoids the O(|BlockType| + |ItemType|) per-pixel scan by using gathers at
    tile resolution followed by reshape.
    """
    textures = TEXTURES[block_pixel_size]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # RENDER MAP
    map = state.map[state.player_level]
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    boss_block = jax.lax.select(
        is_boss_vulnerable(state),
        BlockType.NECROMANCER_VULNERABLE.value,
        BlockType.NECROMANCER.value,
    )
    map_view_boss = map_view == BlockType.NECROMANCER.value
    map_view = map_view_boss * boss_block + (1 - map_view_boss) * map_view

    # Gather tile textures and reshape (instead of per-type scan at pixel resolution).
    block_textures = textures["block_textures"].astype(jnp.uint8)  # [B, ts, ts, 3]
    map_tiles = block_textures[map_view]  # [H, W, ts, ts, 3]
    map_pixels = _tile_to_pixels(map_tiles, block_pixel_size)  # [H*ts, W*ts, 3]

    # ITEMS
    padded_item_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )
    item_map_view = jax.lax.dynamic_slice(padded_item_map, tl_corner, OBS_DIM)

    is_ladder_down_open = (
        state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    ladder_down_item = jax.lax.select(
        is_ladder_down_open,
        ItemType.LADDER_DOWN.value,
        ItemType.LADDER_DOWN_BLOCKED.value,
    )
    item_map_view_is_ladder_down = item_map_view == ItemType.LADDER_DOWN.value
    item_map_view = (
        item_map_view_is_ladder_down * ladder_down_item
        + (1 - item_map_view_is_ladder_down) * item_map_view
    )

    # Gather per-tile RGBA item textures.
    item_textures = textures["item_textures"].astype(jnp.uint8)  # [I, ts, ts, 4]
    item_tiles = item_textures[item_map_view]  # [H, W, ts, ts, 4]
    item_pixels = _tile_to_pixels(item_tiles, block_pixel_size)  # [H*ts, W*ts, 4]

    # Mask out NONE (the reference renderer doesn't draw it at all).
    has_item_tiles = (item_map_view != ItemType.NONE.value).astype(jnp.float32)  # [H, W]
    has_item_pixels = has_item_tiles.repeat(block_pixel_size, axis=0).repeat(
        block_pixel_size, axis=1
    )  # [H*ts, W*ts]

    item_mask = (item_pixels[:, :, 3] > 0) & (has_item_pixels > 0)
    map_pixels = jnp.where(item_mask[:, :, None], item_pixels[:, :, :3], map_pixels)

    # Render player (center-tile patch, equivalent to full-screen overlay but cheaper).
    player_texture_index = jax.lax.select(
        state.is_sleeping, 4, state.player_direction - 1
    )
    player_rgba = textures["player_textures_rgba"][player_texture_index].astype(jnp.uint8)  # [ts, ts, 4]
    player_rgb = player_rgba[:, :, :3]
    player_a = player_rgba[:, :, 3] > 0

    py = (OBS_DIM[0] // 2) * block_pixel_size
    px = (OBS_DIM[1] // 2) * block_pixel_size
    patch = jax.lax.dynamic_slice(map_pixels, (py, px, 0), (block_pixel_size, block_pixel_size, 3))
    patch = jnp.where(player_a[:, :, None], player_rgb, patch)
    map_pixels = jax.lax.dynamic_update_slice(map_pixels, patch, (py, px, 0))

    # Render mobs/projectiles via per-tile z-buffer (gather/reshape + 1 blend per group).
    a_melee, src_melee = _mobs_to_tile_layer(
        state,
        state.melee_mobs,
        textures["melee_mob_textures"],
        textures["melee_mob_texture_alphas"],
        obs_dim_array,
        block_pixel_size,
    )
    map_pixels = _blend_tile_layer(map_pixels, a_melee, src_melee, block_pixel_size)

    a_passive, src_passive = _mobs_to_tile_layer(
        state,
        state.passive_mobs,
        textures["passive_mob_textures"],
        textures["passive_mob_texture_alphas"],
        obs_dim_array,
        block_pixel_size,
    )
    map_pixels = _blend_tile_layer(map_pixels, a_passive, src_passive, block_pixel_size)

    a_ranged, src_ranged = _mobs_to_tile_layer(
        state,
        state.ranged_mobs,
        textures["ranged_mob_textures"],
        textures["ranged_mob_texture_alphas"],
        obs_dim_array,
        block_pixel_size,
    )
    map_pixels = _blend_tile_layer(map_pixels, a_ranged, src_ranged, block_pixel_size)

    a_mob_proj, src_mob_proj = _projectiles_to_tile_layer(
        state,
        state.mob_projectiles,
        state.mob_projectile_directions,
        textures["projectile_textures"],
        textures["projectile_texture_alphas"],
        obs_dim_array,
        block_pixel_size,
    )
    map_pixels = _blend_tile_layer(map_pixels, a_mob_proj, src_mob_proj, block_pixel_size)

    a_player_proj, src_player_proj = _projectiles_to_tile_layer(
        state,
        state.player_projectiles,
        state.player_projectile_directions,
        textures["projectile_textures"],
        textures["projectile_texture_alphas"],
        obs_dim_array,
        block_pixel_size,
    )
    map_pixels = _blend_tile_layer(map_pixels, a_player_proj, src_player_proj, block_pixel_size)

    # Switch to float32 for lighting/night effects (and downstream normalization).
    map_pixels = map_pixels.astype(jnp.float32)

    # Apply darkness (underground)
    light_map = state.light_map[state.player_level]
    padded_light_map = jnp.pad(
        light_map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=False,
    )
    light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, OBS_DIM)
    light_map_pixels = light_map_view.repeat(block_pixel_size, axis=0).repeat(
        block_pixel_size, axis=1
    )
    map_pixels = (light_map_pixels)[:, :, None] * map_pixels

    # Apply night
    night_pixels = textures["night_texture"]
    daylight = state.light_level
    daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)

    if do_night_noise:
        night_noise = (
            jax.random.uniform(state.state_rng, night_pixels.shape[:2]) * 95 + 32
        )
        night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        night_mask = textures["night_noise_intensity_texture"] * night_intensity
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night
    else:
        night_noise = jnp.ones(night_pixels.shape[:2]) * 64
        night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)

        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        night_mask = (
            jnp.ones_like(textures["night_noise_intensity_texture"])
            * night_intensity
            * 0.5
        )
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise

        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night

    # Apply sleep
    sleep_pixels = jnp.zeros_like(map_pixels)
    sleep_level = 1.0 - state.is_sleeping * 0.5
    map_pixels = sleep_level * map_pixels + (1 - sleep_level) * sleep_pixels

    inv_pixels = _render_inventory_pixels_fast(state, textures, block_pixel_size)
    inv_pixels = inv_pixels.astype(jnp.float32)

    pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)
    return pixels
