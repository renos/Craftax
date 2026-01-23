from craftax.craftax.util.relative_positions import update_closest_blocks, update_closest_blocks_per_floor
from craftax.craftax.util.game_logic_utils import *


# Helper function for mob collision check with explicit mob_map
def is_position_in_bounds_not_in_mob_not_colliding_with_mob_map(state, position, collision_map, in_mob):
    """Like is_position_in_bounds_not_in_mob_not_colliding but takes in_mob as parameter."""
    pos_in_bounds = in_bounds(state, position)
    in_solid_block = is_in_solid_block(state, position)
    in_lava = state.map[state.player_level, position[0], position[1]] == BlockType.LAVA.value
    in_water = state.map[state.player_level, position[0], position[1]] == BlockType.WATER.value
    on_ground_block = jnp.logical_and(
        jnp.logical_not(in_solid_block),
        jnp.logical_and(jnp.logical_not(in_water), jnp.logical_not(in_lava)),
    )

    valid_move = jnp.logical_and(
        pos_in_bounds,
        jnp.logical_and(jnp.logical_not(in_mob), jnp.logical_not(in_solid_block)),
    )

    # Ground blocks
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[0]), jnp.logical_not(on_ground_block)),
    )
    # Water
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[1]), jnp.logical_not(in_water)),
    )
    # Lava
    valid_move = jnp.logical_and(
        valid_move,
        jnp.logical_or(jnp.logical_not(collision_map[2]), jnp.logical_not(in_lava)),
    )

    return valid_move


# ========== MINING LOOKUP TABLES ==========
# BLOCK_REPLACEMENT[tile] = what block it becomes when mined (or itself if not mineable)
# BLOCK_REQUIRED_PICKAXE[tile] = minimum pickaxe level needed (255 = can't mine)
def _build_mining_tables():
    num_blocks = len(BlockType)
    # Default: block stays the same, requires pickaxe level 255 (unminenable)
    replacement = jnp.arange(num_blocks, dtype=jnp.int32)  # identity by default
    required_pickaxe = jnp.full(num_blocks, 255, dtype=jnp.int32)

    # Trees -> grass variants (no pickaxe needed)
    replacement = replacement.at[BlockType.TREE.value].set(BlockType.GRASS.value)
    replacement = replacement.at[BlockType.FIRE_TREE.value].set(BlockType.FIRE_GRASS.value)
    replacement = replacement.at[BlockType.ICE_SHRUB.value].set(BlockType.ICE_GRASS.value)
    required_pickaxe = required_pickaxe.at[BlockType.TREE.value].set(0)
    required_pickaxe = required_pickaxe.at[BlockType.FIRE_TREE.value].set(0)
    required_pickaxe = required_pickaxe.at[BlockType.ICE_SHRUB.value].set(0)

    # Ores -> PATH (various pickaxe requirements)
    for block, req in [
        (BlockType.STONE, 1),
        (BlockType.COAL, 1),
        (BlockType.STALAGMITE, 1),
        (BlockType.IRON, 2),
        (BlockType.DIAMOND, 3),
        (BlockType.SAPPHIRE, 4),
        (BlockType.RUBY, 4),
    ]:
        replacement = replacement.at[block.value].set(BlockType.PATH.value)
        required_pickaxe = required_pickaxe.at[block.value].set(req)

    # Structures -> PATH (no pickaxe needed)
    for block in [BlockType.FURNACE, BlockType.CRAFTING_TABLE, BlockType.CHEST]:
        replacement = replacement.at[block.value].set(BlockType.PATH.value)
        required_pickaxe = required_pickaxe.at[block.value].set(0)

    # Ripe plant -> unripe plant (no pickaxe needed)
    replacement = replacement.at[BlockType.RIPE_PLANT.value].set(BlockType.PLANT.value)
    required_pickaxe = required_pickaxe.at[BlockType.RIPE_PLANT.value].set(0)

    return replacement, required_pickaxe

BLOCK_REPLACEMENT, BLOCK_REQUIRED_PICKAXE = _build_mining_tables()


# ========== RESOURCE GAIN LOOKUP TABLES ==========
# BLOCK_GIVES_X[tile] = amount of resource X gained when mining tile (before can_mine check)
def _build_resource_tables():
    num_blocks = len(BlockType)

    # Wood: trees give 1 wood
    gives_wood = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_wood = gives_wood.at[BlockType.TREE.value].set(1)
    gives_wood = gives_wood.at[BlockType.FIRE_TREE.value].set(1)
    gives_wood = gives_wood.at[BlockType.ICE_SHRUB.value].set(1)

    # Stone: stone and stalagmite give 1 stone
    gives_stone = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_stone = gives_stone.at[BlockType.STONE.value].set(1)
    gives_stone = gives_stone.at[BlockType.STALAGMITE.value].set(1)

    # Coal
    gives_coal = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_coal = gives_coal.at[BlockType.COAL.value].set(1)

    # Iron
    gives_iron = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_iron = gives_iron.at[BlockType.IRON.value].set(1)

    # Diamond
    gives_diamond = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_diamond = gives_diamond.at[BlockType.DIAMOND.value].set(1)

    # Sapphire
    gives_sapphire = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_sapphire = gives_sapphire.at[BlockType.SAPPHIRE.value].set(1)

    # Ruby
    gives_ruby = jnp.zeros(num_blocks, dtype=jnp.int32)
    gives_ruby = gives_ruby.at[BlockType.RUBY.value].set(1)

    # Special blocks (for drink/food/chest logic)
    is_water_source = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_water_source = is_water_source.at[BlockType.WATER.value].set(True)
    is_water_source = is_water_source.at[BlockType.FOUNTAIN.value].set(True)

    is_ripe_plant = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_ripe_plant = is_ripe_plant.at[BlockType.RIPE_PLANT.value].set(True)

    is_chest = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_chest = is_chest.at[BlockType.CHEST.value].set(True)

    is_necromancer = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_necromancer = is_necromancer.at[BlockType.NECROMANCER.value].set(True)

    is_grass = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_grass = is_grass.at[BlockType.GRASS.value].set(True)

    return (gives_wood, gives_stone, gives_coal, gives_iron, gives_diamond,
            gives_sapphire, gives_ruby, is_water_source, is_ripe_plant,
            is_chest, is_necromancer, is_grass)

(BLOCK_GIVES_WOOD, BLOCK_GIVES_STONE, BLOCK_GIVES_COAL, BLOCK_GIVES_IRON,
 BLOCK_GIVES_DIAMOND, BLOCK_GIVES_SAPPHIRE, BLOCK_GIVES_RUBY,
 BLOCK_IS_WATER_SOURCE, BLOCK_IS_RIPE_PLANT, BLOCK_IS_CHEST,
 BLOCK_IS_NECROMANCER, BLOCK_IS_GRASS) = _build_resource_tables()

# Achievement indices for batched updates in do_action
DO_ACTION_ACHIEVEMENT_INDICES = jnp.array([
    Achievement.COLLECT_DRINK.value,
    Achievement.EAT_PLANT.value,
    Achievement.OPEN_CHEST.value,
    Achievement.DAMAGE_NECROMANCER.value,
], dtype=jnp.int32)

# Achievement indices for batched updates in place_block
PLACE_BLOCK_ACHIEVEMENT_INDICES = jnp.array([
    Achievement.PLACE_TABLE.value,
    Achievement.PLACE_FURNACE.value,
    Achievement.PLACE_STONE.value,
    Achievement.PLACE_TORCH.value,
    Achievement.PLACE_PLANT.value,
], dtype=jnp.int32)

# Achievement indices for batched updates in do_crafting
DO_CRAFTING_ACHIEVEMENT_INDICES = jnp.array([
    Achievement.MAKE_IRON_ARMOUR.value,
    Achievement.MAKE_DIAMOND_ARMOUR.value,
], dtype=jnp.int32)

# Achievement indices for batched updates in calculate_inventory_achievements
# These are split into groups by the condition type for clarity
INVENTORY_ACHIEVEMENT_COLLECT_INDICES = jnp.array([
    Achievement.COLLECT_WOOD.value,
    Achievement.COLLECT_STONE.value,
    Achievement.COLLECT_COAL.value,
    Achievement.COLLECT_IRON.value,
    Achievement.COLLECT_DIAMOND.value,
    Achievement.COLLECT_RUBY.value,
    Achievement.COLLECT_SAPPHIRE.value,
    Achievement.COLLECT_SAPLING.value,
    Achievement.FIND_BOW.value,
    Achievement.MAKE_ARROW.value,
    Achievement.MAKE_TORCH.value,
], dtype=jnp.int32)

INVENTORY_ACHIEVEMENT_PICKAXE_INDICES = jnp.array([
    Achievement.MAKE_WOOD_PICKAXE.value,
    Achievement.MAKE_STONE_PICKAXE.value,
    Achievement.MAKE_IRON_PICKAXE.value,
    Achievement.MAKE_DIAMOND_PICKAXE.value,
], dtype=jnp.int32)

INVENTORY_ACHIEVEMENT_SWORD_INDICES = jnp.array([
    Achievement.MAKE_WOOD_SWORD.value,
    Achievement.MAKE_STONE_SWORD.value,
    Achievement.MAKE_IRON_SWORD.value,
    Achievement.MAKE_DIAMOND_SWORD.value,
], dtype=jnp.int32)

# Potion action lookup: maps action -> potion index (-1 for non-potion actions)
# Potion actions are consecutive: RED=29, GREEN=30, BLUE=31, PINK=32, CYAN=33, YELLOW=34
POTION_ACTION_START = Action.DRINK_POTION_RED.value  # 29
POTION_ACTION_END = Action.DRINK_POTION_YELLOW.value  # 34
NUM_POTIONS = 6

# Achievement indices for batched updates in enchant
ENCHANT_ACHIEVEMENT_INDICES = jnp.array([
    Achievement.ENCHANT_SWORD.value,
    Achievement.ENCHANT_ARMOUR.value,
], dtype=jnp.int32)

# ========== SPAWN BLOCK TYPE LOOKUP TABLES ==========
# These replace multiple OR comparisons in spawn_mobs with single lookups
def _build_spawn_block_tables():
    num_blocks = len(BlockType)

    # Grave blocks (for boss spawning)
    is_grave = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_grave = is_grave.at[BlockType.GRAVE.value].set(True)
    is_grave = is_grave.at[BlockType.GRAVE2.value].set(True)
    is_grave = is_grave.at[BlockType.GRAVE3.value].set(True)

    # Valid mob spawn blocks (grass, path, fire_grass, ice_grass)
    is_valid_spawn = jnp.zeros(num_blocks, dtype=jnp.bool_)
    is_valid_spawn = is_valid_spawn.at[BlockType.GRASS.value].set(True)
    is_valid_spawn = is_valid_spawn.at[BlockType.PATH.value].set(True)
    is_valid_spawn = is_valid_spawn.at[BlockType.FIRE_GRASS.value].set(True)
    is_valid_spawn = is_valid_spawn.at[BlockType.ICE_GRASS.value].set(True)

    return is_grave, is_valid_spawn

BLOCK_IS_GRAVE, BLOCK_IS_VALID_SPAWN = _build_spawn_block_tables()


def is_game_over(state, params, static_env_params):
    done_steps = state.timestep >= params.max_timesteps
    is_dead = state.player_health <= 0
    defeated_boss = has_beaten_boss(state, static_env_params)

    return done_steps | is_dead | defeated_boss


def update_plants_with_eat(state, plant_position, static_params):
    # Vectorized: no scan needed
    is_plant = (state.growing_plants_positions == plant_position).all(axis=-1)
    plant_index = jnp.argmax(is_plant)
    return state.growing_plants_age.at[plant_index].set(0)


def add_items_from_chest(keys, state, inventory, is_opening_chest):
    # keys shape: (16,) - pre-split keys passed from caller
    k_wood_chance, k_wood_amt, k_torch_chance, k_torch_amt = keys[0], keys[1], keys[2], keys[3]
    k_ore_chance, k_ore_id, k_ore_amt = keys[4], keys[5], keys[6]
    k_potion_chance, k_potion_idx, k_potion_amt = keys[7], keys[8], keys[9]
    k_arrow_chance, k_arrow_amt = keys[10], keys[11]
    k_tool_chance, k_tool_id, k_pickaxe, k_sword = keys[12], keys[13], keys[14], keys[15]

    # Wood (60%)
    is_looting_wood = jax.random.uniform(k_wood_chance) < 0.6
    wood_loot_amount = (
        jax.random.randint(k_wood_amt, shape=(), minval=1, maxval=6) * is_looting_wood
    )

    # Torch (60%)
    is_looting_torch = jax.random.uniform(k_torch_chance) < 0.6
    torch_loot_amount = (
        jax.random.randint(k_torch_amt, shape=(), minval=4, maxval=8) * is_looting_torch
    )

    # Ores (60%)
    is_looting_ore = jax.random.uniform(k_ore_chance) < 0.6
    ore_loot_id = jax.random.choice(
        k_ore_id,
        jnp.arange(5, dtype=jnp.int32),
        shape=(),
        p=jnp.array([0.3, 0.3, 0.15, 0.125, 0.125]),
    )

    # Use the same rng as events are mutually exclusive
    coal_loot_amount = (
        jax.random.randint(k_ore_amt, shape=(), minval=1, maxval=4)
        * (ore_loot_id == 0)
        * is_looting_ore
    )
    iron_loot_amount = (
        jax.random.randint(k_ore_amt, shape=(), minval=1, maxval=3)
        * (ore_loot_id == 1)
        * is_looting_ore
    )
    diamond_loot_amount = (
        jax.random.randint(k_ore_amt, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 2)
        * is_looting_ore
    )
    sapphire_loot_amount = (
        jax.random.randint(k_ore_amt, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 3)
        * is_looting_ore
    )
    ruby_loot_amount = (
        jax.random.randint(k_ore_amt, shape=(), minval=1, maxval=2)
        * (ore_loot_id == 4)
        * is_looting_ore
    )

    # Potion (50%)
    is_looting_potion = jax.random.uniform(k_potion_chance) < 0.5
    potion_loot_index = jax.random.randint(k_potion_idx, shape=(), minval=0, maxval=6)
    potion_loot_amount = jax.random.randint(k_potion_amt, shape=(), minval=1, maxval=3)

    # Arrows (25%)
    is_looting_arrows = jax.random.uniform(k_arrow_chance) < 0.25
    arrows_loot_amount = (
        jax.random.randint(k_arrow_amt, shape=(), minval=1, maxval=5) * is_looting_arrows
    )

    # Tools (20%)
    is_looting_tool = jax.random.uniform(k_tool_chance) < 0.2
    tool_id = jax.random.randint(k_tool_id, shape=(), minval=0, maxval=2)

    is_looting_pickaxe = jnp.logical_and(
        jnp.logical_and(is_looting_tool, tool_id == 0), is_opening_chest
    )
    pickaxe_loot_level = (
        jax.random.choice(
            k_pickaxe,
            (jnp.arange(4) + 1).astype(int),
            shape=(),
            p=jnp.array([0.4, 0.3, 0.2, 0.1]),
        )
        * is_looting_pickaxe
    )
    pickaxe_loot_level = jnp.maximum(pickaxe_loot_level, inventory.pickaxe)
    new_pickaxe_level = (
        is_looting_pickaxe * pickaxe_loot_level
        + (1 - is_looting_pickaxe) * inventory.pickaxe
    )

    is_looting_sword = jnp.logical_and(
        jnp.logical_and(is_looting_tool, tool_id == 1), is_opening_chest
    )
    sword_loot_level = (
        jax.random.choice(
            k_sword,
            (jnp.arange(4) + 1).astype(int),
            shape=(),
            p=jnp.array([0.4, 0.3, 0.2, 0.1]),
        )
        * is_looting_sword
    )
    sword_loot_level = jnp.maximum(sword_loot_level, inventory.sword)
    new_sword_level = (
        is_looting_sword * sword_loot_level + (1 - is_looting_sword) * inventory.sword
    )

    # Special chests
    is_looting_bow = jnp.logical_and(
        is_opening_chest,
        jnp.logical_and(
            state.player_level == 1,
            jnp.logical_not(state.chests_opened[state.player_level]),
        ),
    )
    new_bow_level = is_looting_bow * 1 + (1 - is_looting_bow) * inventory.bow

    is_looting_book = jnp.logical_and(
        jnp.logical_not(state.chests_opened[state.player_level]),
        jnp.logical_or(state.player_level == 3, state.player_level == 4),
    )

    # Update inventory
    return inventory.replace(
        torches=inventory.torches + torch_loot_amount * is_opening_chest,
        coal=inventory.coal + coal_loot_amount * is_opening_chest,
        iron=inventory.iron + iron_loot_amount * is_opening_chest,
        diamond=inventory.diamond + diamond_loot_amount * is_opening_chest,
        sapphire=inventory.sapphire + sapphire_loot_amount * is_opening_chest,
        ruby=inventory.ruby + ruby_loot_amount * is_opening_chest,
        arrows=inventory.arrows + arrows_loot_amount * is_opening_chest,
        pickaxe=new_pickaxe_level,
        sword=new_sword_level,
        potions=inventory.potions.at[potion_loot_index].set(
            inventory.potions[potion_loot_index]
            + potion_loot_amount * is_looting_potion * is_opening_chest
        ),
        bow=new_bow_level,
        books=inventory.books + 1 * is_looting_book * is_opening_chest,
    )


def do_action(keys, state, action, static_params):
    # keys shape: (17,) - pre-split keys passed from caller
    # keys[0] = sapling_rng
    # keys[1:17] = chest_keys (16 keys for add_items_from_chest)
    old_state = state

    sapling_rng = keys[0]
    chest_keys = keys[1:17]

    block_position = state.player_position + DIRECTIONS[state.player_direction]
    r, c = block_position[0], block_position[1]
    lvl = state.player_level

    state, did_attack_mob, did_kill_mob = attack_mob(
        state, block_position, get_player_damage_vector(state), True
    )

    # Read tile ONCE
    tile = state.map[lvl, r, c]

    # ========== LOOKUP TABLE BASED RESOURCE/BLOCK CHECKS ==========
    # Single lookups replace 15+ individual comparisons
    can_mine = state.inventory.pickaxe >= BLOCK_REQUIRED_PICKAXE[tile]
    tile_out = jax.lax.select(can_mine, BLOCK_REPLACEMENT[tile], tile)

    # Resource gains via lookup (replaces is_mining_X = is_block_X & can_mine pattern)
    wood_gain = BLOCK_GIVES_WOOD[tile] * can_mine
    stone_gain = BLOCK_GIVES_STONE[tile] * can_mine
    coal_gain = BLOCK_GIVES_COAL[tile] * can_mine
    iron_gain = BLOCK_GIVES_IRON[tile] * can_mine
    diamond_gain = BLOCK_GIVES_DIAMOND[tile] * can_mine
    sapphire_gain = BLOCK_GIVES_SAPPHIRE[tile] * can_mine
    ruby_gain = BLOCK_GIVES_RUBY[tile] * can_mine

    # Special block checks via lookup
    is_drinking_water = BLOCK_IS_WATER_SOURCE[tile]
    is_eating_plant = jnp.logical_and(BLOCK_IS_RIPE_PLANT[tile], can_mine)
    is_opening_chest = jnp.logical_and(BLOCK_IS_CHEST[tile], can_mine)
    is_block_necromancer = BLOCK_IS_NECROMANCER[tile]

    # Sapling (random chance when mining grass)
    is_mining_sapling = jnp.logical_and(BLOCK_IS_GRASS[tile], jax.random.uniform(sapling_rng) < 0.1)

    # ========== SINGLE MAP UPDATE ==========
    new_map = state.map[lvl].at[r, c].set(tile_out)

    # ========== INVENTORY UPDATE (single replace) ==========
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood + wood_gain,
        stone=state.inventory.stone + stone_gain,
        coal=state.inventory.coal + coal_gain,
        iron=state.inventory.iron + iron_gain,
        diamond=state.inventory.diamond + diamond_gain,
        sapphire=state.inventory.sapphire + sapphire_gain,
        ruby=state.inventory.ruby + ruby_gain,
        sapling=state.inventory.sapling + is_mining_sapling,
    )

    # Chest items
    new_inventory = add_items_from_chest(chest_keys, state, new_inventory, is_opening_chest)

    # ========== WATER/DRINK ==========
    new_drink = jax.lax.select(
        is_drinking_water,
        jnp.minimum(get_max_drink(state), state.player_drink + 1),
        state.player_drink,
    )
    new_thirst = jax.lax.select(is_drinking_water, 0.0, state.player_thirst)

    # ========== PLANT/FOOD ==========
    new_food = jax.lax.select(
        is_eating_plant,
        jnp.minimum(get_max_food(state), state.player_food + 4),
        state.player_food,
    )
    new_hunger = jax.lax.select(is_eating_plant, 0.0, state.player_hunger)
    new_growing_plants_age = update_plants_with_eat(
        state, block_position, static_params
    )

    # ========== BOSS ==========
    is_damaging_boss = jnp.logical_and(
        is_block_necromancer,
        jnp.logical_and(
            is_boss_vulnerable(state), is_fighting_boss(state, static_params)
        ),
    )
    new_boss_progress = state.boss_progress + 1 * is_damaging_boss
    new_boss_timesteps_to_spawn_this_round = (
        BOSS_FIGHT_SPAWN_TURNS * is_damaging_boss
        + state.boss_timesteps_to_spawn_this_round * (1 - is_damaging_boss)
    )

    # ========== BATCHED ACHIEVEMENT UPDATE ==========
    # Single vectorized update replaces 4 separate .at[].set() calls
    achievement_triggers = jnp.array([
        is_drinking_water, is_eating_plant, is_opening_chest, is_damaging_boss
    ])
    current_achievements = state.achievements[DO_ACTION_ACHIEVEMENT_INDICES]
    updated_achievements = jnp.logical_or(current_achievements, achievement_triggers)
    new_achievements = state.achievements.at[DO_ACTION_ACHIEVEMENT_INDICES].set(updated_achievements)

    # ========== CHEST STATE ==========
    new_chests_opened = state.chests_opened.at[lvl].set(
        jnp.logical_or(state.chests_opened[lvl], is_opening_chest)
    )

    # ========== BOUNDS CHECK - APPLY OR REVERT ==========
    action_block_in_bounds = in_bounds(state, block_position)
    action_block_in_bounds = jnp.logical_and(
        action_block_in_bounds, jnp.logical_not(did_attack_mob)
    )
    new_map = jax.lax.select(action_block_in_bounds, new_map, state.map[lvl])
    new_inventory = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(action_block_in_bounds, x, y),
        new_inventory,
        state.inventory,
    )
    new_drink = jax.lax.select(action_block_in_bounds, new_drink, state.player_drink)
    new_thirst = jax.lax.select(action_block_in_bounds, new_thirst, state.player_thirst)
    new_food = jax.lax.select(action_block_in_bounds, new_food, state.player_food)
    new_hunger = jax.lax.select(action_block_in_bounds, new_hunger, state.player_hunger)
    new_growing_plants_age = jax.lax.select(
        jnp.logical_and(action_block_in_bounds, is_eating_plant),
        new_growing_plants_age,
        state.growing_plants_age,
    )
    new_achievements = jax.lax.select(
        action_block_in_bounds, new_achievements, state.achievements
    )

    new_whole_map = state.map.at[lvl].set(new_map)

    state = state.replace(
        map=new_whole_map,
        inventory=new_inventory,
        player_drink=new_drink,
        player_thirst=new_thirst,
        player_food=new_food,
        player_hunger=new_hunger,
        growing_plants_age=new_growing_plants_age,
        achievements=new_achievements,
        chests_opened=new_chests_opened,
        boss_progress=new_boss_progress,
        boss_timesteps_to_spawn_this_round=new_boss_timesteps_to_spawn_this_round,
    )

    # Do?
    doing_mining = action == Action.DO.value
    state = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(doing_mining, x, y),
        state,
        old_state,
    )

    return state


def do_crafting(state, action):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value)
    inv = state.inventory

    # Since only ONE action can happen per step, we can compute all conditions
    # using the original inventory (no need to chain through new_inventory)

    # Wood pickaxe: 1 wood -> pickaxe level 1
    is_crafting_wood_pickaxe = jnp.logical_and(
        action == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(
            inv.wood >= 1,
            jnp.logical_and(is_at_crafting_table, inv.pickaxe < 1),
        ),
    )

    # Stone pickaxe: 1 wood + 1 stone -> pickaxe level 2
    is_crafting_stone_pickaxe = jnp.logical_and(
        action == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(
            jnp.logical_and(inv.wood >= 1, inv.stone >= 1),
            jnp.logical_and(is_at_crafting_table, inv.pickaxe < 2),
        ),
    )

    # Iron pickaxe: 1 wood + 1 stone + 1 iron + 1 coal -> pickaxe level 3
    is_crafting_iron_pickaxe = jnp.logical_and(
        action == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(inv.wood >= 1, inv.stone >= 1),
                jnp.logical_and(inv.iron >= 1, inv.coal >= 1),
            ),
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, inv.pickaxe < 3),
            ),
        ),
    )

    # Diamond pickaxe: 1 wood + 3 diamond -> pickaxe level 4
    is_crafting_diamond_pickaxe = jnp.logical_and(
        action == Action.MAKE_DIAMOND_PICKAXE.value,
        jnp.logical_and(
            jnp.logical_and(inv.wood >= 1, inv.diamond >= 3),
            jnp.logical_and(is_at_crafting_table, inv.pickaxe < 4),
        ),
    )

    # Wood sword: 1 wood -> sword level 1
    is_crafting_wood_sword = jnp.logical_and(
        action == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(
            inv.wood >= 1,
            jnp.logical_and(is_at_crafting_table, inv.sword < 1),
        ),
    )

    # Stone sword: 1 wood + 1 stone -> sword level 2
    is_crafting_stone_sword = jnp.logical_and(
        action == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(
            jnp.logical_and(inv.stone >= 1, inv.wood >= 1),
            jnp.logical_and(is_at_crafting_table, inv.sword < 2),
        ),
    )

    # Iron sword: 1 wood + 1 stone + 1 iron + 1 coal -> sword level 3
    is_crafting_iron_sword = jnp.logical_and(
        action == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(inv.iron >= 1, inv.wood >= 1),
                jnp.logical_and(inv.stone >= 1, inv.coal >= 1),
            ),
            jnp.logical_and(
                is_at_furnace,
                jnp.logical_and(is_at_crafting_table, inv.sword < 3),
            ),
        ),
    )

    # Diamond sword: 1 wood + 2 diamond -> sword level 4
    is_crafting_diamond_sword = jnp.logical_and(
        action == Action.MAKE_DIAMOND_SWORD.value,
        jnp.logical_and(
            jnp.logical_and(inv.diamond >= 2, inv.wood >= 1),
            jnp.logical_and(is_at_crafting_table, inv.sword < 4),
        ),
    )

    # Iron armour: 3 iron + 3 coal -> armour slot = 1
    iron_armour_index_to_craft = jnp.argmax(inv.armour < 1)
    can_craft_iron_armour = jnp.logical_and(
        (inv.armour < 1).sum() > 0,
        jnp.logical_and(inv.iron >= 3, inv.coal >= 3),
    )
    is_crafting_iron_armour = jnp.logical_and(
        action == Action.MAKE_IRON_ARMOUR.value,
        jnp.logical_and(
            can_craft_iron_armour,
            jnp.logical_and(is_at_crafting_table, is_at_furnace),
        ),
    )

    # Diamond armour: 3 diamond -> armour slot = 2
    diamond_armour_index_to_craft = jnp.argmax(inv.armour < 2)
    can_craft_diamond_armour = jnp.logical_and(
        (inv.armour < 2).sum() > 0,
        inv.diamond >= 3,
    )
    is_crafting_diamond_armour = jnp.logical_and(
        action == Action.MAKE_DIAMOND_ARMOUR.value,
        jnp.logical_and(can_craft_diamond_armour, is_at_crafting_table),
    )

    # Arrow: 1 wood + 1 stone -> 2 arrows
    is_crafting_arrow = jnp.logical_and(
        action == Action.MAKE_ARROW.value,
        jnp.logical_and(
            jnp.logical_and(inv.stone >= 1, inv.wood >= 1),
            jnp.logical_and(is_at_crafting_table, inv.arrows < 99),
        ),
    )

    # Torch: 1 wood + 1 coal -> 4 torches
    is_crafting_torch = jnp.logical_and(
        action == Action.MAKE_TORCH.value,
        jnp.logical_and(
            jnp.logical_and(inv.coal >= 1, inv.wood >= 1),
            jnp.logical_and(is_at_crafting_table, inv.torches < 99),
        ),
    )

    # Compute total resource consumption
    wood_cost = (
        1 * is_crafting_wood_pickaxe
        + 1 * is_crafting_stone_pickaxe
        + 1 * is_crafting_iron_pickaxe
        + 1 * is_crafting_diamond_pickaxe
        + 1 * is_crafting_wood_sword
        + 1 * is_crafting_stone_sword
        + 1 * is_crafting_iron_sword
        + 1 * is_crafting_diamond_sword
        + 1 * is_crafting_arrow
        + 1 * is_crafting_torch
    )

    stone_cost = (
        1 * is_crafting_stone_pickaxe
        + 1 * is_crafting_iron_pickaxe
        + 1 * is_crafting_stone_sword
        + 1 * is_crafting_iron_sword
        + 1 * is_crafting_arrow
    )

    iron_cost = (
        1 * is_crafting_iron_pickaxe
        + 1 * is_crafting_iron_sword
        + 3 * is_crafting_iron_armour
    )

    coal_cost = (
        1 * is_crafting_iron_pickaxe
        + 1 * is_crafting_iron_sword
        + 3 * is_crafting_iron_armour
        + 1 * is_crafting_torch
    )

    diamond_cost = (
        3 * is_crafting_diamond_pickaxe
        + 2 * is_crafting_diamond_sword
        + 3 * is_crafting_diamond_armour
    )

    # Compute new tool levels (only one crafting can happen, so we can sum)
    new_pickaxe = (
        inv.pickaxe * (1 - is_crafting_wood_pickaxe - is_crafting_stone_pickaxe
                        - is_crafting_iron_pickaxe - is_crafting_diamond_pickaxe)
        + 1 * is_crafting_wood_pickaxe
        + 2 * is_crafting_stone_pickaxe
        + 3 * is_crafting_iron_pickaxe
        + 4 * is_crafting_diamond_pickaxe
    )

    new_sword = (
        inv.sword * (1 - is_crafting_wood_sword - is_crafting_stone_sword
                      - is_crafting_iron_sword - is_crafting_diamond_sword)
        + 1 * is_crafting_wood_sword
        + 2 * is_crafting_stone_sword
        + 3 * is_crafting_iron_sword
        + 4 * is_crafting_diamond_sword
    )

    # Compute new armour (handle array updates)
    new_armour = inv.armour
    new_armour = jax.lax.select(
        is_crafting_iron_armour,
        new_armour.at[iron_armour_index_to_craft].set(1),
        new_armour,
    )
    new_armour = jax.lax.select(
        is_crafting_diamond_armour,
        new_armour.at[diamond_armour_index_to_craft].set(2),
        new_armour,
    )

    # Single consolidated inventory update
    new_inventory = inv.replace(
        wood=inv.wood - wood_cost,
        stone=inv.stone - stone_cost,
        iron=inv.iron - iron_cost,
        coal=inv.coal - coal_cost,
        diamond=inv.diamond - diamond_cost,
        pickaxe=new_pickaxe,
        sword=new_sword,
        armour=new_armour,
        arrows=inv.arrows + 2 * is_crafting_arrow,
        torches=inv.torches + 4 * is_crafting_torch,
    )

    # Batched achievements update (2 ops -> 1)
    crafting_triggers = jnp.array([is_crafting_iron_armour, is_crafting_diamond_armour])
    current_crafting_achievements = state.achievements[DO_CRAFTING_ACHIEVEMENT_INDICES]
    updated_crafting_achievements = jnp.logical_or(current_crafting_achievements, crafting_triggers)
    new_achievements = state.achievements.at[DO_CRAFTING_ACHIEVEMENT_INDICES].set(updated_crafting_achievements)

    state = state.replace(
        inventory=new_inventory,
        achievements=new_achievements,
    )

    return state


def add_new_growing_plant(state, position, is_placing_sapling, static_params):
    # Vectorized: no scan needed
    is_empty = jnp.logical_not(state.growing_plants_mask)
    plant_index = jnp.argmax(is_empty)
    is_an_empty_slot = is_empty.sum() > 0

    is_adding_plant = jnp.logical_and(is_an_empty_slot, is_placing_sapling)

    new_growing_plants_positions = jax.lax.select(
        is_adding_plant,
        state.growing_plants_positions.at[plant_index].set(position),
        state.growing_plants_positions,
    )
    new_growing_plants_age = jax.lax.select(
        is_adding_plant,
        state.growing_plants_age.at[plant_index].set(0),
        state.growing_plants_age,
    )
    new_growing_plants_mask = jax.lax.select(
        is_adding_plant,
        state.growing_plants_mask.at[plant_index].set(True),
        state.growing_plants_mask,
    )

    return new_growing_plants_positions, new_growing_plants_age, new_growing_plants_mask


def place_block(state, action, static_params):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]
    r, c = placing_block_position[0], placing_block_position[1]
    lvl = state.player_level
    inv = state.inventory

    current_block = state.map[lvl, r, c]
    current_item = state.item_map[lvl, r, c]

    is_solid = is_in_solid_block(state, placing_block_position)
    has_item = current_item != ItemType.NONE.value
    is_placement_on_solid_block_or_item = jnp.logical_or(is_solid, has_item)
    not_blocked = jnp.logical_not(is_placement_on_solid_block_or_item)

    # Since only ONE action can happen per step, compute all conditions with original inventory

    # Crafting table: 2 wood
    is_placing_crafting_table = jnp.logical_and(
        action == Action.PLACE_TABLE.value,
        jnp.logical_and(not_blocked, inv.wood >= 2),
    )

    # Furnace: 1 stone
    is_placing_furnace = jnp.logical_and(
        action == Action.PLACE_FURNACE.value,
        jnp.logical_and(not_blocked, inv.stone >= 1),
    )

    # Stone: 1 stone (can also place on water)
    can_place_stone = jnp.logical_or(current_block == BlockType.WATER.value, not_blocked)
    is_placing_stone = jnp.logical_and(
        action == Action.PLACE_STONE.value,
        jnp.logical_and(can_place_stone, inv.stone >= 1),
    )

    # Torch: 1 torch (item placement, different rules)
    can_place_torch = jnp.logical_and(
        CAN_PLACE_ITEM_MAPPING[current_block],
        current_item == ItemType.NONE.value,
    )
    is_placing_torch = jnp.logical_and(
        action == Action.PLACE_TORCH.value,
        jnp.logical_and(can_place_torch, inv.torches >= 1),
    )

    # Plant/sapling: 1 sapling (must be on grass, no item)
    can_place_sapling = jnp.logical_and(
        current_block == BlockType.GRASS.value,
        current_item == ItemType.NONE.value,
    )
    is_placing_sapling = jnp.logical_and(
        action == Action.PLACE_PLANT.value,
        jnp.logical_and(can_place_sapling, inv.sapling >= 1),
    )

    # Compute final block via priority cascade (last wins)
    block_out = current_block
    block_out = jax.lax.select(is_placing_crafting_table, BlockType.CRAFTING_TABLE.value, block_out)
    block_out = jax.lax.select(is_placing_furnace, BlockType.FURNACE.value, block_out)
    block_out = jax.lax.select(is_placing_stone, BlockType.STONE.value, block_out)
    block_out = jax.lax.select(is_placing_sapling, BlockType.PLANT.value, block_out)

    # Single map update
    new_map = state.map[lvl].at[r, c].set(block_out)

    # Item map update (torch only)
    item_out = jax.lax.select(is_placing_torch, ItemType.TORCH.value, current_item)
    new_item_map = state.item_map[lvl].at[r, c].set(item_out)

    # Single consolidated inventory update
    new_inventory = inv.replace(
        wood=inv.wood - 2 * is_placing_crafting_table,
        stone=inv.stone - 1 * is_placing_furnace - 1 * is_placing_stone,
        torches=inv.torches - 1 * is_placing_torch,
        sapling=inv.sapling - 1 * is_placing_sapling,
    )

    # Batched achievements update (5 ops -> 1)
    place_triggers = jnp.array([
        is_placing_crafting_table, is_placing_furnace, is_placing_stone,
        is_placing_torch, is_placing_sapling
    ])
    current_place_achievements = state.achievements[PLACE_BLOCK_ACHIEVEMENT_INDICES]
    updated_place_achievements = jnp.logical_or(current_place_achievements, place_triggers)
    new_achievements = state.achievements.at[PLACE_BLOCK_ACHIEVEMENT_INDICES].set(updated_place_achievements)

    # Light map update for torch placement
    light_map_padding = 6
    padded_light_map = jnp.pad(
        state.light_map[lvl],
        (light_map_padding, light_map_padding),
        constant_values=0,
    )

    current_light_map = jax.lax.dynamic_slice(
        padded_light_map,
        placing_block_position
        - jnp.array([4, 4])
        + jnp.array([light_map_padding, light_map_padding]),
        (9, 9),
    )

    torch_light_map = jnp.clip(TORCH_LIGHT_MAP + current_light_map, 0.0, 1.0)
    torch_light_map = torch_light_map * is_placing_torch + current_light_map * (1 - is_placing_torch)

    new_padded_light_map_floor = jax.lax.dynamic_update_slice(
        padded_light_map,
        torch_light_map,
        placing_block_position
        - jnp.array([4, 4])
        + jnp.array([light_map_padding, light_map_padding]),
    )
    new_light_map_floor = new_padded_light_map_floor[
        light_map_padding:-light_map_padding, light_map_padding:-light_map_padding
    ]
    new_light_map = state.light_map.at[lvl].set(new_light_map_floor)

    # Growing plants for sapling placement
    (
        new_growing_plants_positions,
        new_growing_plants_age,
        new_growing_plants_mask,
    ) = add_new_growing_plant(
        state, placing_block_position, is_placing_sapling, static_params
    )

    # Do?

    action_block = state.player_position + DIRECTIONS[state.player_direction]
    action_block_in_bounds = in_bounds(state, action_block)
    action_block_in_bounds = jnp.logical_and(
        action_block_in_bounds, jnp.logical_not(is_in_mob(state, action_block))
    )

    new_map = jax.lax.select(
        action_block_in_bounds, new_map, state.map[state.player_level]
    )
    new_item_map = jax.lax.select(
        action_block_in_bounds, new_item_map, state.item_map[state.player_level]
    )
    new_inventory = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(action_block_in_bounds, x, y),
        new_inventory,
        state.inventory,
    )
    new_achievements = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(action_block_in_bounds, x, y),
        new_achievements,
        state.achievements,
    )
    new_growing_plants_positions = jax.lax.select(
        action_block_in_bounds,
        new_growing_plants_positions,
        state.growing_plants_positions,
    )
    new_growing_plants_age = jax.lax.select(
        action_block_in_bounds, new_growing_plants_age, state.growing_plants_age
    )
    new_growing_plants_mask = jax.lax.select(
        action_block_in_bounds, new_growing_plants_mask, state.growing_plants_mask
    )
    new_light_map = jax.lax.select(
        action_block_in_bounds, new_light_map, state.light_map
    )

    new_whole_map = state.map.at[state.player_level].set(new_map)
    new_whole_item_map = state.item_map.at[state.player_level].set(new_item_map)

    state = state.replace(
        map=new_whole_map,
        item_map=new_whole_item_map,
        light_map=new_light_map,
        inventory=new_inventory,
        achievements=new_achievements,
        growing_plants_positions=new_growing_plants_positions,
        growing_plants_age=new_growing_plants_age,
        growing_plants_mask=new_growing_plants_mask,
    )

    return state


def update_mobs(keys, state, params, static_params):
    # keys shape: (4*max_melee + 1*max_passive + 3*max_ranged, 2)
    # Slice offsets
    melee_keys_per = 4
    passive_keys_per = 1
    ranged_keys_per = 3

    melee_offset = 0
    passive_offset = melee_keys_per * static_params.max_melee_mobs
    ranged_offset = passive_offset + passive_keys_per * static_params.max_passive_mobs

    # Cache is_fighting_boss and player_level (used many times in inner scans)
    fighting_boss = is_fighting_boss(state, static_params)
    lvl = state.player_level

    # Move melee_mobs - use carry tuple to avoid state.replace inside scan
    # Carry: (player_health, is_sleeping, is_resting, achievements, melee_positions, melee_cooldowns, melee_masks, mob_map)
    melee_initial_carry = (
        state.player_health,
        state.is_sleeping,
        state.is_resting,
        state.achievements,
        state.melee_mobs.position,
        state.melee_mobs.attack_cooldown,
        state.melee_mobs.mask,
        state.mob_map,
    )

    def _move_melee_mob(carry, melee_mob_index):
        (player_health, is_sleeping, is_resting, achievements,
         melee_positions, melee_cooldowns, melee_masks, mob_map) = carry

        # Get pre-split keys for this iteration
        k_base = melee_offset + melee_mob_index * melee_keys_per
        k_random_move = keys[k_base]
        k_player_dir = keys[k_base + 1]
        k_close_check = keys[k_base + 2]

        current_pos = melee_positions[lvl, melee_mob_index]
        current_mask = melee_masks[lvl, melee_mob_index]
        current_cooldown = melee_cooldowns[lvl, melee_mob_index]

        # Random move
        random_move_direction = jax.random.choice(k_random_move, DIRECTIONS[1:5])
        random_move_proposed_position = current_pos + random_move_direction

        # Move towards player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(state.player_position - current_pos)
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        player_move_direction_index = jax.random.choice(
            k_player_dir, jnp.arange(2), p=player_move_direction_index_p
        )
        player_move_direction = player_move_direction.at[player_move_direction_index].set(
            jnp.sign(
                state.player_position[player_move_direction_index]
                - current_pos[player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = current_pos + player_move_direction

        # Choose movement
        close_to_player = jnp.sum(jnp.abs(current_pos - state.player_position)) < 10
        close_to_player = jnp.logical_or(close_to_player, fighting_boss)
        close_to_player = jnp.logical_and(
            close_to_player, jax.random.uniform(k_close_check) < 0.75
        )

        proposed_position = jax.lax.select(
            close_to_player, player_move_proposed_position, random_move_proposed_position
        )

        # Choose attack or not
        is_attacking_player = jnp.sum(jnp.abs(current_pos - state.player_position)) == 1
        is_attacking_player = jnp.logical_and(is_attacking_player, current_cooldown <= 0)
        is_attacking_player = jnp.logical_and(is_attacking_player, current_mask)

        proposed_position = jax.lax.select(is_attacking_player, current_pos, proposed_position)

        melee_mob_base_damage = MOB_TYPE_DAMAGE_MAPPING[
            state.melee_mobs.type_id[lvl, melee_mob_index], MobType.MELEE.value
        ]
        melee_mob_damage = get_damage_done_to_player(
            state, static_params, melee_mob_base_damage * (1 + 2.5 * is_sleeping)
        )

        new_cooldown = jax.lax.select(is_attacking_player, 5, current_cooldown - 1)
        is_waking_player = jnp.logical_and(is_sleeping, is_attacking_player)

        # Update player state
        new_player_health = player_health - melee_mob_damage * is_attacking_player
        new_is_sleeping = jnp.logical_and(is_sleeping, jnp.logical_not(is_attacking_player))
        new_is_resting = jnp.logical_and(is_resting, jnp.logical_not(is_attacking_player))
        new_achievements = achievements.at[Achievement.WAKE_UP.value].set(
            jnp.logical_or(achievements[Achievement.WAKE_UP.value], is_waking_player)
        )

        # Check valid move - inline is_in_mob using carry's mob_map
        mob_type = state.melee_mobs.type_id[lvl, melee_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 1]
        # Use mob_map from carry instead of state.mob_map for correct sequential behavior
        in_mob_carry = jnp.logical_or(
            mob_map[lvl, proposed_position[0], proposed_position[1]],
            (state.player_position == proposed_position).all(),
        )
        valid_move = is_position_in_bounds_not_in_mob_not_colliding_with_mob_map(
            state, proposed_position, collision_map, in_mob_carry
        )
        new_position = jax.lax.select(valid_move, proposed_position, current_pos)

        # Despawn check
        should_not_despawn = jnp.abs(current_pos - state.player_position).sum() < params.mob_despawn_distance
        should_not_despawn = jnp.logical_or(should_not_despawn, fighting_boss)
        new_mask = jnp.logical_and(current_mask, should_not_despawn)

        # Update mob_map: clear old, set new
        new_mob_map = mob_map.at[lvl, current_pos[0], current_pos[1]].set(
            jnp.logical_and(mob_map[lvl, current_pos[0], current_pos[1]], jnp.logical_not(current_mask))
        )
        new_mob_map = new_mob_map.at[lvl, new_position[0], new_position[1]].set(
            jnp.logical_or(new_mob_map[lvl, new_position[0], new_position[1]], new_mask)
        )

        # Update mob arrays
        new_melee_positions = melee_positions.at[lvl, melee_mob_index].set(new_position)
        new_melee_cooldowns = melee_cooldowns.at[lvl, melee_mob_index].set(new_cooldown)
        new_melee_masks = melee_masks.at[lvl, melee_mob_index].set(new_mask)

        new_carry = (
            new_player_health, new_is_sleeping, new_is_resting, new_achievements,
            new_melee_positions, new_melee_cooldowns, new_melee_masks, new_mob_map
        )
        return new_carry, None

    melee_final_carry, _ = jax.lax.scan(
        _move_melee_mob, melee_initial_carry, jnp.arange(static_params.max_melee_mobs)
    )

    # Apply melee mob updates with ONE state.replace
    (player_health, is_sleeping, is_resting, achievements,
     melee_positions, melee_cooldowns, melee_masks, mob_map) = melee_final_carry

    state = state.replace(
        player_health=player_health,
        is_sleeping=is_sleeping,
        is_resting=is_resting,
        achievements=achievements,
        melee_mobs=state.melee_mobs.replace(
            position=melee_positions,
            attack_cooldown=melee_cooldowns,
            mask=melee_masks,
        ),
        mob_map=mob_map,
    )

    # Move passive_mobs - use carry tuple to avoid state.replace inside scan
    # Carry: (passive_positions, passive_masks, mob_map)
    passive_initial_carry = (
        state.passive_mobs.position,
        state.passive_mobs.mask,
        state.mob_map,
    )

    def _move_passive_mob(carry, passive_mob_index):
        passive_positions, passive_masks, mob_map = carry

        # Get pre-split key for this iteration
        k_base = passive_offset + passive_mob_index * passive_keys_per
        k_random_move = keys[k_base]

        current_pos = passive_positions[lvl, passive_mob_index]
        current_mask = passive_masks[lvl, passive_mob_index]

        # Random move
        random_move_direction = jax.random.choice(k_random_move, DIRECTIONS[1:9])
        proposed_position = current_pos + random_move_direction

        mob_type = state.passive_mobs.type_id[lvl, passive_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 0]
        # Use mob_map from carry instead of state.mob_map for correct sequential behavior
        in_mob_carry = jnp.logical_or(
            mob_map[lvl, proposed_position[0], proposed_position[1]],
            (state.player_position == proposed_position).all(),
        )
        valid_move = is_position_in_bounds_not_in_mob_not_colliding_with_mob_map(
            state, proposed_position, collision_map, in_mob_carry
        )
        new_position = jax.lax.select(valid_move, proposed_position, current_pos)

        should_not_despawn = (
            jnp.abs(current_pos - state.player_position).sum() < params.mob_despawn_distance
        )
        new_mask = jnp.logical_and(current_mask, should_not_despawn)

        # Update mob_map: clear old, set new
        new_mob_map = mob_map.at[lvl, current_pos[0], current_pos[1]].set(
            jnp.logical_and(mob_map[lvl, current_pos[0], current_pos[1]], jnp.logical_not(current_mask))
        )
        new_mob_map = new_mob_map.at[lvl, new_position[0], new_position[1]].set(
            jnp.logical_or(new_mob_map[lvl, new_position[0], new_position[1]], new_mask)
        )

        # Update mob arrays
        new_passive_positions = passive_positions.at[lvl, passive_mob_index].set(new_position)
        new_passive_masks = passive_masks.at[lvl, passive_mob_index].set(new_mask)

        new_carry = (new_passive_positions, new_passive_masks, new_mob_map)
        return new_carry, None

    passive_final_carry, _ = jax.lax.scan(
        _move_passive_mob, passive_initial_carry, jnp.arange(static_params.max_passive_mobs)
    )

    # Apply passive mob updates with ONE state.replace
    passive_positions, passive_masks, mob_map = passive_final_carry
    state = state.replace(
        passive_mobs=state.passive_mobs.replace(
            position=passive_positions,
            mask=passive_masks,
        ),
        mob_map=mob_map,
    )

    # Move ranged_mobs - use carry tuple to avoid state.replace inside scan
    # Carry: (ranged_positions, ranged_cooldowns, ranged_masks, mob_map,
    #         proj_positions, proj_masks, proj_type_ids, proj_directions)
    ranged_initial_carry = (
        state.ranged_mobs.position,
        state.ranged_mobs.attack_cooldown,
        state.ranged_mobs.mask,
        state.mob_map,
        state.mob_projectiles.position,
        state.mob_projectiles.mask,
        state.mob_projectiles.type_id,
        state.mob_projectile_directions,
    )

    def _move_ranged_mob(carry, ranged_mob_index):
        (ranged_positions, ranged_cooldowns, ranged_masks, mob_map,
         proj_positions, proj_masks, proj_type_ids, proj_directions) = carry

        # Get pre-split keys for this iteration
        k_base = ranged_offset + ranged_mob_index * ranged_keys_per
        k_random_move = keys[k_base]
        k_player_dir = keys[k_base + 1]
        k_chance = keys[k_base + 2]

        current_pos = ranged_positions[lvl, ranged_mob_index]
        current_mask = ranged_masks[lvl, ranged_mob_index]
        current_cooldown = ranged_cooldowns[lvl, ranged_mob_index]

        # Random move
        random_move_direction = jax.random.choice(k_random_move, DIRECTIONS[1:5])
        random_move_proposed_position = current_pos + random_move_direction

        # Move towards player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(state.player_position - current_pos)
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        player_move_direction_index = jax.random.choice(
            k_player_dir, jnp.arange(2), p=player_move_direction_index_p
        )
        player_move_direction = player_move_direction.at[player_move_direction_index].set(
            jnp.sign(
                state.player_position[player_move_direction_index]
                - current_pos[player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_towards_proposed_position = current_pos + player_move_direction
        player_move_away_proposed_position = current_pos - player_move_direction

        # Choose movement
        distance_to_player = jnp.sum(jnp.abs(current_pos - state.player_position))

        far_from_player = distance_to_player >= 6
        too_close_to_player = distance_to_player <= 3

        proposed_position = jax.lax.select(
            far_from_player,
            player_move_towards_proposed_position,
            random_move_proposed_position,
        )
        proposed_position = jax.lax.select(
            too_close_to_player,
            player_move_away_proposed_position,
            proposed_position,
        )
        proposed_position = jax.lax.select(
            jax.random.uniform(k_chance) > 0.85,
            proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = jnp.logical_and(
            distance_to_player >= 4, distance_to_player <= 5
        )
        # If we're too close to player (so we want to run) but are blocked, we shoot
        mob_type = state.ranged_mobs.type_id[lvl, ranged_mob_index]
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type, 2]
        # Use mob_map from carry instead of state.mob_map for correct sequential behavior
        in_mob_carry = jnp.logical_or(
            mob_map[lvl, proposed_position[0], proposed_position[1]],
            (state.player_position == proposed_position).all(),
        )
        is_attacking_player = jnp.logical_or(
            is_attacking_player,
            jnp.logical_and(
                too_close_to_player,
                jnp.logical_not(
                    is_position_in_bounds_not_in_mob_not_colliding_with_mob_map(
                        state, proposed_position, collision_map, in_mob_carry
                    )
                ),
            ),
        )
        is_attacking_player = jnp.logical_and(is_attacking_player, current_cooldown <= 0)
        is_attacking_player = jnp.logical_and(is_attacking_player, current_mask)

        # Spawn projectile (inline instead of calling spawn_projectile to avoid pytree creation)
        can_spawn_projectile = proj_masks[lvl].sum() < static_params.max_mob_projectiles
        is_spawning_projectile = jnp.logical_and(is_attacking_player, can_spawn_projectile)

        new_projectile_index = jnp.argmax(jnp.logical_not(proj_masks[lvl]))
        new_proj_position = jax.lax.select(
            is_spawning_projectile,
            current_pos,
            proj_positions[lvl, new_projectile_index],
        )
        new_proj_mask = jax.lax.select(
            is_spawning_projectile,
            True,
            proj_masks[lvl, new_projectile_index],
        )
        new_proj_direction = jax.lax.select(
            is_spawning_projectile,
            player_move_direction,
            proj_directions[lvl, new_projectile_index],
        )
        projectile_type = RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING[
            state.ranged_mobs.type_id[lvl, ranged_mob_index]
        ]
        new_proj_type = jax.lax.select(
            is_spawning_projectile,
            projectile_type,
            proj_type_ids[lvl, new_projectile_index],
        )

        # Update projectile arrays
        new_proj_positions = proj_positions.at[lvl, new_projectile_index].set(new_proj_position)
        new_proj_masks = proj_masks.at[lvl, new_projectile_index].set(new_proj_mask)
        new_proj_type_ids = proj_type_ids.at[lvl, new_projectile_index].set(new_proj_type)
        new_proj_directions = proj_directions.at[lvl, new_projectile_index].set(new_proj_direction)

        # If attacking, don't move
        proposed_position = jax.lax.select(is_attacking_player, current_pos, proposed_position)
        new_cooldown = jax.lax.select(is_attacking_player, 4, current_cooldown - 1)

        # Valid move check - recompute in_mob_carry since proposed_position may have changed
        in_mob_carry_move = jnp.logical_or(
            mob_map[lvl, proposed_position[0], proposed_position[1]],
            (state.player_position == proposed_position).all(),
        )
        valid_move = is_position_in_bounds_not_in_mob_not_colliding_with_mob_map(
            state, proposed_position, collision_map, in_mob_carry_move
        )
        new_position = jax.lax.select(valid_move, proposed_position, current_pos)

        # Despawn check
        should_not_despawn = jnp.abs(current_pos - state.player_position).sum() < params.mob_despawn_distance
        should_not_despawn = jnp.logical_or(should_not_despawn, fighting_boss)
        new_mask = jnp.logical_and(current_mask, should_not_despawn)

        # Update mob_map: clear old, set new
        new_mob_map = mob_map.at[lvl, current_pos[0], current_pos[1]].set(
            jnp.logical_and(mob_map[lvl, current_pos[0], current_pos[1]], jnp.logical_not(current_mask))
        )
        new_mob_map = new_mob_map.at[lvl, new_position[0], new_position[1]].set(
            jnp.logical_or(new_mob_map[lvl, new_position[0], new_position[1]], new_mask)
        )

        # Update ranged mob arrays
        new_ranged_positions = ranged_positions.at[lvl, ranged_mob_index].set(new_position)
        new_ranged_cooldowns = ranged_cooldowns.at[lvl, ranged_mob_index].set(new_cooldown)
        new_ranged_masks = ranged_masks.at[lvl, ranged_mob_index].set(new_mask)

        new_carry = (
            new_ranged_positions, new_ranged_cooldowns, new_ranged_masks, new_mob_map,
            new_proj_positions, new_proj_masks, new_proj_type_ids, new_proj_directions
        )
        return new_carry, None

    ranged_final_carry, _ = jax.lax.scan(
        _move_ranged_mob, ranged_initial_carry, jnp.arange(static_params.max_ranged_mobs)
    )

    # Apply ranged mob updates with ONE state.replace
    (ranged_positions, ranged_cooldowns, ranged_masks, mob_map,
     proj_positions, proj_masks, proj_type_ids, proj_directions) = ranged_final_carry

    state = state.replace(
        ranged_mobs=state.ranged_mobs.replace(
            position=ranged_positions,
            attack_cooldown=ranged_cooldowns,
            mask=ranged_masks,
        ),
        mob_map=mob_map,
        mob_projectiles=state.mob_projectiles.replace(
            position=proj_positions,
            mask=proj_masks,
            type_id=proj_type_ids,
        ),
        mob_projectile_directions=proj_directions,
    )

    # Move mob projectiles - use carry tuple to avoid state.replace inside scan
    # Carry: (proj_positions, proj_masks, player_health, is_sleeping, is_resting, map_data)
    mob_proj_initial_carry = (
        state.mob_projectiles.position,
        state.mob_projectiles.mask,
        state.player_health,
        state.is_sleeping,
        state.is_resting,
        state.map,
    )

    def _move_mob_projectile(carry, projectile_index):
        (proj_positions, proj_masks, player_health, is_sleeping, is_resting, map_data) = carry

        current_pos = proj_positions[lvl, projectile_index]
        current_mask = proj_masks[lvl, projectile_index]
        direction = state.mob_projectile_directions[lvl, projectile_index]

        proposed_position = current_pos + direction
        proposed_position_in_player = (proposed_position == state.player_position).all()

        proposed_position_in_bounds = in_bounds(state, proposed_position)
        in_wall = is_in_solid_block(state, proposed_position)
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(map_data[lvl, proposed_position[0], proposed_position[1]] == BlockType.WATER.value),
        )  # Arrows can go over water
        in_mob = is_in_mob(state, proposed_position)

        continue_move = jnp.logical_and(proposed_position_in_bounds, jnp.logical_not(in_wall))
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))

        hit_player0 = jnp.logical_and((current_pos == state.player_position).all(), current_mask)
        hit_player1 = jnp.logical_and(proposed_position_in_player, current_mask)
        hit_player = jnp.logical_or(hit_player0, hit_player1)

        continue_move = jnp.logical_and(continue_move, jnp.logical_not(hit_player))

        new_position = proposed_position
        new_mask = jnp.logical_and(continue_move, current_mask)

        # Check if projectile hits furnace or crafting table
        hit_bench_or_furnace = jnp.logical_or(
            map_data[lvl, new_position[0], new_position[1]] == BlockType.FURNACE.value,
            map_data[lvl, new_position[0], new_position[1]] == BlockType.CRAFTING_TABLE.value,
        )
        removing_block = jnp.logical_and(hit_bench_or_furnace, current_mask)
        new_block = jax.lax.select(
            removing_block,
            BlockType.PATH.value,
            map_data[lvl, new_position[0], new_position[1]],
        )

        # Compute damage
        projectile_type = state.mob_projectiles.type_id[lvl, projectile_index]
        projectile_damage = get_damage_done_to_player(
            state, static_params,
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value],
        )

        # Update arrays
        new_proj_positions = proj_positions.at[lvl, projectile_index].set(new_position)
        new_proj_masks = proj_masks.at[lvl, projectile_index].set(new_mask)
        new_player_health = player_health - projectile_damage * hit_player
        new_is_sleeping = jnp.logical_and(is_sleeping, jnp.logical_not(hit_player))
        new_is_resting = jnp.logical_and(is_resting, jnp.logical_not(hit_player))
        new_map_data = map_data.at[lvl, new_position[0], new_position[1]].set(new_block)

        new_carry = (new_proj_positions, new_proj_masks, new_player_health, new_is_sleeping, new_is_resting, new_map_data)
        return new_carry, None

    mob_proj_final_carry, _ = jax.lax.scan(
        _move_mob_projectile,
        mob_proj_initial_carry,
        jnp.arange(static_params.max_mob_projectiles),
    )

    # Apply mob projectile updates with ONE state.replace
    (proj_positions, proj_masks, player_health, is_sleeping, is_resting, map_data) = mob_proj_final_carry
    state = state.replace(
        mob_projectiles=state.mob_projectiles.replace(
            position=proj_positions,
            mask=proj_masks,
        ),
        player_health=player_health,
        is_sleeping=is_sleeping,
        is_resting=is_resting,
        map=map_data,
    )

    def _move_player_projectile(state, projectile_index):
        projectiles = state.player_projectiles

        projectile_type = state.player_projectiles.type_id[
            state.player_level, projectile_index
        ]

        projectile_damage_vector = (
            MOB_TYPE_DAMAGE_MAPPING[projectile_type, MobType.PROJECTILE.value]
            * projectiles.mask[state.player_level, projectile_index]
        )

        is_arrow = jnp.logical_or(
            projectile_type == ProjectileType.ARROW.value,
            projectile_type == ProjectileType.ARROW2.value,
        )

        # Bow enchantment
        arrow_damage_add = jnp.zeros(3, dtype=jnp.float32)
        arrow_damage_add = arrow_damage_add.at[state.bow_enchantment].set(
            projectile_damage_vector[0] / 2
        )
        arrow_damage_add = arrow_damage_add.at[0].set(0)

        projectile_damage_vector += jax.lax.select(
            is_arrow,
            arrow_damage_add,
            jnp.zeros(3, dtype=jnp.float32),
        )

        # Apply attribute scaling
        arrow_damage_coeff = 1 + 0.2 * (state.player_dexterity - 1)
        magic_damage_coeff = 1 + 0.5 * (state.player_intelligence - 1)

        projectile_damage_vector *= jax.lax.select(
            is_arrow,
            arrow_damage_coeff,
            1.0,
        )

        projectile_damage_vector *= jax.lax.select(
            jnp.logical_or(
                projectile_type == ProjectileType.FIREBALL.value,
                projectile_type == ProjectileType.ICEBALL.value,
            ),
            magic_damage_coeff,
            1.0,
        )

        proposed_position = (
            projectiles.position[state.player_level, projectile_index]
            + state.player_projectile_directions[state.player_level, projectile_index]
        )

        proposed_position_in_bounds = in_bounds(state, proposed_position)
        in_wall = is_in_solid_block(state, proposed_position)
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[state.player_level][
                    proposed_position[0], proposed_position[1]
                ]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water

        state, did_attack_mob0, did_kill_mob0 = attack_mob(
            state,
            projectiles.position[state.player_level, projectile_index],
            projectile_damage_vector,
            False,
        )

        projectile_damage_vector = projectile_damage_vector * (1 - did_attack_mob0)

        state, did_attack_mob1, did_kill_mob1 = attack_mob(
            state, proposed_position, projectile_damage_vector, False
        )

        did_attack_mob = jnp.logical_or(did_attack_mob0, did_attack_mob1)

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(did_attack_mob))
        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(
            continue_move, projectiles.mask[state.player_level, projectile_index]
        )

        state = state.replace(
            player_projectiles=state.player_projectiles.replace(
                position=state.player_projectiles.position.at[
                    state.player_level, projectile_index
                ].set(position),
                mask=state.player_projectiles.mask.at[
                    state.player_level, projectile_index
                ].set(new_mask),
            ),
        )

        return state, None

    state, _ = jax.lax.scan(
        _move_player_projectile,
        state,
        jnp.arange(static_params.max_player_projectiles),
    )

    return state


def update_player_intrinsics(state, action, static_params):
    # Cache max values (each called 3 times in this function)
    max_energy = get_max_energy(state)
    max_health = get_max_health(state)

    # Start sleeping?
    is_starting_sleep = jnp.logical_and(
        action == Action.SLEEP.value, state.player_energy < max_energy
    )
    new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
    state = state.replace(is_sleeping=new_is_sleeping)

    # Wake up?
    is_waking_up = jnp.logical_and(
        state.player_energy >= max_energy, state.is_sleeping
    )
    new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_sleeping=new_is_sleeping,
        achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
            jnp.logical_or(state.achievements[Achievement.WAKE_UP.value], is_waking_up)
        ),
    )

    # Start resting?
    is_starting_rest = jnp.logical_and(
        action == Action.REST.value, state.player_health < max_health
    )
    new_is_resting = jnp.logical_or(state.is_resting, is_starting_rest)
    state = state.replace(is_resting=new_is_resting)

    # Wake up from resting
    is_waking_up = jnp.logical_and(
        state.is_resting,
        jnp.logical_or(
            state.player_health >= max_health,
            jnp.logical_or(state.player_food <= 0, state.player_drink <= 0),
        ),
    )
    new_is_resting = jnp.logical_and(state.is_resting, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_resting=new_is_resting,
    )

    not_boss = jnp.logical_not(is_fighting_boss(state, static_params))

    intrinsic_decay_coeff = 1.0 - (0.125 * (state.player_dexterity - 1))

    # Hunger
    hunger_add = jax.lax.select(state.is_sleeping, 0.5, 1.0) * intrinsic_decay_coeff
    new_hunger = state.player_hunger + hunger_add

    hungered_food = jnp.maximum(state.player_food - 1 * not_boss, 0)
    new_food = jax.lax.select(new_hunger > 25, hungered_food, state.player_food)
    new_hunger = jax.lax.select(new_hunger > 25, 0.0, new_hunger)

    state = state.replace(
        player_hunger=new_hunger,
        player_food=new_food,
    )

    # Thirst
    thirst_add = jax.lax.select(state.is_sleeping, 0.5, 1.0) * intrinsic_decay_coeff
    new_thirst = state.player_thirst + thirst_add
    thirsted_drink = jnp.maximum(state.player_drink - 1 * not_boss, 0)
    new_drink = jax.lax.select(new_thirst > 20, thirsted_drink, state.player_drink)
    new_thirst = jax.lax.select(new_thirst > 20, 0.0, new_thirst)

    state = state.replace(
        player_thirst=new_thirst,
        player_drink=new_drink,
    )

    # Fatigue
    new_fatigue = jax.lax.select(
        state.is_sleeping,
        jnp.minimum(state.player_fatigue - 1, 0),
        state.player_fatigue + intrinsic_decay_coeff,
    )

    new_energy = jax.lax.select(
        new_fatigue > 30,
        jnp.maximum(state.player_energy - 1 * not_boss, 0),
        state.player_energy,
    )
    new_fatigue = jax.lax.select(new_fatigue > 30, 0.0, new_fatigue)

    new_energy = jax.lax.select(
        new_fatigue < -10,
        jnp.minimum(state.player_energy + 1, max_energy),
        new_energy,
    )
    new_fatigue = jax.lax.select(new_fatigue < -10, 0.0, new_fatigue)

    state = state.replace(
        player_fatigue=new_fatigue,
        player_energy=new_energy,
    )

    # Health
    necessities = jnp.array(
        [
            state.player_food > 0,
            state.player_drink > 0,
            jnp.logical_or(state.player_energy > 0, state.is_sleeping),
        ],
        dtype=bool,
    )

    all_necessities = necessities.all()
    recover_all = jax.lax.select(state.is_sleeping, 2.0, 1.0)
    recover_not_all = jax.lax.select(state.is_sleeping, -0.5, -1.0) * not_boss
    recover_add = jax.lax.select(all_necessities, recover_all, recover_not_all)

    new_recover = state.player_recover + recover_add

    recovered_health = jnp.minimum(state.player_health + 1, max_health)
    derecovered_health = state.player_health - 1

    new_health = jax.lax.select(new_recover > 25, recovered_health, state.player_health)
    new_recover = jax.lax.select(new_recover > 25, 0.0, new_recover)
    new_health = jax.lax.select(new_recover < -15, derecovered_health, new_health)
    new_recover = jax.lax.select(new_recover < -15, 0.0, new_recover)

    state = state.replace(
        player_recover=new_recover,
        player_health=new_health,
    )

    # Mana
    mana_recover_coeff = 1 + 0.25 * (state.player_intelligence - 1)
    new_recover_mana = (
        jax.lax.select(
            state.is_sleeping,
            state.player_recover_mana + 2,
            state.player_recover_mana + 1,
        )
        * mana_recover_coeff
    )

    new_mana = jax.lax.select(
        new_recover_mana > 30, state.player_mana + 1, state.player_mana
    )
    new_recover_mana = jax.lax.select(new_recover_mana > 30, 0.0, new_recover_mana)

    state = state.replace(
        player_recover_mana=new_recover_mana,
        player_mana=new_mana,
    )

    return state


def update_plants(state, static_params):
    growing_plants_age = state.growing_plants_age + 1
    growing_plants_age *= state.growing_plants_mask

    finished_growing_plants = growing_plants_age >= 600

    new_plant_blocks = jnp.where(
        finished_growing_plants,
        BlockType.RIPE_PLANT.value,
        BlockType.PLANT.value,
    )

    def _set_plant_block(map, plant_index):
        new_block = jax.lax.select(
            finished_growing_plants[plant_index],
            new_plant_blocks[plant_index],
            map[
                state.growing_plants_positions[plant_index][0],
                state.growing_plants_positions[plant_index][1],
            ],
        )
        map = map.at[
            state.growing_plants_positions[plant_index][0],
            state.growing_plants_positions[plant_index][1],
        ].set(new_block)
        return map, None

    new_map, _ = jax.lax.scan(
        _set_plant_block,
        state.map[0],
        jnp.arange(static_params.max_growing_plants),
    )

    new_whole_map = state.map.at[0].set(new_map)

    state = state.replace(
        map=new_whole_map,
        growing_plants_age=growing_plants_age,
    )

    return state


def move_player(state, action, params):
    proposed_position = state.player_position + DIRECTIONS[action]

    valid_move = is_position_in_bounds_not_in_mob_not_colliding(
        state, proposed_position, COLLISION_LAND_CREATURE
    )
    valid_move = jnp.logical_or(valid_move, params.god_mode)

    position = state.player_position + valid_move.astype(jnp.int32) * DIRECTIONS[action]

    is_new_direction = jnp.sum(jnp.abs(DIRECTIONS[action])) != 0
    new_direction = (
        state.player_direction * (1 - is_new_direction) + action * is_new_direction
    )

    state = state.replace(
        player_position=position,
        player_direction=new_direction,
    )

    return state


def spawn_mobs(state, keys, params, static_params):
    # keys shape: (6,) - pre-split keys passed from caller
    k_passive_chance, k_passive_pos, k_melee_chance, k_melee_pos, k_ranged_chance, k_ranged_pos = keys

    # Cache frequently accessed values
    lvl = state.player_level
    map_floor = state.map[lvl]
    mob_map_floor = state.mob_map[lvl]
    fighting_boss = is_fighting_boss(state, static_params)

    player_distance_map = get_distance_map(
        state.player_position, static_params.map_size
    )
    # Use lookup table instead of 3 OR comparisons
    grave_map = BLOCK_IS_GRAVE[map_floor]

    monster_spawn_coeff = (
        1
        + (state.monsters_killed[state.player_level] < MONSTERS_KILLED_TO_CLEAR_LEVEL)
        * 2
    )  # Triple spawn rate if we are on an uncleared level

    monster_spawn_coeff *= jax.lax.select(
        fighting_boss,
        is_boss_spawn_wave(state, static_params) * 1000,
        1,
    )

    # Passive mobs
    can_spawn_passive_mob = (
        state.passive_mobs.mask[state.player_level].sum()
        < static_params.max_passive_mobs
    )

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob,
        jax.random.uniform(k_passive_chance) < FLOOR_MOB_SPAWN_CHANCE[state.player_level, 0],
    )

    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, jnp.logical_not(fighting_boss)
    )

    # Use lookup table instead of 4 OR comparisons
    all_valid_blocks_map = BLOCK_IS_VALID_SPAWN[map_floor]
    grass_map = map_floor == BlockType.GRASS.value
    path_map = map_floor == BlockType.PATH.value
    new_passive_mob_type = FLOOR_MOB_MAPPING[lvl, MobType.PASSIVE.value]

    passive_mobs_can_spawn_map = all_valid_blocks_map

    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map > 3
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    passive_mobs_can_spawn_map = jnp.logical_and(
        passive_mobs_can_spawn_map, jnp.logical_not(mob_map_floor)
    )
    can_spawn_passive_mob = jnp.logical_and(
        can_spawn_passive_mob, passive_mobs_can_spawn_map.sum() > 0
    )

    passive_mob_position = jax.random.choice(
        k_passive_pos,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(passive_mobs_can_spawn_map, -1)
        / jnp.sum(passive_mobs_can_spawn_map),
    )
    passive_mob_position = jnp.array(
        [
            passive_mob_position // static_params.map_size[0],
            passive_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_passive_mob_index = jnp.argmax(
        jnp.logical_not(state.passive_mobs.mask[state.player_level])
    )

    new_passive_mob_position = jax.lax.select(
        can_spawn_passive_mob,
        passive_mob_position,
        state.passive_mobs.position[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_health = jax.lax.select(
        can_spawn_passive_mob,
        MOB_TYPE_HEALTH_MAPPING[new_passive_mob_type, MobType.PASSIVE.value],
        state.passive_mobs.health[state.player_level, new_passive_mob_index],
    )

    new_passive_mob_mask = jax.lax.select(
        can_spawn_passive_mob,
        True,
        state.passive_mobs.mask[state.player_level, new_passive_mob_index],
    )

    passive_mobs = Mobs(
        position=state.passive_mobs.position.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_position),
        health=state.passive_mobs.health.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_health),
        mask=state.passive_mobs.mask.at[state.player_level, new_passive_mob_index].set(
            new_passive_mob_mask
        ),
        attack_cooldown=state.passive_mobs.attack_cooldown,
        type_id=state.passive_mobs.type_id.at[
            state.player_level, new_passive_mob_index
        ].set(new_passive_mob_type),
    )

    state = state.replace(
        passive_mobs=passive_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_passive_mob_position[0], new_passive_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_passive_mob_position[0],
                    new_passive_mob_position[1],
                ],
                new_passive_mob_mask,
            )
        ),
    )

    # Monsters
    monsters_can_spawn_player_range_map = player_distance_map > 9
    monsters_can_spawn_player_range_map_boss = player_distance_map <= 6

    monsters_can_spawn_player_range_map = jax.lax.select(
        fighting_boss,
        monsters_can_spawn_player_range_map_boss,
        monsters_can_spawn_player_range_map,
    )

    # Melee mobs
    can_spawn_melee_mob = (
        state.melee_mobs.mask[state.player_level].sum() < static_params.max_melee_mobs
    )

    new_melee_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.MELEE.value]
    new_melee_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.MELEE.value
    ]

    new_melee_mob_type = jax.lax.select(
        fighting_boss,
        new_melee_mob_type_boss,
        new_melee_mob_type,
    )

    melee_mob_spawn_chance = FLOOR_MOB_SPAWN_CHANCE[
        state.player_level, 1
    ] + FLOOR_MOB_SPAWN_CHANCE[state.player_level, 3] * jnp.square(
        1 - state.light_level
    )
    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob,
        jax.random.uniform(k_melee_chance) < melee_mob_spawn_chance * monster_spawn_coeff,
    )

    melee_mobs_can_spawn_map = jax.lax.select(
        fighting_boss, grave_map, all_valid_blocks_map
    )

    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    melee_mobs_can_spawn_map = jnp.logical_and(
        melee_mobs_can_spawn_map, jnp.logical_not(mob_map_floor)
    )

    can_spawn_melee_mob = jnp.logical_and(
        can_spawn_melee_mob, melee_mobs_can_spawn_map.sum() > 0
    )

    melee_mob_position = jax.random.choice(
        k_melee_pos,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(melee_mobs_can_spawn_map, -1) / jnp.sum(melee_mobs_can_spawn_map),
    )
    melee_mob_position = jnp.array(
        [
            melee_mob_position // static_params.map_size[0],
            melee_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_melee_mob_index = jnp.argmax(
        jnp.logical_not(state.melee_mobs.mask[state.player_level])
    )

    new_melee_mob_position = jax.lax.select(
        can_spawn_melee_mob,
        melee_mob_position,
        state.melee_mobs.position[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_health = jax.lax.select(
        can_spawn_melee_mob,
        MOB_TYPE_HEALTH_MAPPING[new_melee_mob_type, MobType.MELEE.value],
        state.melee_mobs.health[state.player_level, new_melee_mob_index],
    )

    new_melee_mob_mask = jax.lax.select(
        can_spawn_melee_mob,
        True,
        state.melee_mobs.mask[state.player_level, new_melee_mob_index],
    )

    melee_mobs = Mobs(
        position=state.melee_mobs.position.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_position),
        health=state.melee_mobs.health.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_health
        ),
        mask=state.melee_mobs.mask.at[state.player_level, new_melee_mob_index].set(
            new_melee_mob_mask
        ),
        attack_cooldown=state.melee_mobs.attack_cooldown,
        type_id=state.melee_mobs.type_id.at[
            state.player_level, new_melee_mob_index
        ].set(new_melee_mob_type),
    )

    state = state.replace(
        melee_mobs=melee_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_melee_mob_position[0], new_melee_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_melee_mob_position[0],
                    new_melee_mob_position[1],
                ],
                new_melee_mob_mask,
            )
        ),
    )

    # Ranged mobs
    can_spawn_ranged_mob = (
        state.ranged_mobs.mask[state.player_level].sum() < static_params.max_ranged_mobs
    )

    new_ranged_mob_type = FLOOR_MOB_MAPPING[state.player_level, MobType.RANGED.value]
    new_ranged_mob_type_boss = FLOOR_MOB_MAPPING[
        state.boss_progress, MobType.RANGED.value
    ]

    new_ranged_mob_type = jax.lax.select(
        fighting_boss,
        new_ranged_mob_type_boss,
        new_ranged_mob_type,
    )

    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob,
        jax.random.uniform(k_ranged_chance)
        < FLOOR_MOB_SPAWN_CHANCE[state.player_level, 2] * monster_spawn_coeff,
    )

    # Hack for deep thing
    ranged_mobs_can_spawn_map = jax.lax.select(
        new_ranged_mob_type == 5,
        map_floor == BlockType.WATER.value,
        all_valid_blocks_map,
    )
    ranged_mobs_can_spawn_map = jax.lax.select(
        fighting_boss, grave_map, ranged_mobs_can_spawn_map
    )

    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, monsters_can_spawn_player_range_map
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    ranged_mobs_can_spawn_map = jnp.logical_and(
        ranged_mobs_can_spawn_map, jnp.logical_not(mob_map_floor)
    )

    can_spawn_ranged_mob = jnp.logical_and(
        can_spawn_ranged_mob, ranged_mobs_can_spawn_map.sum() > 0
    )

    ranged_mob_position = jax.random.choice(
        k_ranged_pos,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(ranged_mobs_can_spawn_map, -1)
        / jnp.sum(ranged_mobs_can_spawn_map),
    )
    ranged_mob_position = jnp.array(
        [
            ranged_mob_position // static_params.map_size[0],
            ranged_mob_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_ranged_mob_index = jnp.argmax(
        jnp.logical_not(state.ranged_mobs.mask[state.player_level])
    )

    new_ranged_mob_position = jax.lax.select(
        can_spawn_ranged_mob,
        ranged_mob_position,
        state.ranged_mobs.position[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_health = jax.lax.select(
        can_spawn_ranged_mob,
        MOB_TYPE_HEALTH_MAPPING[new_ranged_mob_type, MobType.RANGED.value],
        state.ranged_mobs.health[state.player_level, new_ranged_mob_index],
    )

    new_ranged_mob_mask = jax.lax.select(
        can_spawn_ranged_mob,
        True,
        state.ranged_mobs.mask[state.player_level, new_ranged_mob_index],
    )

    ranged_mobs = Mobs(
        position=state.ranged_mobs.position.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_position),
        health=state.ranged_mobs.health.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_health),
        mask=state.ranged_mobs.mask.at[state.player_level, new_ranged_mob_index].set(
            new_ranged_mob_mask
        ),
        attack_cooldown=state.ranged_mobs.attack_cooldown,
        type_id=state.ranged_mobs.type_id.at[
            state.player_level, new_ranged_mob_index
        ].set(new_ranged_mob_type),
    )

    state = state.replace(
        ranged_mobs=ranged_mobs,
        mob_map=state.mob_map.at[
            state.player_level, new_ranged_mob_position[0], new_ranged_mob_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[
                    state.player_level,
                    new_ranged_mob_position[0],
                    new_ranged_mob_position[1],
                ],
                new_ranged_mob_mask,
            )
        ),
    )

    return state


def change_floor(
    state: EnvState, action, env_params: EnvParams, static_params: StaticEnvParams
):
    on_down_ladder = (
        state.item_map[
            state.player_level, state.player_position[0], state.player_position[1]
        ]
        == ItemType.LADDER_DOWN.value
    )
    is_moving_down = jnp.logical_and(
        action == Action.DESCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            jnp.logical_and(
                on_down_ladder,
                state.monsters_killed[state.player_level]
                >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
            ),
        ),
    )
    is_moving_down = jnp.logical_and(
        is_moving_down, state.player_level < static_params.num_levels - 1
    )

    moving_down_position = state.up_ladders[state.player_level + 1]

    on_up_ladder = (
        state.item_map[
            state.player_level, state.player_position[0], state.player_position[1]
        ]
        == ItemType.LADDER_UP.value
    )
    is_moving_up = jnp.logical_and(
        action == Action.ASCEND.value,
        jnp.logical_or(
            env_params.god_mode,
            on_up_ladder,
        ),
    )
    is_moving_up = jnp.logical_and(is_moving_up, state.player_level > 0)

    moving_up_position = state.down_ladders[state.player_level - 1]

    is_not_moving = jnp.logical_not(jnp.logical_or(is_moving_up, is_moving_down))

    delta_floor = 1 * is_moving_down - 1 * is_moving_up
    position = (
        (state.player_position * is_not_moving)
        + (is_moving_down * moving_down_position)
        + (is_moving_up * moving_up_position)
    )

    move_down_achievement = LEVEL_ACHIEVEMENT_MAP[state.player_level + delta_floor]

    new_achievements = state.achievements.at[move_down_achievement].set(
        jnp.logical_or(
            (state.player_level + delta_floor) != 0,
            state.achievements[move_down_achievement],
        )
    )

    new_floor = jnp.logical_and(
        (state.player_level + delta_floor) != 0,
        jnp.logical_not(state.achievements[move_down_achievement]),
    )

    state = state.replace(
        player_level=state.player_level + delta_floor,
        player_position=position,
        achievements=new_achievements,
        player_xp=state.player_xp + 1 * new_floor,
    )

    return state


def shoot_projectile(state: EnvState, action: int, static_params: StaticEnvParams):
    # Arrow
    is_shooting_arrow = jnp.logical_and(
        action == Action.SHOOT_ARROW.value,
        jnp.logical_and(
            state.inventory.bow >= 1,
            jnp.logical_and(
                state.inventory.arrows >= 1,
                state.player_projectiles.mask[state.player_level].sum()
                < static_params.max_player_projectiles,
            ),
        ),
    )

    new_player_projectiles, new_player_projectile_directions = spawn_projectile(
        state,
        static_params,
        state.player_projectiles,
        state.player_projectile_directions,
        state.player_position,
        is_shooting_arrow,
        DIRECTIONS[state.player_direction],
        ProjectileType.ARROW2.value,
    )

    new_achievements = state.achievements.at[Achievement.FIRE_BOW.value].set(
        jnp.logical_or(
            state.achievements[Achievement.FIRE_BOW.value], is_shooting_arrow
        )
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        inventory=state.inventory.replace(
            arrows=state.inventory.arrows - 1 * is_shooting_arrow
        ),
        achievements=new_achievements,
    )


def cast_spell(state, action, static_params):
    # Arrow
    is_casting_fireball = jnp.logical_and(
        action == Action.CAST_FIREBALL.value,
        jnp.logical_and(
            state.player_mana >= 2,
            state.player_projectiles.mask[state.player_level].sum()
            < static_params.max_player_projectiles,
        ),
    )
    is_casting_fireball = jnp.logical_and(is_casting_fireball, state.learned_spells[0])

    is_casting_iceball = jnp.logical_and(
        action == Action.CAST_ICEBALL.value,
        jnp.logical_and(
            state.player_mana >= 2,
            state.player_projectiles.mask[state.player_level].sum()
            < static_params.max_player_projectiles,
        ),
    )
    is_casting_iceball = jnp.logical_and(is_casting_iceball, state.learned_spells[1])

    is_casting_spell = jnp.logical_or(is_casting_fireball, is_casting_iceball)
    projectile_type = (
        is_casting_fireball * ProjectileType.FIREBALL.value
        + is_casting_iceball * ProjectileType.ICEBALL.value
    )

    new_player_projectiles, new_player_projectile_directions = spawn_projectile(
        state,
        static_params,
        state.player_projectiles,
        state.player_projectile_directions,
        state.player_position,
        is_casting_spell,
        DIRECTIONS[state.player_direction],
        projectile_type,
    )

    casting_achievement = (
        is_casting_fireball * Achievement.CAST_FIREBALL.value
        + is_casting_iceball * Achievement.CAST_ICEBALL.value
    )
    new_achievements = state.achievements.at[casting_achievement].set(
        jnp.logical_or(state.achievements[casting_achievement], is_casting_spell)
    )

    return state.replace(
        player_projectiles=new_player_projectiles,
        player_projectile_directions=new_player_projectile_directions,
        player_mana=state.player_mana - is_casting_spell * 2,
        achievements=new_achievements,
    )


def drink_potion(state, action):
    # OPTIMIZED: Replace 6 individual potion checks with lookup-based approach
    # Potion actions are consecutive: RED=29, GREEN=30, ..., YELLOW=34
    # So potion_index = action - 29 when action is in [29, 34]

    # Compute potion index from action (will be in range [0, 5] for valid potion actions)
    potion_index_raw = action - POTION_ACTION_START

    # Check if this is a valid potion action
    is_potion_action = jnp.logical_and(
        action >= POTION_ACTION_START,
        action <= POTION_ACTION_END
    )

    # Clamp to valid range for safe indexing (will be ignored if not a potion action)
    potion_index = jnp.clip(potion_index_raw, 0, NUM_POTIONS - 1)

    # Check if player has this potion
    has_potion = state.inventory.potions[potion_index] > 0

    # Final check: valid potion action AND has the potion
    is_drinking_potion = jnp.logical_and(is_potion_action, has_potion)

    # Use potion_index for effect lookup (safe because we clipped it)
    drinking_potion_index = jax.lax.select(is_drinking_potion, potion_index, 0)

    # Potion mapping (randomized per episode)
    potion_effect_index = state.potion_mapping[drinking_potion_index]

    # Potion effects via lookup instead of 6 comparisons
    # Effects: 0=+8 health, 1=-3 health, 2=+8 mana, 3=-3 mana, 4=+8 energy, 5=-3 energy
    POTION_HEALTH_DELTA = jnp.array([8, -3, 0, 0, 0, 0], dtype=jnp.int32)
    POTION_MANA_DELTA = jnp.array([0, 0, 8, -3, 0, 0], dtype=jnp.int32)
    POTION_ENERGY_DELTA = jnp.array([0, 0, 0, 0, 8, -3], dtype=jnp.int32)

    delta_health = is_drinking_potion * POTION_HEALTH_DELTA[potion_effect_index]
    delta_mana = is_drinking_potion * POTION_MANA_DELTA[potion_effect_index]
    delta_energy = is_drinking_potion * POTION_ENERGY_DELTA[potion_effect_index]

    new_achievements = state.achievements.at[Achievement.DRINK_POTION.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DRINK_POTION.value], is_drinking_potion
        )
    )

    return state.replace(
        inventory=state.inventory.replace(
            potions=state.inventory.potions.at[drinking_potion_index].set(
                state.inventory.potions[drinking_potion_index] - 1 * is_drinking_potion
            )
        ),
        player_health=state.player_health + delta_health,
        player_mana=state.player_mana + delta_mana,
        player_energy=state.player_energy + delta_energy,
        achievements=new_achievements,
    )


def read_book(key, state, action):
    # key: single pre-split key passed from caller
    is_reading_book = jnp.logical_and(
        action == Action.READ_BOOK.value, state.inventory.books > 0
    )
    spells_to_learn = jnp.logical_not(state.learned_spells).astype(float)
    spells_to_learn /= spells_to_learn.sum()

    spell_to_learn_index = jax.random.choice(
        key, jnp.arange(2), shape=(), p=spells_to_learn
    )

    learn_spell_achievement = jax.lax.select(
        spell_to_learn_index,
        Achievement.LEARN_ICEBALL.value,
        Achievement.LEARN_FIREBALL.value,
    )

    new_achievements = state.achievements.at[learn_spell_achievement].set(
        jnp.logical_or(state.achievements[learn_spell_achievement], is_reading_book)
    )

    return state.replace(
        inventory=state.inventory.replace(
            books=state.inventory.books - 1 * is_reading_book
        ),
        learned_spells=state.learned_spells.at[spell_to_learn_index].set(
            jnp.logical_or(state.learned_spells[spell_to_learn_index], is_reading_book)
        ),
        achievements=new_achievements,
    )


def enchant(key, state: EnvState, action):
    # key: single pre-split key passed from caller
    target_block_position = state.player_position + DIRECTIONS[state.player_direction]
    target_block = state.map[
        state.player_level, target_block_position[0], target_block_position[1]
    ]

    # Compute fire table check once, reuse for all dependent calculations
    is_fire_table = target_block == BlockType.ENCHANTMENT_TABLE_FIRE.value
    is_ice_table = target_block == BlockType.ENCHANTMENT_TABLE_ICE.value
    target_block_is_enchantment_table = jnp.logical_or(is_fire_table, is_ice_table)

    # Fire = 1, Ice = 2
    enchantment_type = jax.lax.select(is_fire_table, 1, 2)

    # Fire uses ruby, Ice uses sapphire
    num_gems = jax.lax.select(is_fire_table, state.inventory.ruby, state.inventory.sapphire)

    could_enchant = jnp.logical_and(
        state.player_mana >= 9,
        jnp.logical_and(target_block_is_enchantment_table, num_gems >= 1),
    )

    is_enchanting_bow = jnp.logical_and(
        could_enchant,
        jnp.logical_and(action == Action.ENCHANT_BOW.value, state.inventory.bow > 0),
    )

    is_enchanting_sword = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_SWORD.value, state.inventory.sword > 0
        ),
    )

    is_enchanting_armour = jnp.logical_and(
        could_enchant,
        jnp.logical_and(
            action == Action.ENCHANT_ARMOUR.value, state.inventory.armour.sum() > 0
        ),
    )

    unenchanted_armour = state.armour_enchantments == 0
    opposite_enchanted_armour = jnp.logical_and(
        state.armour_enchantments != 0, state.armour_enchantments != enchantment_type
    )

    armour_targets = (
        unenchanted_armour + (unenchanted_armour.sum() == 0) * opposite_enchanted_armour
    )
    armour_target = jax.random.choice(key, jnp.arange(4), shape=(), p=armour_targets)

    is_enchanting = jnp.logical_or(
        is_enchanting_sword, jnp.logical_or(is_enchanting_bow, is_enchanting_armour)
    )

    new_sword_enchantment = (
        is_enchanting_sword * enchantment_type
        + (1 - is_enchanting_sword) * state.sword_enchantment
    )
    new_bow_enchantment = (
        is_enchanting_bow * enchantment_type
        + (1 - is_enchanting_bow) * state.bow_enchantment
    )

    new_armour_enchantments = state.armour_enchantments.at[armour_target].set(
        is_enchanting_armour * enchantment_type
        + (1 - is_enchanting_armour) * state.armour_enchantments[armour_target]
    )

    new_sapphire = state.inventory.sapphire - 1 * is_enchanting * (
        enchantment_type == 2
    )
    new_ruby = state.inventory.ruby - 1 * is_enchanting * (enchantment_type == 1)
    new_mana = state.player_mana - 9 * is_enchanting

    # Batched achievements update (2 ops -> 1)
    enchant_triggers = jnp.array([is_enchanting_sword, is_enchanting_armour])
    current_enchant_achievements = state.achievements[ENCHANT_ACHIEVEMENT_INDICES]
    updated_enchant_achievements = jnp.logical_or(current_enchant_achievements, enchant_triggers)
    new_achievements = state.achievements.at[ENCHANT_ACHIEVEMENT_INDICES].set(updated_enchant_achievements)

    return state.replace(
        sword_enchantment=new_sword_enchantment,
        bow_enchantment=new_bow_enchantment,
        armour_enchantments=new_armour_enchantments,
        inventory=state.inventory.replace(
            sapphire=new_sapphire,
            ruby=new_ruby,
        ),
        player_mana=new_mana,
        achievements=new_achievements,
    )


def boss_logic(state, static_params):
    new_achievements = state.achievements.at[Achievement.DEFEAT_NECROMANCER.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DEFEAT_NECROMANCER.value],
            has_beaten_boss(state, static_params),
        )
    )

    return state.replace(
        boss_timesteps_to_spawn_this_round=state.boss_timesteps_to_spawn_this_round
        - 1 * is_fighting_boss(state, static_params),
        achievements=new_achievements,
    )


def calculate_inventory_achievements(state):
    # Some achievements (e.g. make_diamond_pickaxe) can be achieved in multiple ways (finding in chest or crafting)
    # Rather than duplicating achievement code, we simply look in the inventory for these types of achievements
    # at the end of each timestep
    #
    # OPTIMIZED: 19 separate .at[].set() calls -> 3 batched updates

    inv = state.inventory
    achievements = state.achievements

    # Batch 1: Collection achievements (item > 0)
    # Order must match INVENTORY_ACHIEVEMENT_COLLECT_INDICES
    collect_triggers = jnp.array([
        inv.wood > 0,      # COLLECT_WOOD
        inv.stone > 0,     # COLLECT_STONE
        inv.coal > 0,      # COLLECT_COAL
        inv.iron > 0,      # COLLECT_IRON
        inv.diamond > 0,   # COLLECT_DIAMOND
        inv.ruby > 0,      # COLLECT_RUBY
        inv.sapphire > 0,  # COLLECT_SAPPHIRE
        inv.sapling > 0,   # COLLECT_SAPLING
        inv.bow > 0,       # FIND_BOW
        inv.arrows > 0,    # MAKE_ARROW
        inv.torches > 0,   # MAKE_TORCH
    ])
    current_collect = achievements[INVENTORY_ACHIEVEMENT_COLLECT_INDICES]
    achievements = achievements.at[INVENTORY_ACHIEVEMENT_COLLECT_INDICES].set(
        jnp.logical_or(current_collect, collect_triggers)
    )

    # Batch 2: Pickaxe level achievements
    # Order must match INVENTORY_ACHIEVEMENT_PICKAXE_INDICES
    pickaxe_triggers = jnp.array([
        inv.pickaxe >= 1,  # MAKE_WOOD_PICKAXE
        inv.pickaxe >= 2,  # MAKE_STONE_PICKAXE
        inv.pickaxe >= 3,  # MAKE_IRON_PICKAXE
        inv.pickaxe >= 4,  # MAKE_DIAMOND_PICKAXE
    ])
    current_pickaxe = achievements[INVENTORY_ACHIEVEMENT_PICKAXE_INDICES]
    achievements = achievements.at[INVENTORY_ACHIEVEMENT_PICKAXE_INDICES].set(
        jnp.logical_or(current_pickaxe, pickaxe_triggers)
    )

    # Batch 3: Sword level achievements
    # Order must match INVENTORY_ACHIEVEMENT_SWORD_INDICES
    sword_triggers = jnp.array([
        inv.sword >= 1,  # MAKE_WOOD_SWORD
        inv.sword >= 2,  # MAKE_STONE_SWORD
        inv.sword >= 3,  # MAKE_IRON_SWORD
        inv.sword >= 4,  # MAKE_DIAMOND_SWORD
    ])
    current_sword = achievements[INVENTORY_ACHIEVEMENT_SWORD_INDICES]
    achievements = achievements.at[INVENTORY_ACHIEVEMENT_SWORD_INDICES].set(
        jnp.logical_or(current_sword, sword_triggers)
    )

    return state.replace(achievements=achievements)


def level_up_attributes(state, action, params):
    can_level_up = state.player_xp >= 1

    is_levelling_up_dex = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_DEXTERITY.value,
            state.player_dexterity < params.max_attribute,
        ),
    )
    is_levelling_up_str = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_STRENGTH.value,
            state.player_strength < params.max_attribute,
        ),
    )
    is_levelling_up_int = jnp.logical_and(
        can_level_up,
        jnp.logical_and(
            action == Action.LEVEL_UP_INTELLIGENCE.value,
            state.player_intelligence < params.max_attribute,
        ),
    )
    is_levelling_up = jnp.logical_or(
        is_levelling_up_dex, jnp.logical_or(is_levelling_up_str, is_levelling_up_int)
    )

    return state.replace(
        player_dexterity=state.player_dexterity + 1 * is_levelling_up_dex,
        player_strength=state.player_strength + 1 * is_levelling_up_str,
        player_intelligence=state.player_intelligence + 1 * is_levelling_up_int,
        player_xp=state.player_xp - 1 * is_levelling_up,
    )


def update_diffs(
    state, init_intrinsics, updated_intrinsics, init_inventory, updated_inventory, init_achievements, updated_achievements
):
    intrinsics_diff = jnp.array(updated_intrinsics - init_intrinsics, dtype=jnp.int32)
    inventory_diff = jax.tree.map(
    lambda x, y: jnp.array(x - y, dtype=jnp.int32),
    updated_inventory,
    init_inventory
    )
    achievements_diff = jnp.array(jnp.array(updated_achievements, dtype=jnp.int32) - jnp.array(init_achievements, dtype=jnp.int32), dtype=jnp.bool)

    state = state.replace(
        intrinsics_diff=intrinsics_diff,
        inventory_diff=inventory_diff,
        achievements_diff=achievements_diff,
    )

    return state


def craftax_step(rng, state, action, params, static_params):
    # ONE monolithic split for all randomness in the step
    # Key allocation:
    #   [0:17]  = do_action (1 sapling + 16 chest)
    #   [17]    = read_book
    #   [18]    = enchant
    #   [19:25] = spawn_mobs (6 keys)
    #   [25:25+mob_keys] = update_mobs
    #   [-1]    = state_rng (for renderer)
    total_mob_keys = (
        4 * static_params.max_melee_mobs +
        1 * static_params.max_passive_mobs +
        3 * static_params.max_ranged_mobs
    )
    total_keys = 17 + 1 + 1 + 6 + total_mob_keys + 1  # = 26 + mob_keys
    keys = jax.random.split(rng, total_keys)

    # Slice out key ranges for each function
    action_keys = keys[0:17]
    read_key = keys[17]
    enchant_key = keys[18]
    spawn_keys = keys[19:25]
    mob_keys = keys[25:-1]
    state_rng = keys[-1]

    init_achievements = state.achievements
    init_health = state.player_health
    init_intrinsics = jnp.array(
        [
            state.player_health,
            state.player_food,
            state.player_drink,
            state.player_energy,
        ]
    )
    init_inventory = state.inventory

    closest_blocks_init = state.closest_blocks


    # Interrupt action if sleeping or resting
    action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)
    action = jax.lax.select(state.is_resting, Action.NOOP.value, action)

    # Change floor
    state = change_floor(state, action, params, static_params)


    # Crafting
    state = do_crafting(state, action)

    # Interact (mining, melee attacking, eating plants, drinking water)
    state = do_action(action_keys, state, action, static_params)

    # Placing
    state = place_block(state, action, static_params)

    # Shooting
    state = shoot_projectile(state, action, static_params)

    # Casting
    state = cast_spell(state, action, static_params)

    # Potions
    state = drink_potion(state, action)

    # Read
    state = read_book(read_key, state, action)

    # Enchant
    state = enchant(enchant_key, state, action)

    # Boss
    state = boss_logic(state, static_params)

    # Attributes
    state = level_up_attributes(state, action, params)

    # Movement
    old_position = state.player_position
    state = move_player(state, action, params)
    new_position = state.player_position

    # Mobs
    state = update_mobs(mob_keys, state, params, static_params)
    state = spawn_mobs(state, spawn_keys, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, action, static_params)

    # Cap inv
    state = clip_inventory_and_intrinsics(state, params)

    # Inventory achievements
    state = calculate_inventory_achievements(state)

    # Reward
    achievement_coefficients = ACHIEVEMENT_REWARD_MAP
    achievement_reward = (
        (state.achievements.astype(int) - init_achievements.astype(int))
        * achievement_coefficients
    ).sum()
    health_reward = (state.player_health - init_health) * 0.1
    reward = achievement_reward + health_reward

    # update 5 closest blocks
    if static_params.include_relative_positions:
        print("[TRACING] include_relative_positions=True branch in game_logic_warp.py")
        if static_params.use_floor_aware_closest_blocks:
            state = update_closest_blocks_per_floor(state, old_position, new_position, OBS_DIM, MAX_OBS_DIM, BlockType)
        else:
            state = update_closest_blocks(state, old_position, new_position, OBS_DIM, MAX_OBS_DIM, BlockType)

    updated_intrinsics = jnp.array(
        [
            state.player_health,
            state.player_food,
            state.player_drink,
            state.player_energy,
        ]
    )
    updated_inventory = state.inventory
    state = update_diffs(
        state,
        init_intrinsics,
        updated_intrinsics,
        init_inventory,
        updated_inventory,
        init_achievements,
        state.achievements,
    )
    state = state.replace(closest_blocks_prev=closest_blocks_init)

    state = state.replace(
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=state_rng,
    )

    return state, reward
