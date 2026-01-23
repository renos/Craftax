from craftax.craftax_classic.util.relative_positions import update_closest_blocks
import chex

from craftax.fabrax.constants import *
from craftax.fabrax.envs.craftax_state import *


def is_game_over(state, params):
    done_steps = state.timestep >= params.max_timesteps
    in_lava = (
        state.map[state.player_position[0], state.player_position[1]]
        == BlockType.LAVA.value
    )
    is_dead = state.player_health <= 0

    return done_steps | in_lava | is_dead


def in_bounds(state, position):
    in_bounds_x = jnp.logical_and(0 <= position[0], position[0] < state.map.shape[0])
    in_bounds_y = jnp.logical_and(0 <= position[1], position[1] < state.map.shape[1])
    return jnp.logical_and(in_bounds_x, in_bounds_y)


def is_in_wall(state, position):
    def _is_given_solid_block(unused, block):
        return None, state.map[position[0], position[1]] == block

    _, is_in_block = jax.lax.scan(_is_given_solid_block, None, SOLID_BLOCKS)

    return is_in_block.sum() > 0


def is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(state, position):
    pos_in_bounds = in_bounds(state, position)
    in_wall = is_in_wall(state, position)
    in_mob = is_in_mob(state, position)
    in_lava = state.map[position[0], position[1]] == BlockType.LAVA.value
    valid_move = jnp.logical_and(pos_in_bounds, jnp.logical_not(in_wall))
    valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_mob))
    valid_move = jnp.logical_and(valid_move, jnp.logical_not(in_lava))

    return valid_move


def get_player_attack_damage(state):
    damages = jnp.array(
        [
            1,
            2 * state.inventory.wood_sword,
            3 * state.inventory.stone_sword,
            5 * state.inventory.iron_sword,
            # NEW FABRAX WEAPONS
            4 * state.inventory.bronze_sword,  # Between stone and iron
            6 * state.inventory.steel_sword,   # Better than iron
        ],
        dtype=jnp.int32,
    )
    return jnp.max(damages)


def update_plants_with_eat(state, plant_position, static_params):
    def _is_plant(unused, index):
        return None, (state.growing_plants_positions[index] == plant_position).all()

    _, is_plant = jax.lax.scan(
        _is_plant, None, jnp.arange(static_params.max_growing_plants)
    )

    plant_index = jnp.argmax(is_plant)

    return state.growing_plants_age.at[plant_index].set(0)


def do_action(rng, state, action, static_params):
    old_state = state

    block_position = state.player_position + DIRECTIONS[state.player_direction]

    # Zombie
    def is_attacking_zombie_at_index(unused, zombie_index):
        in_zombie = (state.zombies.position[zombie_index] == block_position).all()
        return None, jnp.logical_and(in_zombie, state.zombies.mask[zombie_index])

    _, is_attacking_zombie_array = jax.lax.scan(
        is_attacking_zombie_at_index, None, jnp.arange(static_params.max_zombies)
    )
    is_attacking_zombie = is_attacking_zombie_array.sum() > 0
    target_zombie_index = jnp.argmax(is_attacking_zombie_array)
    new_zombies = state.zombies

    new_zombie_health = new_zombies.health.at[target_zombie_index].add(
        -get_player_attack_damage(state) * is_attacking_zombie
    )
    new_zombies = new_zombies.replace(health=new_zombie_health)

    old_mask = new_zombies.mask[target_zombie_index]
    new_zombies = new_zombies.replace(mask=new_zombies.health > 0)
    did_kill_zombie = jnp.logical_and(
        old_mask, jnp.logical_not(new_zombies.mask[target_zombie_index])
    )
    new_achievements = state.achievements.at[Achievement.DEFEAT_ZOMBIE.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DEFEAT_ZOMBIE.value], did_kill_zombie
        )
    )

    state = state.replace(
        zombies=new_zombies,
        achievements=new_achievements,
    )

    # Cow
    def is_attacking_cow_at_index(unused, cow_index):
        in_cow = (state.cows.position[cow_index] == block_position).all()
        return None, jnp.logical_and(in_cow, state.cows.mask[cow_index])

    _, is_attacking_cow_array = jax.lax.scan(
        is_attacking_cow_at_index, None, jnp.arange(static_params.max_cows)
    )
    is_attacking_cow = is_attacking_cow_array.sum() > 0
    target_cow_index = jnp.argmax(is_attacking_cow_array)
    new_cows = state.cows

    new_cow_health = new_cows.health.at[target_cow_index].add(
        -get_player_attack_damage(state) * is_attacking_cow
    )
    new_cows = new_cows.replace(health=new_cow_health)

    old_mask = new_cows.mask[target_cow_index]
    new_cows = new_cows.replace(mask=new_cows.health > 0)
    did_kill_cow = jnp.logical_and(
        old_mask, jnp.logical_not(new_cows.mask[target_cow_index])
    )
    new_achievements = state.achievements.at[Achievement.EAT_COW.value].set(
        jnp.logical_or(state.achievements[Achievement.EAT_COW.value], did_kill_cow)
    )
    new_food = jax.lax.select(
        did_kill_cow, jnp.minimum(9, state.player_food + 6), state.player_food
    )
    new_hunger = jax.lax.select(did_kill_cow, 0.0, state.player_hunger)
    # NEW FABRAX: Cows drop leather
    new_leather = jax.lax.select(
        did_kill_cow, state.inventory.leather + 1, state.inventory.leather
    )
    new_inventory = state.inventory.replace(leather=new_leather)

    state = state.replace(cows=new_cows, player_food=new_food, player_hunger=new_hunger, inventory=new_inventory)
    state = state.replace(achievements=new_achievements)

    # Skeleton
    def is_attacking_skeleton_at_index(unused, skeleton_index):
        in_skeleton = (state.skeletons.position[skeleton_index] == block_position).all()
        return None, jnp.logical_and(in_skeleton, state.skeletons.mask[skeleton_index])

    _, is_attacking_skeleton_array = jax.lax.scan(
        is_attacking_skeleton_at_index, None, jnp.arange(static_params.max_skeletons)
    )
    is_attacking_skeleton = is_attacking_skeleton_array.sum() > 0
    target_skeleton_index = jnp.argmax(is_attacking_skeleton_array)
    new_skeletons = state.skeletons

    new_skeleton_health = new_skeletons.health.at[target_skeleton_index].add(
        -get_player_attack_damage(state) * is_attacking_skeleton
    )
    new_skeletons = new_skeletons.replace(health=new_skeleton_health)

    old_mask = new_skeletons.mask[target_skeleton_index]
    new_skeletons = new_skeletons.replace(mask=new_skeletons.health > 0)
    did_kill_skeleton = jnp.logical_and(
        old_mask, jnp.logical_not(new_skeletons.mask[target_skeleton_index])
    )
    new_achievements = state.achievements.at[Achievement.DEFEAT_SKELETON.value].set(
        jnp.logical_or(
            state.achievements[Achievement.DEFEAT_SKELETON.value], did_kill_skeleton
        )
    )

    state = state.replace(skeletons=new_skeletons)
    state = state.replace(achievements=new_achievements)

    did_attack_mob = jnp.logical_or(
        jnp.logical_or(is_attacking_zombie, is_attacking_cow), is_attacking_skeleton
    )

    did_kill_mob = jnp.logical_or(
        jnp.logical_or(did_kill_zombie, did_kill_cow), did_kill_skeleton
    )
    state = state.replace(
        mob_map=state.mob_map.at[block_position[0], block_position[1]].set(
            jnp.logical_and(
                state.mob_map[block_position[0], block_position[1]],
                jnp.logical_not(did_kill_mob),
            )
        )
    )

    # BLOCKS
    # Tree
    can_mine_tree = True
    is_mining_tree = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.TREE.value,
        can_mine_tree,
    )
    mined_tree_block = jax.lax.select(
        is_mining_tree,
        BlockType.GRASS.value,
        state.map[block_position[0], block_position[1]],
    )
    new_map = state.map.at[block_position[0], block_position[1]].set(mined_tree_block)
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood + 1 * is_mining_tree
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_WOOD.value].set(
        jnp.logical_or(new_achievements[Achievement.COLLECT_WOOD.value], is_mining_tree)
    )

    # Stone
    can_mine_stone = state.inventory.wood_pickaxe
    is_mining_stone = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.STONE.value,
        can_mine_stone,
    )
    mined_stone_block = jax.lax.select(
        is_mining_stone,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_stone_block)
    new_inventory = new_inventory.replace(
        stone=state.inventory.stone + 1 * is_mining_stone
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_STONE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_STONE.value], is_mining_stone
        )
    )

    # Coal
    can_mine_coal = state.inventory.wood_pickaxe
    is_mining_coal = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.COAL.value,
        can_mine_coal,
    )
    mined_coal_block = jax.lax.select(
        is_mining_coal,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_coal_block)
    new_inventory = new_inventory.replace(
        coal=state.inventory.coal + 1 * is_mining_coal
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_COAL.value].set(
        jnp.logical_or(new_achievements[Achievement.COLLECT_COAL.value], is_mining_coal)
    )

    # Iron
    can_mine_iron = state.inventory.stone_pickaxe
    is_mining_iron = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.IRON.value,
        can_mine_iron,
    )
    mined_iron_block = jax.lax.select(
        is_mining_iron,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_iron_block)
    new_inventory = new_inventory.replace(
        iron=state.inventory.iron + 1 * is_mining_iron
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_IRON.value].set(
        jnp.logical_or(new_achievements[Achievement.COLLECT_IRON.value], is_mining_iron)
    )

    # Diamond
    can_mine_diamond = state.inventory.iron_pickaxe
    is_mining_diamond = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.DIAMOND.value,
        can_mine_diamond,
    )
    mined_diamond_block = jax.lax.select(
        is_mining_diamond,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_diamond_block)
    new_inventory = new_inventory.replace(
        diamond=state.inventory.diamond + 1 * is_mining_diamond
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_DIAMOND.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_DIAMOND.value], is_mining_diamond
        )
    )

    # NEW FABRAX MINING BLOCKS
    # Copper (requires stone pickaxe)
    can_mine_copper = jnp.logical_or(
        state.inventory.stone_pickaxe > 0,
        jnp.logical_or(state.inventory.iron_pickaxe > 0,
                      jnp.logical_or(state.inventory.bronze_pickaxe > 0, state.inventory.steel_pickaxe > 0))
    )
    is_mining_copper = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.COPPER.value,
        can_mine_copper,
    )
    mined_copper_block = jax.lax.select(
        is_mining_copper,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_copper_block)
    new_inventory = new_inventory.replace(
        copper=state.inventory.copper + 1 * is_mining_copper
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_COPPER.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_COPPER.value], is_mining_copper
        )
    )

    # Tin (requires stone pickaxe)
    can_mine_tin = jnp.logical_or(
        state.inventory.stone_pickaxe > 0,
        jnp.logical_or(state.inventory.iron_pickaxe > 0,
                      jnp.logical_or(state.inventory.bronze_pickaxe > 0, state.inventory.steel_pickaxe > 0))
    )
    is_mining_tin = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.TIN.value,
        can_mine_tin,
    )
    mined_tin_block = jax.lax.select(
        is_mining_tin,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_tin_block)
    new_inventory = new_inventory.replace(
        tin=state.inventory.tin + 1 * is_mining_tin
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_TIN.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_TIN.value], is_mining_tin
        )
    )

    # Clay (bare hands)
    is_mining_clay = (
        state.map[block_position[0], block_position[1]] == BlockType.CLAY.value
    )
    mined_clay_block = jax.lax.select(
        is_mining_clay,
        BlockType.GRASS.value,  # Clay becomes grass when mined
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_clay_block)
    new_inventory = new_inventory.replace(
        clay=state.inventory.clay + 1 * is_mining_clay
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_CLAY.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_CLAY.value], is_mining_clay
        )
    )

    # Limestone (requires wood pickaxe)
    can_mine_limestone = jnp.logical_or(
        state.inventory.wood_pickaxe > 0,
        jnp.logical_or(state.inventory.stone_pickaxe > 0,
                      jnp.logical_or(state.inventory.iron_pickaxe > 0,
                                    jnp.logical_or(state.inventory.bronze_pickaxe > 0, state.inventory.steel_pickaxe > 0)))
    )
    is_mining_limestone = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.LIMESTONE.value,
        can_mine_limestone,
    )
    mined_limestone_block = jax.lax.select(
        is_mining_limestone,
        BlockType.PATH.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(mined_limestone_block)
    new_inventory = new_inventory.replace(
        limestone=state.inventory.limestone + 1 * is_mining_limestone
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_LIMESTONE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_LIMESTONE.value], is_mining_limestone
        )
    )

    # Sand collection (bare hands)
    is_collecting_sand = (
        state.map[block_position[0], block_position[1]] == BlockType.SAND.value
    )
    new_inventory = new_inventory.replace(
        sand=state.inventory.sand + 1 * is_collecting_sand
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_SAND.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_SAND.value], is_collecting_sand
        )
    )

    # Sapling
    rng, _rng = jax.random.split(rng)
    is_mining_sapling = jnp.logical_and(
        state.map[block_position[0], block_position[1]] == BlockType.GRASS.value,
        jax.random.uniform(_rng) < 0.1,
    )

    new_inventory = new_inventory.replace(
        sapling=state.inventory.sapling + 1 * is_mining_sapling
    )
    new_achievements = new_achievements.at[Achievement.COLLECT_SAPLING.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_SAPLING.value], is_mining_sapling
        )
    )

    # Water
    is_drinking_water = (
        state.map[block_position[0], block_position[1]] == BlockType.WATER.value
    )
    new_drink = jax.lax.select(
        is_drinking_water, jnp.minimum(9, state.player_drink + 1), state.player_drink
    )
    new_thirst = jax.lax.select(is_drinking_water, 0.0, state.player_thirst)
    new_achievements = new_achievements.at[Achievement.COLLECT_DRINK.value].set(
        jnp.logical_or(
            new_achievements[Achievement.COLLECT_DRINK.value], is_drinking_water
        )
    )

    # Plant
    is_eating_plant = (
        state.map[block_position[0], block_position[1]] == BlockType.RIPE_PLANT.value
    )
    new_plant = jax.lax.select(
        is_eating_plant,
        BlockType.PLANT.value,
        new_map[block_position[0], block_position[1]],
    )
    new_map = new_map.at[block_position[0], block_position[1]].set(new_plant)
    new_food = jax.lax.select(
        is_eating_plant, jnp.minimum(9, state.player_food + 4), state.player_food
    )
    new_hunger = jax.lax.select(is_eating_plant, 0.0, state.player_hunger)
    new_achievements = new_achievements.at[Achievement.EAT_PLANT.value].set(
        jnp.logical_or(new_achievements[Achievement.EAT_PLANT.value], is_eating_plant)
    )
    new_growing_plants_age = update_plants_with_eat(
        state, block_position, static_params
    )

    # Action mining
    action_block_in_bounds = in_bounds(state, block_position)
    action_block_in_bounds = jnp.logical_and(
        action_block_in_bounds, jnp.logical_not(did_attack_mob)
    )
    new_map = jax.lax.select(action_block_in_bounds, new_map, state.map)
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

    state = state.replace(
        map=new_map,
        inventory=new_inventory,
        player_drink=new_drink,
        player_thirst=new_thirst,
        player_food=new_food,
        player_hunger=new_hunger,
        growing_plants_age=new_growing_plants_age,
        achievements=new_achievements,
    )

    # Do?
    doing_mining = action == Action.DO.value
    state = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(doing_mining, x, y),
        state,
        old_state,
    )

    return state


def is_near_block(state, block_type):
    def _is_given_block(unused, loc_add):
        pos = state.player_position + loc_add
        is_in_bounds = in_bounds(state, pos)
        is_correct_block = state.map[pos[0], pos[1]] == block_type
        return None, jnp.logical_and(is_in_bounds, is_correct_block)

    _, is_block = jax.lax.scan(_is_given_block, None, CLOSE_BLOCKS)

    return is_block.sum() > 0


def do_crafting(state, action):
    is_at_crafting_table = is_near_block(state, BlockType.CRAFTING_TABLE.value)
    is_at_furnace = is_near_block(state, BlockType.FURNACE.value)

    new_achievements = state.achievements

    # Wood pickaxe
    can_craft_wood_pickaxe = state.inventory.wood >= 1

    is_crafting_wood_pickaxe = jnp.logical_and(
        action == Action.MAKE_WOOD_PICKAXE.value,
        jnp.logical_and(can_craft_wood_pickaxe, is_at_crafting_table),
    )

    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 1 * is_crafting_wood_pickaxe,
        wood_pickaxe=state.inventory.wood_pickaxe + 1 * is_crafting_wood_pickaxe,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_WOOD_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_WOOD_PICKAXE.value],
            is_crafting_wood_pickaxe,
        )
    )

    # Stone pickaxe
    can_craft_stone_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.stone >= 1
    )
    is_crafting_stone_pickaxe = jnp.logical_and(
        action == Action.MAKE_STONE_PICKAXE.value,
        jnp.logical_and(can_craft_stone_pickaxe, is_at_crafting_table),
    )

    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_crafting_stone_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_stone_pickaxe,
        stone_pickaxe=new_inventory.stone_pickaxe + 1 * is_crafting_stone_pickaxe,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_STONE_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_STONE_PICKAXE.value],
            is_crafting_stone_pickaxe,
        )
    )

    # Iron pickaxe
    can_craft_iron_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(
            new_inventory.stone >= 1,
            jnp.logical_and(
                new_inventory.iron >= 1,
                new_inventory.coal >= 1,
            ),
        ),
    )
    is_crafting_iron_pickaxe = jnp.logical_and(
        action == Action.MAKE_IRON_PICKAXE.value,
        jnp.logical_and(
            can_craft_iron_pickaxe, jnp.logical_and(is_at_furnace, is_at_crafting_table)
        ),
    )

    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_crafting_iron_pickaxe,
        wood=new_inventory.wood - 1 * is_crafting_iron_pickaxe,
        stone=new_inventory.stone - 1 * is_crafting_iron_pickaxe,
        coal=new_inventory.coal - 1 * is_crafting_iron_pickaxe,
        iron_pickaxe=new_inventory.iron_pickaxe + 1 * is_crafting_iron_pickaxe,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_IRON_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_IRON_PICKAXE.value],
            is_crafting_iron_pickaxe,
        )
    )

    # Wood sword
    can_craft_wood_sword = new_inventory.wood >= 1
    is_crafting_wood_sword = jnp.logical_and(
        action == Action.MAKE_WOOD_SWORD.value,
        jnp.logical_and(can_craft_wood_sword, is_at_crafting_table),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_wood_sword,
        wood_sword=new_inventory.wood_sword + 1 * is_crafting_wood_sword,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_WOOD_SWORD.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_WOOD_SWORD.value], is_crafting_wood_sword
        )
    )

    # Stone sword
    can_craft_stone_sword = jnp.logical_and(
        new_inventory.stone >= 1, new_inventory.wood >= 1
    )
    is_crafting_stone_sword = jnp.logical_and(
        action == Action.MAKE_STONE_SWORD.value,
        jnp.logical_and(can_craft_stone_sword, is_at_crafting_table),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_stone_sword,
        stone=new_inventory.stone - 1 * is_crafting_stone_sword,
        stone_sword=new_inventory.stone_sword + 1 * is_crafting_stone_sword,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_STONE_SWORD.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_STONE_SWORD.value],
            is_crafting_stone_sword,
        )
    )

    # Iron sword
    can_craft_iron_sword = jnp.logical_and(
        new_inventory.iron >= 1,
        jnp.logical_and(
            new_inventory.wood >= 1,
            jnp.logical_and(new_inventory.stone >= 1, new_inventory.coal >= 1),
        ),
    )
    is_crafting_iron_sword = jnp.logical_and(
        action == Action.MAKE_IRON_SWORD.value,
        jnp.logical_and(
            can_craft_iron_sword, jnp.logical_and(is_at_furnace, is_at_crafting_table)
        ),
    )

    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_iron_sword,
        iron=new_inventory.iron - 1 * is_crafting_iron_sword,
        stone=new_inventory.stone - 1 * is_crafting_iron_sword,
        coal=new_inventory.coal - 1 * is_crafting_iron_sword,
        iron_sword=new_inventory.iron_sword + 1 * is_crafting_iron_sword,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_IRON_SWORD.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_IRON_SWORD.value], is_crafting_iron_sword
        )
    )

    # NEW FABRAX CRAFTING SYSTEMS

    # Check for new station adjacencies
    is_at_anvil = is_near_block(state, BlockType.ANVIL.value)
    is_at_kiln = is_near_block(state, BlockType.KILN.value)
    is_at_composter = is_near_block(state, BlockType.COMPOSTER.value)
    is_at_alchemy_bench = is_near_block(state, BlockType.ALCHEMY_BENCH.value)

    # METALLURGY BRANCH

    # Make Bellows (table + wood + leather)
    can_craft_bellows = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.leather >= 1
    )
    is_crafting_bellows = jnp.logical_and(
        action == Action.MAKE_BELLOWS.value,
        jnp.logical_and(can_craft_bellows, is_at_crafting_table),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_crafting_bellows,
        leather=new_inventory.leather - 1 * is_crafting_bellows,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_BELLOWS.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_BELLOWS.value], is_crafting_bellows
        )
    )

    # Smelt Iron Bar (furnace + iron + coal)
    can_smelt_iron_bar = jnp.logical_and(
        new_inventory.iron >= 1, new_inventory.coal >= 1
    )
    is_smelting_iron_bar = jnp.logical_and(
        action == Action.SMELT_IRON_BAR.value,
        jnp.logical_and(can_smelt_iron_bar, is_at_furnace),
    )
    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_smelting_iron_bar,
        coal=new_inventory.coal - 1 * is_smelting_iron_bar,
        iron_bar=new_inventory.iron_bar + 1 * is_smelting_iron_bar,
    )
    new_achievements = new_achievements.at[Achievement.SMELT_IRON_BAR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.SMELT_IRON_BAR.value], is_smelting_iron_bar
        )
    )

    # Make Steel Bar (furnace + iron_bar + coal, bellows optional for speed)
    can_make_steel_bar = jnp.logical_and(
        new_inventory.iron_bar >= 1, new_inventory.coal >= 1
    )
    is_making_steel_bar = jnp.logical_and(
        action == Action.MAKE_STEEL_BAR.value,
        jnp.logical_and(can_make_steel_bar, is_at_furnace),
    )
    new_inventory = new_inventory.replace(
        iron_bar=new_inventory.iron_bar - 1 * is_making_steel_bar,
        coal=new_inventory.coal - 1 * is_making_steel_bar,
        steel_bar=new_inventory.steel_bar + 1 * is_making_steel_bar,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_STEEL_BAR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_STEEL_BAR.value], is_making_steel_bar
        )
    )

    # Make Bronze Bar (furnace + copper + tin + coal)
    can_make_bronze_bar = jnp.logical_and(
        new_inventory.copper >= 1,
        jnp.logical_and(new_inventory.tin >= 1, new_inventory.coal >= 1)
    )
    is_making_bronze_bar = jnp.logical_and(
        action == Action.MAKE_BRONZE_BAR.value,
        jnp.logical_and(can_make_bronze_bar, is_at_furnace),
    )
    new_inventory = new_inventory.replace(
        copper=new_inventory.copper - 1 * is_making_bronze_bar,
        tin=new_inventory.tin - 1 * is_making_bronze_bar,
        coal=new_inventory.coal - 1 * is_making_bronze_bar,
        bronze_bar=new_inventory.bronze_bar + 1 * is_making_bronze_bar,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_BRONZE_BAR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_BRONZE_BAR.value], is_making_bronze_bar
        )
    )

    # Forge Steel Pickaxe (anvil + table + wood + stone + steel_bar)
    can_forge_steel_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(new_inventory.stone >= 1, new_inventory.steel_bar >= 1)
    )
    is_forging_steel_pickaxe = jnp.logical_and(
        action == Action.FORGE_STEEL_PICKAXE.value,
        jnp.logical_and(can_forge_steel_pickaxe, jnp.logical_and(is_at_anvil, is_at_crafting_table)),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_forging_steel_pickaxe,
        stone=new_inventory.stone - 1 * is_forging_steel_pickaxe,
        steel_bar=new_inventory.steel_bar - 1 * is_forging_steel_pickaxe,
        steel_pickaxe=new_inventory.steel_pickaxe + 1 * is_forging_steel_pickaxe,
    )
    new_achievements = new_achievements.at[Achievement.FORGE_STEEL_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.FORGE_STEEL_PICKAXE.value], is_forging_steel_pickaxe
        )
    )

    # Forge Bronze Pickaxe (anvil + table + wood + stone + bronze_bar)
    can_forge_bronze_pickaxe = jnp.logical_and(
        new_inventory.wood >= 1,
        jnp.logical_and(new_inventory.stone >= 1, new_inventory.bronze_bar >= 1)
    )
    is_forging_bronze_pickaxe = jnp.logical_and(
        action == Action.FORGE_BRONZE_PICKAXE.value,
        jnp.logical_and(can_forge_bronze_pickaxe, jnp.logical_and(is_at_anvil, is_at_crafting_table)),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_forging_bronze_pickaxe,
        stone=new_inventory.stone - 1 * is_forging_bronze_pickaxe,
        bronze_bar=new_inventory.bronze_bar - 1 * is_forging_bronze_pickaxe,
        bronze_pickaxe=new_inventory.bronze_pickaxe + 1 * is_forging_bronze_pickaxe,
    )
    new_achievements = new_achievements.at[Achievement.FORGE_BRONZE_PICKAXE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.FORGE_BRONZE_PICKAXE.value], is_forging_bronze_pickaxe
        )
    )

    # Forge Steel Sword (anvil + table + wood + steel_bar)
    can_forge_steel_sword = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.steel_bar >= 1
    )
    is_forging_steel_sword = jnp.logical_and(
        action == Action.FORGE_STEEL_SWORD.value,
        jnp.logical_and(can_forge_steel_sword, jnp.logical_and(is_at_anvil, is_at_crafting_table)),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_forging_steel_sword,
        steel_bar=new_inventory.steel_bar - 1 * is_forging_steel_sword,
        steel_sword=new_inventory.steel_sword + 1 * is_forging_steel_sword,
    )
    new_achievements = new_achievements.at[Achievement.FORGE_STEEL_SWORD.value].set(
        jnp.logical_or(
            new_achievements[Achievement.FORGE_STEEL_SWORD.value], is_forging_steel_sword
        )
    )

    # Forge Bronze Sword (anvil + table + wood + bronze_bar)
    can_forge_bronze_sword = jnp.logical_and(
        new_inventory.wood >= 1, new_inventory.bronze_bar >= 1
    )
    is_forging_bronze_sword = jnp.logical_and(
        action == Action.FORGE_BRONZE_SWORD.value,
        jnp.logical_and(can_forge_bronze_sword, jnp.logical_and(is_at_anvil, is_at_crafting_table)),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_forging_bronze_sword,
        bronze_bar=new_inventory.bronze_bar - 1 * is_forging_bronze_sword,
        bronze_sword=new_inventory.bronze_sword + 1 * is_forging_bronze_sword,
    )
    new_achievements = new_achievements.at[Achievement.FORGE_BRONZE_SWORD.value].set(
        jnp.logical_or(
            new_achievements[Achievement.FORGE_BRONZE_SWORD.value], is_forging_bronze_sword
        )
    )

    # Harden Edge (anvil + tool hardening - requires steel or bronze pickaxe)
    can_harden_edge = jnp.logical_or(
        new_inventory.steel_pickaxe >= 1, new_inventory.bronze_pickaxe >= 1
    )
    is_hardening_edge = jnp.logical_and(
        action == Action.HARDEN_EDGE.value,
        jnp.logical_and(can_harden_edge, is_at_anvil),
    )
    # This action sets a "buff" rather than consuming/producing items
    # For now, we'll just track the achievement since the tool_hardened system needs state tracking
    new_achievements = new_achievements.at[Achievement.HARDEN_EDGE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.HARDEN_EDGE.value], is_hardening_edge
        )
    )

    # GLASS & CERAMICS BRANCH

    # Smelt Glass (kiln + sand + coal)
    can_smelt_glass = jnp.logical_and(
        new_inventory.sand >= 1, new_inventory.coal >= 1
    )
    is_smelting_glass = jnp.logical_and(
        action == Action.SMELT_GLASS.value,
        jnp.logical_and(can_smelt_glass, is_at_kiln),
    )
    new_inventory = new_inventory.replace(
        sand=new_inventory.sand - 1 * is_smelting_glass,
        coal=new_inventory.coal - 1 * is_smelting_glass,
        glass=new_inventory.glass + 1 * is_smelting_glass,
    )
    new_achievements = new_achievements.at[Achievement.SMELT_GLASS.value].set(
        jnp.logical_or(
            new_achievements[Achievement.SMELT_GLASS.value], is_smelting_glass
        )
    )

    # Fire Brick (kiln + clay + coal)
    can_fire_brick = jnp.logical_and(
        new_inventory.clay >= 1, new_inventory.coal >= 1
    )
    is_firing_brick = jnp.logical_and(
        action == Action.FIRE_BRICK.value,
        jnp.logical_and(can_fire_brick, is_at_kiln),
    )
    new_inventory = new_inventory.replace(
        clay=new_inventory.clay - 1 * is_firing_brick,
        coal=new_inventory.coal - 1 * is_firing_brick,
        brick=new_inventory.brick + 1 * is_firing_brick,
    )
    new_achievements = new_achievements.at[Achievement.FIRE_BRICK.value].set(
        jnp.logical_or(
            new_achievements[Achievement.FIRE_BRICK.value], is_firing_brick
        )
    )

    # Make Glass Bottle (kiln + glass)
    can_make_bottle = new_inventory.glass >= 1
    is_making_bottle = jnp.logical_and(
        action == Action.MAKE_GLASS_BOTTLE.value,
        jnp.logical_and(can_make_bottle, is_at_kiln),
    )
    new_inventory = new_inventory.replace(
        glass=new_inventory.glass - 1 * is_making_bottle,
        bottle=new_inventory.bottle + 1 * is_making_bottle,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_GLASS_BOTTLE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_GLASS_BOTTLE.value], is_making_bottle
        )
    )

    # Make Lens (kiln + table + glass)
    can_make_lens = new_inventory.glass >= 1
    is_making_lens = jnp.logical_and(
        action == Action.MAKE_LENS.value,
        jnp.logical_and(can_make_lens, jnp.logical_and(is_at_kiln, is_at_crafting_table)),
    )
    new_inventory = new_inventory.replace(
        glass=new_inventory.glass - 1 * is_making_lens,
        lens=new_inventory.lens + 1 * is_making_lens,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_LENS.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_LENS.value], is_making_lens
        )
    )

    # Make Telescope (table + lens + wood)
    can_make_telescope = jnp.logical_and(
        new_inventory.lens >= 1, new_inventory.wood >= 1
    )
    is_making_telescope = jnp.logical_and(
        action == Action.MAKE_TELESCOPE.value,
        jnp.logical_and(can_make_telescope, is_at_crafting_table),
    )
    new_inventory = new_inventory.replace(
        lens=new_inventory.lens - 1 * is_making_telescope,
        wood=new_inventory.wood - 1 * is_making_telescope,
        telescope=new_inventory.telescope + 1 * is_making_telescope,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_TELESCOPE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_TELESCOPE.value], is_making_telescope
        )
    )

    # Make Lime (kiln + limestone + coal)
    can_make_lime = jnp.logical_and(
        new_inventory.limestone >= 1, new_inventory.coal >= 1
    )
    is_making_lime = jnp.logical_and(
        action == Action.MAKE_LIME.value,
        jnp.logical_and(can_make_lime, is_at_kiln),
    )
    new_inventory = new_inventory.replace(
        limestone=new_inventory.limestone - 1 * is_making_lime,
        coal=new_inventory.coal - 1 * is_making_lime,
        lime=new_inventory.lime + 1 * is_making_lime,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_LIME.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_LIME.value], is_making_lime
        )
    )

    # Make Mortar (table + lime + sand)
    can_make_mortar = jnp.logical_and(
        new_inventory.lime >= 1, new_inventory.sand >= 1
    )
    is_making_mortar = jnp.logical_and(
        action == Action.MAKE_MORTAR.value,
        jnp.logical_and(can_make_mortar, is_at_crafting_table),
    )
    new_inventory = new_inventory.replace(
        lime=new_inventory.lime - 1 * is_making_mortar,
        sand=new_inventory.sand - 1 * is_making_mortar,
        mortar=new_inventory.mortar + 1 * is_making_mortar,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_MORTAR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_MORTAR.value], is_making_mortar
        )
    )

    # CHEMISTRY & ALCHEMY BRANCH

    # Make Tar (kiln + wood)
    can_make_tar = new_inventory.wood >= 1
    is_making_tar = jnp.logical_and(
        action == Action.MAKE_TAR.value,
        jnp.logical_and(can_make_tar, is_at_kiln),
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_making_tar,
        tar=new_inventory.tar + 1 * is_making_tar,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_TAR.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_TAR.value], is_making_tar
        )
    )

    # Make Fertilizer (composter + plant + tar)
    # Note: This would require plant item - for now simplified to available plant food
    can_make_fertilizer = jnp.logical_and(
        new_inventory.tar >= 1, state.player_food >= 1  # Using food as plant proxy
    )
    is_making_fertilizer = jnp.logical_and(
        action == Action.MAKE_FERTILIZER.value,
        jnp.logical_and(can_make_fertilizer, is_at_composter),
    )
    new_food_after_fertilizer = state.player_food - 1 * is_making_fertilizer
    new_inventory = new_inventory.replace(
        tar=new_inventory.tar - 1 * is_making_fertilizer,
        fertilizer=new_inventory.fertilizer + 1 * is_making_fertilizer,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_FERTILIZER.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_FERTILIZER.value], is_making_fertilizer
        )
    )

    # Brew Tonic (alchemy_bench + bottle + plant food)
    can_brew_tonic = jnp.logical_and(
        new_inventory.bottle >= 1, state.player_food >= 1
    )
    is_brewing_tonic = jnp.logical_and(
        action == Action.BREW_TONIC.value,
        jnp.logical_and(can_brew_tonic, is_at_alchemy_bench),
    )
    new_food_after_tonic = new_food_after_fertilizer - 1 * is_brewing_tonic
    new_inventory = new_inventory.replace(
        bottle=new_inventory.bottle - 1 * is_brewing_tonic,
        tonic_basic=new_inventory.tonic_basic + 1 * is_brewing_tonic,
    )
    new_achievements = new_achievements.at[Achievement.BREW_TONIC.value].set(
        jnp.logical_or(
            new_achievements[Achievement.BREW_TONIC.value], is_brewing_tonic
        )
    )

    # Brew Stoneskin (alchemy_bench + bottle + tar)
    can_brew_stoneskin = jnp.logical_and(
        new_inventory.bottle >= 1, new_inventory.tar >= 1
    )
    is_brewing_stoneskin = jnp.logical_and(
        action == Action.BREW_STONE_SKIN.value,
        jnp.logical_and(can_brew_stoneskin, is_at_alchemy_bench),
    )
    new_inventory = new_inventory.replace(
        bottle=new_inventory.bottle - 1 * is_brewing_stoneskin,
        tar=new_inventory.tar - 1 * is_brewing_stoneskin,
        tonic_stoneskin=new_inventory.tonic_stoneskin + 1 * is_brewing_stoneskin,
    )
    new_achievements = new_achievements.at[Achievement.BREW_STONE_SKIN.value].set(
        jnp.logical_or(
            new_achievements[Achievement.BREW_STONE_SKIN.value], is_brewing_stoneskin
        )
    )

    # Make Flux (alchemy_bench + lime + tar)
    can_make_flux = jnp.logical_and(
        new_inventory.lime >= 1, new_inventory.tar >= 1
    )
    is_making_flux = jnp.logical_and(
        action == Action.MAKE_FLUX.value,
        jnp.logical_and(can_make_flux, is_at_alchemy_bench),
    )
    new_inventory = new_inventory.replace(
        lime=new_inventory.lime - 1 * is_making_flux,
        tar=new_inventory.tar - 1 * is_making_flux,
        flux=new_inventory.flux + 1 * is_making_flux,
    )
    new_achievements = new_achievements.at[Achievement.MAKE_FLUX.value].set(
        jnp.logical_or(
            new_achievements[Achievement.MAKE_FLUX.value], is_making_flux
        )
    )

    state = state.replace(
        inventory=new_inventory,
        achievements=new_achievements,
        player_food=new_food_after_tonic,  # Update food if consumed in crafting
    )

    return state


def add_new_growing_plant(state, position, is_placing_sapling, static_params):
    def _is_empty(unused, index):
        return None, jnp.logical_not(state.growing_plants_mask[index])

    _, is_empty = jax.lax.scan(
        _is_empty, None, jnp.arange(static_params.max_growing_plants)
    )

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


def calculate_light_level(timestep, params):
    progress = (timestep / params.day_length) % 1 + 0.3
    return 1 - jnp.abs(jnp.cos(jnp.pi * progress)) ** 3


def place_block(state, action, static_params):
    placing_block_position = state.player_position + DIRECTIONS[state.player_direction]

    # Crafting table
    crafting_table_key_down = action == Action.PLACE_TABLE.value
    has_wood = state.inventory.wood >= 2
    is_placing_crafting_table = jnp.logical_and(
        crafting_table_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_wood
        ),
    )
    placed_crafting_table_block = jax.lax.select(
        is_placing_crafting_table,
        BlockType.CRAFTING_TABLE.value,
        state.map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = state.map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_crafting_table_block
    )
    new_inventory = state.inventory.replace(
        wood=state.inventory.wood - 2 * is_placing_crafting_table
    )
    new_achievements = state.achievements.at[Achievement.PLACE_TABLE.value].set(
        jnp.logical_or(
            state.achievements[Achievement.PLACE_TABLE.value], is_placing_crafting_table
        )
    )

    # Furnace
    furnace_key_down = action == Action.PLACE_FURNACE.value
    has_stone = new_inventory.stone > 0
    is_placing_furnace = jnp.logical_and(
        furnace_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_stone
        ),
    )
    placed_furnace_block = jax.lax.select(
        is_placing_furnace,
        BlockType.FURNACE.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_furnace_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_furnace
    )
    new_achievements = new_achievements.at[Achievement.PLACE_FURNACE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_FURNACE.value], is_placing_furnace
        )
    )

    # Stone
    stone_key_down = action == Action.PLACE_STONE.value
    has_stone = new_inventory.stone > 0
    is_placing_on_valid_block = jnp.logical_or(
        state.map[placing_block_position[0], placing_block_position[1]]
        == BlockType.WATER.value,
        jnp.logical_not(is_in_wall(state, placing_block_position)),
    )
    is_placing_stone = jnp.logical_and(
        stone_key_down,
        jnp.logical_and(is_placing_on_valid_block, has_stone),
    )
    placed_stone_block = jax.lax.select(
        is_placing_stone,
        BlockType.STONE.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_stone_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_stone
    )
    new_achievements = new_achievements.at[Achievement.PLACE_STONE.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_STONE.value], is_placing_stone
        )
    )

    # Plant
    sapling_key_down = action == Action.PLACE_PLANT.value
    has_sapling = new_inventory.sapling > 0
    is_placing_sapling = jnp.logical_and(
        sapling_key_down,
        jnp.logical_and(
            new_map[placing_block_position[0], placing_block_position[1]]
            == BlockType.GRASS.value,
            has_sapling,
        ),
    )
    placed_sapling_block = jax.lax.select(
        is_placing_sapling,
        BlockType.PLANT.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_sapling_block
    )
    new_inventory = new_inventory.replace(
        sapling=new_inventory.sapling - 1 * is_placing_sapling
    )
    new_achievements = new_achievements.at[Achievement.PLACE_PLANT.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_PLANT.value], is_placing_sapling
        )
    )
    (
        new_growing_plants_positions,
        new_growing_plants_age,
        new_growing_plants_mask,
    ) = add_new_growing_plant(
        state, placing_block_position, is_placing_sapling, static_params
    )

    # NEW FABRAX STATION PLACEMENT

    # Place Anvil (requires 1 iron + 2 stone)
    anvil_key_down = action == Action.PLACE_ANVIL.value
    has_materials_for_anvil = jnp.logical_and(
        new_inventory.iron >= 1, new_inventory.stone >= 2
    )
    is_placing_anvil = jnp.logical_and(
        anvil_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_materials_for_anvil
        ),
    )
    placed_anvil_block = jax.lax.select(
        is_placing_anvil,
        BlockType.ANVIL.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_anvil_block
    )
    new_inventory = new_inventory.replace(
        iron=new_inventory.iron - 1 * is_placing_anvil,
        stone=new_inventory.stone - 2 * is_placing_anvil
    )
    new_achievements = new_achievements.at[Achievement.PLACE_ANVIL.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_ANVIL.value], is_placing_anvil
        )
    )

    # Place Kiln
    kiln_key_down = action == Action.PLACE_KILN.value
    has_stone_for_kiln = new_inventory.stone >= 1
    is_placing_kiln = jnp.logical_and(
        kiln_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_stone_for_kiln
        ),
    )
    placed_kiln_block = jax.lax.select(
        is_placing_kiln,
        BlockType.KILN.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_kiln_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_kiln
    )
    new_achievements = new_achievements.at[Achievement.PLACE_KILN.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_KILN.value], is_placing_kiln
        )
    )

    # Place Composter
    composter_key_down = action == Action.PLACE_COMPOSTER.value
    has_wood_for_composter = new_inventory.wood >= 1
    is_placing_composter = jnp.logical_and(
        composter_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_wood_for_composter
        ),
    )
    placed_composter_block = jax.lax.select(
        is_placing_composter,
        BlockType.COMPOSTER.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_composter_block
    )
    new_inventory = new_inventory.replace(
        wood=new_inventory.wood - 1 * is_placing_composter
    )
    new_achievements = new_achievements.at[Achievement.PLACE_COMPOSTER.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_COMPOSTER.value], is_placing_composter
        )
    )

    # Place Alchemy Bench
    alchemy_bench_key_down = action == Action.PLACE_ALCHEMY_BENCH.value
    has_stone_for_alchemy = new_inventory.stone >= 1
    is_placing_alchemy_bench = jnp.logical_and(
        alchemy_bench_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_stone_for_alchemy
        ),
    )
    placed_alchemy_bench_block = jax.lax.select(
        is_placing_alchemy_bench,
        BlockType.ALCHEMY_BENCH.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_alchemy_bench_block
    )
    new_inventory = new_inventory.replace(
        stone=new_inventory.stone - 1 * is_placing_alchemy_bench
    )
    new_achievements = new_achievements.at[Achievement.PLACE_ALCHEMY_BENCH.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_ALCHEMY_BENCH.value], is_placing_alchemy_bench
        )
    )

    # Place Window (requires glass)
    window_key_down = action == Action.PLACE_WINDOW.value
    has_glass_for_window = new_inventory.glass >= 1
    is_placing_window = jnp.logical_and(
        window_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_glass_for_window
        ),
    )
    placed_window_block = jax.lax.select(
        is_placing_window,
        BlockType.WINDOW.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_window_block
    )
    new_inventory = new_inventory.replace(
        glass=new_inventory.glass - 1 * is_placing_window
    )
    new_achievements = new_achievements.at[Achievement.PLACE_WINDOW.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_WINDOW.value], is_placing_window
        )
    )

    # Place Wall Masonry (requires mortar)
    wall_masonry_key_down = action == Action.PLACE_WALL_MASONRY.value
    has_mortar_for_wall = new_inventory.mortar >= 1
    is_placing_wall_masonry = jnp.logical_and(
        wall_masonry_key_down,
        jnp.logical_and(
            jnp.logical_not(is_in_wall(state, placing_block_position)), has_mortar_for_wall
        ),
    )
    placed_wall_masonry_block = jax.lax.select(
        is_placing_wall_masonry,
        BlockType.WALL_MASONRY.value,
        new_map[placing_block_position[0], placing_block_position[1]],
    )
    new_map = new_map.at[placing_block_position[0], placing_block_position[1]].set(
        placed_wall_masonry_block
    )
    new_inventory = new_inventory.replace(
        mortar=new_inventory.mortar - 1 * is_placing_wall_masonry
    )
    new_achievements = new_achievements.at[Achievement.PLACE_WALL_MASONRY.value].set(
        jnp.logical_or(
            new_achievements[Achievement.PLACE_WALL_MASONRY.value], is_placing_wall_masonry
        )
    )

    # Do?

    action_block = state.player_position + DIRECTIONS[state.player_direction]
    action_block_in_bounds = in_bounds(state, action_block)
    action_block_in_bounds = jnp.logical_and(
        action_block_in_bounds, jnp.logical_not(is_in_mob(state, action_block))
    )

    new_map = jax.lax.select(action_block_in_bounds, new_map, state.map)
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

    state = state.replace(
        map=new_map,
        inventory=new_inventory,
        achievements=new_achievements,
        growing_plants_positions=new_growing_plants_positions,
        growing_plants_age=new_growing_plants_age,
        growing_plants_mask=new_growing_plants_mask,
    )

    return state


def is_in_mob(state: EnvState, position: chex.Array):
    return jnp.logical_or(
        state.mob_map[position[0], position[1]],
        (state.player_position == position).all(),
    )


def update_mobs(rng, state, params, static_params):
    # Move zombies

    def _move_zombie(rng_and_state, zombie_index):
        rng, state = rng_and_state
        zombies = state.zombies

        # Random move
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
        )
        random_move_proposed_position = (
            zombies.position[zombie_index] + random_move_direction
        )

        # Move towards player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(
            state.player_position - zombies.position[zombie_index]
        )
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_move_direction_index]
                - zombies.position[zombie_index][player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_proposed_position = (
            zombies.position[zombie_index] + player_move_direction
        )

        # Choose movement
        close_to_player = (
            jnp.sum(jnp.abs(zombies.position[zombie_index] - state.player_position))
            < 10
        )
        rng, _rng = jax.random.split(rng)
        close_to_player = jnp.logical_and(
            close_to_player, jax.random.uniform(_rng) < 0.75
        )

        proposed_position = jax.lax.select(
            close_to_player,
            player_move_proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = (
            jnp.sum(jnp.abs(zombies.position[zombie_index] - state.player_position))
            == 1
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, zombies.attack_cooldown[zombie_index] <= 0
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, zombies.mask[zombie_index]
        )

        proposed_position = jax.lax.select(
            is_attacking_player, zombies.position[zombie_index], proposed_position
        )

        zombie_damage = jax.lax.select(
            state.is_sleeping,
            7,
            2,
        )
        new_cooldown = jax.lax.select(
            is_attacking_player, 5, zombies.attack_cooldown[zombie_index] - 1
        )

        is_waking_player = jnp.logical_and(state.is_sleeping, is_attacking_player)

        state = state.replace(
            player_health=state.player_health - zombie_damage * is_attacking_player,
            is_sleeping=jnp.logical_and(
                state.is_sleeping, jnp.logical_not(is_attacking_player)
            ),
            achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
                jnp.logical_or(
                    state.achievements[Achievement.WAKE_UP.value], is_waking_player
                )
            ),
        )

        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        )
        position = jax.lax.select(
            valid_move, proposed_position, zombies.position[zombie_index]
        )

        should_not_despawn = (
            jnp.abs(zombies.position[zombie_index] - state.player_position).sum()
            < params.mob_despawn_distance
        )

        rng, _rng = jax.random.split(rng)

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.zombies.position[zombie_index, 0],
            state.zombies.position[zombie_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.zombies.position[zombie_index, 0],
                    state.zombies.position[zombie_index, 1],
                ],
                jnp.logical_not(zombies.mask[zombie_index]),
            )
        )
        new_mask = jnp.logical_and(state.zombies.mask[zombie_index], should_not_despawn)
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[position[0], position[1]].set(
            jnp.logical_or(new_mob_map[position[0], position[1]], new_mask)
        )

        state = state.replace(
            zombies=state.zombies.replace(
                position=state.zombies.position.at[zombie_index].set(position),
                attack_cooldown=state.zombies.attack_cooldown.at[zombie_index].set(
                    new_cooldown
                ),
                mask=state.zombies.mask.at[zombie_index].set(new_mask),
            ),
            mob_map=new_mob_map,
        )

        return (_rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, new_state), _ = jax.lax.scan(
        _move_zombie, (rng, state), jnp.arange(static_params.max_zombies)
    )

    # Move cows
    def _move_cow(rng_and_state, cow_index):
        rng, state = rng_and_state
        cows = state.cows

        # Random move
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:9],  # 50% chance of not moving
        )
        proposed_position = cows.position[cow_index] + random_move_direction

        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        )
        position = jax.lax.select(
            valid_move, proposed_position, cows.position[cow_index]
        )

        should_not_despawn = (
            jnp.abs(cows.position[cow_index] - state.player_position).sum()
            < params.mob_despawn_distance
        )

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.cows.position[cow_index, 0], state.cows.position[cow_index, 1]
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.cows.position[cow_index, 0], state.cows.position[cow_index, 1]
                ],
                jnp.logical_not(cows.mask[cow_index]),
            )
        )
        new_mask = jnp.logical_and(state.cows.mask[cow_index], should_not_despawn)
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[position[0], position[1]].set(
            jnp.logical_or(new_mob_map[position[0], position[1]], new_mask)
        )

        state = state.replace(
            cows=state.cows.replace(
                position=state.cows.position.at[cow_index].set(position),
                mask=state.cows.mask.at[cow_index].set(
                    jnp.logical_and(state.cows.mask[cow_index], should_not_despawn)
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, new_state), _ = jax.lax.scan(
        _move_cow, (rng, new_state), jnp.arange(static_params.max_cows)
    )

    # Move skeletons

    def _move_skeleton(rng_and_state, skeleton_index):
        rng, state = rng_and_state
        skeletons = state.skeletons

        # Random move
        rng, _rng = jax.random.split(rng)
        random_move_direction = jax.random.choice(
            _rng,
            DIRECTIONS[1:5],
        )
        random_move_proposed_position = (
            skeletons.position[skeleton_index] + random_move_direction
        )

        # Move towards player
        player_move_direction = jnp.zeros((2,), dtype=jnp.int32)
        player_move_direction_abs = jnp.abs(
            state.player_position - skeletons.position[skeleton_index]
        )
        player_move_direction_index_p = (
            player_move_direction_abs == player_move_direction_abs.max()
        ) / player_move_direction_abs.sum()
        rng, _rng = jax.random.split(rng)
        player_move_direction_index = jax.random.choice(
            _rng,
            jnp.arange(2),
            p=player_move_direction_index_p,
        )

        player_move_direction = player_move_direction.at[
            player_move_direction_index
        ].set(
            jnp.sign(
                state.player_position[player_move_direction_index]
                - skeletons.position[skeleton_index][player_move_direction_index]
            ).astype(jnp.int32)
        )
        player_move_towards_proposed_position = (
            skeletons.position[skeleton_index] + player_move_direction
        )
        player_move_away_proposed_position = (
            skeletons.position[skeleton_index] - player_move_direction
        )

        # Choose movement
        distance_to_player = jnp.sum(
            jnp.abs(skeletons.position[skeleton_index] - state.player_position)
        )

        far_from_player = distance_to_player >= 10
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

        rng, _rng = jax.random.split(rng)

        proposed_position = jax.lax.select(
            jax.random.uniform(_rng) > 0.85,
            proposed_position,
            random_move_proposed_position,
        )

        # Choose attack or not
        is_attacking_player = jnp.logical_and(
            distance_to_player >= 4, distance_to_player <= 5
        )
        # If we're too close to player (so we want to run) but are blocked, we shoot
        is_attacking_player = jnp.logical_or(
            is_attacking_player,
            jnp.logical_and(
                too_close_to_player,
                jnp.logical_not(
                    is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
                        state, proposed_position
                    )
                ),
            ),
        )

        is_attacking_player = jnp.logical_and(
            is_attacking_player, skeletons.attack_cooldown[skeleton_index] <= 0
        )
        is_attacking_player = jnp.logical_and(
            is_attacking_player, skeletons.mask[skeleton_index]
        )

        # Spawn arrow
        can_spawn_arrow = state.arrows.mask.sum() < static_params.max_arrows
        new_arrow_position = skeletons.position[skeleton_index]

        is_spawning_arrow = jnp.logical_and(is_attacking_player, can_spawn_arrow)

        new_arrow_index = jnp.argmax(jnp.logical_not(state.arrows.mask))
        new_arrow_position = jax.lax.select(
            is_spawning_arrow,
            new_arrow_position,
            state.arrows.position[new_arrow_index],
        )
        new_arrow_mask = jax.lax.select(
            is_spawning_arrow, True, state.arrows.mask[new_arrow_index]
        )
        new_arrow_direction = jax.lax.select(
            is_spawning_arrow,
            player_move_direction,
            state.arrow_directions[new_arrow_index],
        )

        new_arrows = state.arrows.replace(
            position=state.arrows.position.at[new_arrow_index].set(new_arrow_position),
            mask=state.arrows.mask.at[new_arrow_index].set(new_arrow_mask),
        )
        state = state.replace(
            arrows=new_arrows,
            arrow_directions=state.arrow_directions.at[new_arrow_index].set(
                new_arrow_direction
            ),
        )

        proposed_position = jax.lax.select(
            is_attacking_player, skeletons.position[skeleton_index], proposed_position
        )

        new_cooldown = jax.lax.select(
            is_attacking_player, 4, skeletons.attack_cooldown[skeleton_index] - 1
        )

        valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
            state, proposed_position
        )

        position = jax.lax.select(
            valid_move, proposed_position, skeletons.position[skeleton_index]
        )

        should_not_despawn = (
            jnp.abs(skeletons.position[skeleton_index] - state.player_position).sum()
            < params.mob_despawn_distance
        )

        # Clear our old entry if we are alive
        new_mob_map = state.mob_map.at[
            state.skeletons.position[skeleton_index, 0],
            state.skeletons.position[skeleton_index, 1],
        ].set(
            jnp.logical_and(
                state.mob_map[
                    state.skeletons.position[skeleton_index, 0],
                    state.skeletons.position[skeleton_index, 1],
                ],
                jnp.logical_not(skeletons.mask[skeleton_index]),
            )
        )
        new_mask = jnp.logical_and(
            state.skeletons.mask[skeleton_index], should_not_despawn
        )
        # Enter new entry if we are alive and not despawning this timestep
        new_mob_map = new_mob_map.at[position[0], position[1]].set(
            jnp.logical_or(new_mob_map[position[0], position[1]], new_mask)
        )

        state = state.replace(
            skeletons=state.skeletons.replace(
                position=state.skeletons.position.at[skeleton_index].set(position),
                attack_cooldown=state.skeletons.attack_cooldown.at[skeleton_index].set(
                    new_cooldown
                ),
                mask=state.skeletons.mask.at[skeleton_index].set(
                    jnp.logical_and(
                        state.skeletons.mask[skeleton_index], should_not_despawn
                    )
                ),
            ),
            mob_map=new_mob_map,
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, new_state), _ = jax.lax.scan(
        _move_skeleton, (rng, new_state), jnp.arange(static_params.max_skeletons)
    )

    # Move arrows
    def _move_arrow(rng_and_state, arrow_index):
        rng, state = rng_and_state
        arrows = state.arrows

        proposed_position = (
            arrows.position[arrow_index] + state.arrow_directions[arrow_index]
        )

        proposed_position_in_player = (proposed_position == state.player_position).all()

        proposed_position_in_bounds = in_bounds(state, proposed_position)
        in_wall = is_in_wall(state, proposed_position)
        in_wall = jnp.logical_and(
            in_wall,
            jnp.logical_not(
                state.map[proposed_position[0], proposed_position[1]]
                == BlockType.WATER.value
            ),
        )  # Arrows can go over water
        in_mob = is_in_mob(state, proposed_position)

        continue_move = jnp.logical_and(
            proposed_position_in_bounds, jnp.logical_not(in_wall)
        )
        continue_move = jnp.logical_and(continue_move, jnp.logical_not(in_mob))
        position = proposed_position

        # Clear our old entry if we are alive
        new_mask = jnp.logical_and(continue_move, arrows.mask[arrow_index])

        hit_player = jnp.logical_and(
            proposed_position_in_player, arrows.mask[arrow_index]
        )

        hit_bench_or_furnace = jnp.logical_or(
            state.map[position[0], position[1]] == BlockType.FURNACE.value,
            state.map[position[0], position[1]] == BlockType.CRAFTING_TABLE.value,
        )
        removing_block = jnp.logical_and(hit_bench_or_furnace, arrows.mask[arrow_index])

        new_block = jax.lax.select(
            removing_block,
            BlockType.PATH.value,
            state.map[position[0], position[1]],
        )

        state = state.replace(
            arrows=state.arrows.replace(
                position=state.arrows.position.at[arrow_index].set(position),
                mask=state.arrows.mask.at[arrow_index].set(new_mask),
            ),
            player_health=state.player_health - 2 * hit_player,
            is_sleeping=jnp.logical_and(state.is_sleeping, jnp.logical_not(hit_player)),
            map=state.map.at[position[0], position[1]].set(new_block),
        )

        return (rng, state), None

    rng, _rng = jax.random.split(rng)
    (rng, new_state), _ = jax.lax.scan(
        _move_arrow, (rng, new_state), jnp.arange(static_params.max_arrows)
    )

    return new_state


def get_distance_map(position, static_params):
    dist_x = jnp.abs(jnp.arange(0, static_params.map_size[0]) - position[0])
    dist_x = jnp.expand_dims(dist_x, axis=1)
    dist_x = jnp.tile(dist_x, (1, static_params.map_size[1]))

    dist_y = jnp.abs(jnp.arange(0, static_params.map_size[1]) - position[1])
    dist_y = jnp.expand_dims(dist_y, axis=0)
    dist_y = jnp.tile(dist_y, (static_params.map_size[0], 1))

    dist = dist_x + dist_y
    return dist


def update_player_intrinsics(state, action):
    # Start sleeping?
    is_starting_sleep = jnp.logical_and(
        action == Action.SLEEP.value, state.player_energy < 9
    )
    new_is_sleeping = jnp.logical_or(state.is_sleeping, is_starting_sleep)
    state = state.replace(is_sleeping=new_is_sleeping)

    # Wake up?
    is_waking_up = jnp.logical_and(state.player_energy >= 9, state.is_sleeping)
    new_is_sleeping = jnp.logical_and(state.is_sleeping, jnp.logical_not(is_waking_up))
    state = state.replace(
        is_sleeping=new_is_sleeping,
        achievements=state.achievements.at[Achievement.WAKE_UP.value].set(
            jnp.logical_or(state.achievements[Achievement.WAKE_UP.value], is_waking_up)
        ),
    )

    # Hunger
    hunger_add = jax.lax.select(state.is_sleeping, 0.5, 1.0)
    new_hunger = state.player_hunger + hunger_add

    hungered_food = jnp.maximum(state.player_food - 1, 0)
    new_food = jax.lax.select(new_hunger > 25, hungered_food, state.player_food)
    new_hunger = jax.lax.select(new_hunger > 25, 0.0, new_hunger)

    state = state.replace(
        player_hunger=new_hunger,
        player_food=new_food,
    )

    # Thirst
    thirst_add = jax.lax.select(state.is_sleeping, 0.5, 1.0)
    new_thirst = state.player_thirst + thirst_add
    thirsted_drink = jnp.maximum(state.player_drink - 1, 0)
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
        state.player_fatigue + 1,
    )

    new_energy = jax.lax.select(
        new_fatigue > 30, jnp.maximum(state.player_energy - 1, 0), state.player_energy
    )
    new_fatigue = jax.lax.select(new_fatigue > 30, 0.0, new_fatigue)

    new_energy = jax.lax.select(
        new_fatigue < -10, jnp.minimum(state.player_energy + 1, 9), new_energy
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
    recover_not_all = jax.lax.select(state.is_sleeping, -0.5, -1.0)
    recover_add = jax.lax.select(all_necessities, recover_all, recover_not_all)

    new_recover = state.player_recover + recover_add

    recovered_health = jnp.minimum(state.player_health + 1, 9)
    derecovered_health = state.player_health - 1

    new_health = jax.lax.select(new_recover > 25, recovered_health, state.player_health)
    new_recover = jax.lax.select(new_recover > 25, 0.0, new_recover)
    new_health = jax.lax.select(new_recover < -15, derecovered_health, new_health)
    new_recover = jax.lax.select(new_recover < -15, 0.0, new_recover)

    state = state.replace(
        player_recover=new_recover,
        player_health=new_health,
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
        _set_plant_block, state.map, jnp.arange(static_params.max_growing_plants)
    )

    state = state.replace(
        map=new_map,
        growing_plants_age=growing_plants_age,
    )

    return state


def move_player(state, action):
    proposed_position = state.player_position + DIRECTIONS[action]

    valid_move = is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(
        state, proposed_position
    )
    valid_move = jnp.logical_or(
        valid_move,
        state.map[proposed_position[0], proposed_position[1]] == BlockType.LAVA.value,
    )

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


def spawn_mobs(state, rng, params, static_params):
    player_distance_map = get_distance_map(state.player_position, static_params)

    # Cows
    can_spawn_cow = state.cows.mask.sum() < static_params.max_cows

    rng, _rng = jax.random.split(rng)
    can_spawn_cow = jnp.logical_and(
        can_spawn_cow, jax.random.uniform(_rng) < params.spawn_cow_chance
    )

    cows_can_spawn_map = state.map == BlockType.GRASS.value
    cows_can_spawn_map = jnp.logical_and(cows_can_spawn_map, player_distance_map > 3)
    cows_can_spawn_map = jnp.logical_and(
        cows_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    cows_can_spawn_map = jnp.logical_and(
        cows_can_spawn_map, jnp.logical_not(state.mob_map)
    )
    can_spawn_cow = jnp.logical_and(can_spawn_cow, cows_can_spawn_map.sum() > 0)

    rng, _rng = jax.random.split(rng)
    cow_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(cows_can_spawn_map, -1) / jnp.sum(cows_can_spawn_map),
    )
    cow_position = jnp.array(
        [
            cow_position // static_params.map_size[0],
            cow_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_cow_index = jnp.argmax(jnp.logical_not(state.cows.mask))

    new_cow_position = jax.lax.select(
        can_spawn_cow,
        cow_position,
        state.cows.position[new_cow_index],
    )

    new_cow_health = jax.lax.select(
        can_spawn_cow,
        params.cow_health,
        state.cows.health[new_cow_index],
    )

    new_cow_mask = jax.lax.select(
        can_spawn_cow,
        True,
        state.cows.mask[new_cow_index],
    )

    cows = Mobs(
        position=state.cows.position.at[new_cow_index].set(new_cow_position),
        health=state.cows.health.at[new_cow_index].set(new_cow_health),
        mask=state.cows.mask.at[new_cow_index].set(new_cow_mask),
        attack_cooldown=state.cows.attack_cooldown,
    )

    state = state.replace(
        cows=cows,
        mob_map=state.mob_map.at[new_cow_position[0], new_cow_position[1]].set(
            jnp.logical_or(
                state.mob_map[new_cow_position[0], new_cow_position[1]], new_cow_mask
            )
        ),
    )

    # Zombies
    can_spawn_zombie = state.zombies.mask.sum() < static_params.max_zombies

    rng, _rng = jax.random.split(rng)
    zombie_spawn_chance = (
        params.spawn_zombie_base_chance
        + params.spawn_zombie_night_chance * jnp.square(1 - state.light_level)
    )
    can_spawn_zombie = jnp.logical_and(
        can_spawn_zombie, jax.random.uniform(_rng) < zombie_spawn_chance
    )

    grass_map = state.map == BlockType.GRASS.value
    path_map = state.map == BlockType.PATH.value
    zombies_can_spawn_map = jnp.logical_or(grass_map, path_map)
    zombies_can_spawn_map = jnp.logical_and(
        zombies_can_spawn_map, player_distance_map > 9
    )
    zombies_can_spawn_map = jnp.logical_and(
        zombies_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    zombies_can_spawn_map = jnp.logical_and(
        zombies_can_spawn_map, jnp.logical_not(state.mob_map)
    )

    can_spawn_zombie = jnp.logical_and(
        can_spawn_zombie, zombies_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    zombie_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(zombies_can_spawn_map, -1) / jnp.sum(zombies_can_spawn_map),
    )
    zombie_position = jnp.array(
        [
            zombie_position // static_params.map_size[0],
            zombie_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_zombie_index = jnp.argmax(jnp.logical_not(state.zombies.mask))

    new_zombie_position = jax.lax.select(
        can_spawn_zombie,
        zombie_position,
        state.zombies.position[new_zombie_index],
    )

    new_zombie_health = jax.lax.select(
        can_spawn_zombie,
        params.zombie_health,
        state.zombies.health[new_zombie_index],
    )

    new_zombie_mask = jax.lax.select(
        can_spawn_zombie,
        True,
        state.zombies.mask[new_zombie_index],
    )

    zombies = Mobs(
        position=state.zombies.position.at[new_zombie_index].set(new_zombie_position),
        health=state.zombies.health.at[new_zombie_index].set(new_zombie_health),
        mask=state.zombies.mask.at[new_zombie_index].set(new_zombie_mask),
        attack_cooldown=state.zombies.attack_cooldown,
    )

    state = state.replace(
        zombies=zombies,
        mob_map=state.mob_map.at[new_zombie_position[0], new_zombie_position[1]].set(
            jnp.logical_or(
                state.mob_map[new_zombie_position[0], new_zombie_position[1]],
                new_zombie_mask,
            )
        ),
    )

    # Skeletons
    can_spawn_skeleton = state.skeletons.mask.sum() < static_params.max_skeletons

    rng, _rng = jax.random.split(rng)
    can_spawn_skeleton = jnp.logical_and(
        can_spawn_skeleton, jax.random.uniform(_rng) < params.spawn_skeleton_chance
    )

    skeletons_can_spawn_map = state.map == BlockType.PATH.value
    skeletons_can_spawn_map = jnp.logical_and(
        skeletons_can_spawn_map, player_distance_map > 9
    )
    skeletons_can_spawn_map = jnp.logical_and(
        skeletons_can_spawn_map, player_distance_map < params.mob_despawn_distance
    )
    skeletons_can_spawn_map = jnp.logical_and(
        skeletons_can_spawn_map, jnp.logical_not(state.mob_map)
    )

    can_spawn_skeleton = jnp.logical_and(
        can_spawn_skeleton, skeletons_can_spawn_map.sum() > 0
    )

    rng, _rng = jax.random.split(rng)
    skeleton_position = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        shape=(1,),
        p=jnp.reshape(skeletons_can_spawn_map, -1) / jnp.sum(skeletons_can_spawn_map),
    )
    skeleton_position = jnp.array(
        [
            skeleton_position // static_params.map_size[0],
            skeleton_position % static_params.map_size[1],
        ]
    ).T.astype(jnp.int32)[0]

    new_skeleton_index = jnp.argmax(jnp.logical_not(state.skeletons.mask))

    new_skeleton_position = jax.lax.select(
        can_spawn_skeleton,
        skeleton_position,
        state.skeletons.position[new_skeleton_index],
    )

    new_skeleton_health = jax.lax.select(
        can_spawn_skeleton,
        params.skeleton_health,
        state.skeletons.health[new_skeleton_index],
    )

    new_skeleton_mask = jax.lax.select(
        can_spawn_skeleton,
        True,
        state.skeletons.mask[new_skeleton_index],
    )

    skeletons = Mobs(
        position=state.skeletons.position.at[new_skeleton_index].set(
            new_skeleton_position
        ),
        health=state.skeletons.health.at[new_skeleton_index].set(new_skeleton_health),
        mask=state.skeletons.mask.at[new_skeleton_index].set(new_skeleton_mask),
        attack_cooldown=state.skeletons.attack_cooldown,
    )

    state = state.replace(
        skeletons=skeletons,
        mob_map=state.mob_map.at[
            new_skeleton_position[0], new_skeleton_position[1]
        ].set(
            jnp.logical_or(
                state.mob_map[new_skeleton_position[0], new_skeleton_position[1]],
                new_skeleton_mask,
            )
        ),
    )

    return state


def cap_inventory(state):
    capped_inv = jax.tree_util.tree_map(lambda x: jnp.minimum(x, 9), state.inventory)

    state = state.replace(inventory=capped_inv)

    return state


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

    # Interrupt action if sleeping
    action = jax.lax.select(state.is_sleeping, Action.NOOP.value, action)

    # Crafting
    state = do_crafting(state, action)

    # Interact (mining, attacking, eating plants, drinking water)
    rng, _rng = jax.random.split(rng)
    state = do_action(_rng, state, action, static_params)

    # Placing
    state = place_block(state, action, static_params)

    # Movement
    old_position = state.player_position
    state = move_player(state, action)
    new_position = state.player_position

    # Mobs
    rng, _rng = jax.random.split(rng)
    state = update_mobs(_rng, state, params, static_params)

    rng, _rng = jax.random.split(rng)
    state = spawn_mobs(state, _rng, params, static_params)

    # Plants
    state = update_plants(state, static_params)

    # Intrinsics
    state = update_player_intrinsics(state, action)

    # Cap inv
    state = cap_inventory(state)

    # Reward
    achievement_reward = (
        state.achievements.astype(jnp.float32).sum()
        - init_achievements.astype(jnp.float32).sum()
    )
    health_reward = (state.player_health - init_health) * 0.1

    # Optional intrinsics reward based on intrinsics_diff
    # intrinsics_diff contains [health_diff, food_diff, drink_diff, energy_diff]
    # Positive changes in food/drink/energy are good, negative changes are bad
    intrinsics_reward = jax.lax.select(
        params.intrinsics_reward,
        # Reward positive changes in food, drink, energy (indices 1, 2, 3)
        jnp.sum(state.intrinsics_diff[1:]) * 0.1,
        0.0  # If False, no intrinsics reward
    )

    reward = achievement_reward + health_reward + intrinsics_reward

    rng, _rng = jax.random.split(rng)

    state = state.replace(
        timestep=state.timestep + 1,
        light_level=calculate_light_level(state.timestep + 1, params),
        state_rng=_rng,
    )

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

    return state, reward
