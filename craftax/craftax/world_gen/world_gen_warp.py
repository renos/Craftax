from craftax.craftax.util.relative_positions import find_closest_blocks
import jax
import jax.scipy as jsp

from craftax.craftax.constants import *
from craftax.craftax.game_logic import calculate_light_level, get_distance_map
from craftax.craftax.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax.util.noise import generate_fractal_noise_2d
from craftax.craftax.world_gen.world_gen_configs import (
    ALL_DUNGEON_CONFIGS,
    ALL_SMOOTHGEN_CONFIGS,
)


def get_new_empty_inventory():
    return Inventory(
        wood=jnp.asarray(0, dtype=jnp.int32),
        stone=jnp.asarray(0, dtype=jnp.int32),
        coal=jnp.asarray(0, dtype=jnp.int32),
        iron=jnp.asarray(0, dtype=jnp.int32),
        diamond=jnp.asarray(0, dtype=jnp.int32),
        sapling=jnp.asarray(0, dtype=jnp.int32),
        pickaxe=jnp.asarray(0, dtype=jnp.int32),
        sword=jnp.asarray(0, dtype=jnp.int32),
        bow=jnp.asarray(0, dtype=jnp.int32),
        arrows=jnp.asarray(0, dtype=jnp.int32),
        torches=jnp.asarray(0, dtype=jnp.int32),
        ruby=jnp.asarray(0, dtype=jnp.int32),
        sapphire=jnp.asarray(0, dtype=jnp.int32),
        books=jnp.asarray(0, dtype=jnp.int32),
        potions=jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=jnp.int32,
        ),
        armour=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
    )


def get_new_full_inventory():
    return Inventory(
        wood=jnp.asarray(99, dtype=jnp.int32),
        stone=jnp.asarray(99, dtype=jnp.int32),
        coal=jnp.asarray(99, dtype=jnp.int32),
        iron=jnp.asarray(99, dtype=jnp.int32),
        diamond=jnp.asarray(99, dtype=jnp.int32),
        sapling=jnp.asarray(99, dtype=jnp.int32),
        pickaxe=jnp.asarray(4, dtype=jnp.int32),
        sword=jnp.asarray(4, dtype=jnp.int32),
        bow=jnp.asarray(1, dtype=jnp.int32),
        arrows=jnp.asarray(99, dtype=jnp.int32),
        torches=jnp.asarray(99, dtype=jnp.int32),
        ruby=jnp.asarray(99, dtype=jnp.int32),
        sapphire=jnp.asarray(99, dtype=jnp.int32),
        books=jnp.asarray(99, dtype=jnp.int32),
        potions=jnp.array(
            [
                99,
                99,
                99,
                99,
                99,
                99,
            ],
            dtype=jnp.int32,
        ),
        armour=jnp.array([2, 2, 2, 2], dtype=jnp.int32),
    )


def generate_dungeon(rng, static_params, config):
    chunk_size = 16
    world_chunk_width = static_params.map_size[0] // chunk_size
    world_chunk_height = static_params.map_size[1] // chunk_size
    room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)

    num_rooms = 8
    min_room_size = 5
    max_room_size = 10

    # ONE monolithic split for all randomness
    # Key allocation:
    #   [0]       = room_sizes
    #   [1:41]    = _add_room (8 rooms * 5 keys each)
    #   [41:49]   = _add_path (8 paths * 1 key each)
    #   [49]      = rare_map
    #   [50]      = ladder_down
    #   [51]      = ladder_up
    total_keys = 1 + num_rooms * 5 + num_rooms * 1 + 3  # = 52
    keys = jax.random.split(rng, total_keys)

    k_room_sizes = keys[0]
    room_keys = keys[1:41]      # 8 rooms * 5 keys
    path_keys = keys[41:49]     # 8 paths * 1 key
    k_rare_map = keys[49]
    k_ladder_down = keys[50]
    k_ladder_up = keys[51]

    room_sizes = jax.random.randint(
        k_room_sizes, shape=(num_rooms, 2), minval=min_room_size, maxval=max_room_size
    )

    map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
    padded_map = jnp.pad(map, max_room_size, constant_values=0)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
    padded_item_map = jnp.pad(item_map, max_room_size, constant_values=0)

    def _add_room(carry, room_index):
        block_map, item_map, room_occupancy_chunks = carry

        # Get pre-split keys for this room (5 keys per room)
        k_base = room_index * 5
        k_chunk = room_keys[k_base]
        k_pos_offset = room_keys[k_base + 1]
        k_chest = room_keys[k_base + 2]
        k_fountain_pos = room_keys[k_base + 3]
        k_fountain_chance = room_keys[k_base + 4]

        room_chunk = jax.random.choice(
            k_chunk,
            jnp.arange(world_chunk_width * world_chunk_height),
            p=room_occupancy_chunks,
        )
        room_occupancy_chunks = room_occupancy_chunks.at[room_chunk].set(0)

        room_position = jnp.array(
            [
                (room_chunk % world_chunk_height) * chunk_size,
                (room_chunk // world_chunk_height) * chunk_size,
            ]
        ) + jnp.array([max_room_size, max_room_size])
        room_position += jax.random.randint(
            k_pos_offset, (2,), minval=0, maxval=chunk_size - min_room_size
        )

        slice = jax.lax.dynamic_slice(
            block_map, room_position, (max_room_size, max_room_size)
        )
        xs = jnp.expand_dims(jnp.arange(max_room_size), axis=-1).repeat(
            max_room_size, axis=-1
        )
        ys = jnp.expand_dims(jnp.arange(max_room_size), axis=0).repeat(
            max_room_size, axis=0
        )

        room_mask = jnp.logical_and(
            xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1]
        )

        slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice

        block_map = jax.lax.dynamic_update_slice(
            block_map,
            slice,
            room_position,
        )

        # Torches in corner
        item_map = item_map.at[room_position[0], room_position[1]].set(
            ItemType.TORCH.value
        )
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1,
            room_position[1] + room_sizes[room_index, 1] - 1,
        ].set(ItemType.TORCH.value)

        # Chest
        chest_position = jax.random.randint(
            k_chest,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        block_map = block_map.at[
            room_position[0] + chest_position[0], room_position[1] + chest_position[1]
        ].set(BlockType.CHEST.value)

        # Fountain
        fountain_position = jax.random.randint(
            k_fountain_pos,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        room_has_fountain = jax.random.uniform(k_fountain_chance) > 0.5
        fountain_block = (
            room_has_fountain * config.fountain_block
            + (1 - room_has_fountain)
            * block_map[
                room_position[0] + fountain_position[0],
                room_position[1] + fountain_position[1],
            ]
        )
        block_map = block_map.at[
            room_position[0] + fountain_position[0],
            room_position[1] + fountain_position[1],
        ].set(fountain_block)

        return (block_map, item_map, room_occupancy_chunks), room_position

    (padded_map, padded_item_map, _), room_positions = jax.lax.scan(
        _add_room,
        (padded_map, padded_item_map, room_occupancy_chunks),
        jnp.arange(num_rooms),
    )

    def _add_path(carry, path_index):
        cmap, included_rooms_mask = carry

        # Get pre-split key for this path
        k_sink = path_keys[path_index]

        path_source = room_positions[path_index]

        sink_index = jax.random.choice(
            k_sink, jnp.arange(num_rooms), p=included_rooms_mask
        )
        path_sink = room_positions[sink_index]

        # Horizontal component
        entire_row = cmap[path_source[0]]
        path_indexes = jnp.arange(static_params.map_size[0] + 2 * max_room_size)
        path_indexes = path_indexes - path_source[1]
        horizontal_distance = path_sink[1] - path_source[1]
        path_indexes = path_indexes * jnp.sign(horizontal_distance)

        horizontal_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, jnp.sign(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, entire_row == BlockType.WALL.value
        )

        new_row = (
            horizontal_mask * BlockType.PATH.value + (1 - horizontal_mask) * entire_row
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_row, axis=0),
            path_source,
        )

        # Vertical component
        entire_col = cmap[:, path_sink[1]]
        path_indexes = jnp.arange(static_params.map_size[1] + 2 * max_room_size)
        path_indexes = path_indexes - path_source[0]
        vertical_distance = path_sink[0] - path_source[0]
        path_indexes = path_indexes * jnp.sign(vertical_distance)

        vertical_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
        )
        vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

        vertical_mask = jnp.logical_and(
            vertical_mask, entire_col == BlockType.WALL.value
        )

        new_col = (
            vertical_mask * BlockType.PATH.value + (1 - vertical_mask) * entire_col
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_col, axis=-1),
            path_sink,
        )

        included_rooms_mask = included_rooms_mask.at[path_index].set(True)
        return (cmap, included_rooms_mask), None

    included_rooms_mask = jnp.zeros(num_rooms, dtype=bool).at[-1].set(True)
    (padded_map, _), _ = jax.lax.scan(
        _add_path, (padded_map, included_rooms_mask), jnp.arange(0, num_rooms)
    )

    # Place special block in a random room
    special_block_position = room_positions[0] + jnp.array([2, 2])
    padded_map = padded_map.at[
        special_block_position[0], special_block_position[1]
    ].set(config.special_block)

    map = padded_map[max_room_size:-max_room_size, max_room_size:-max_room_size]
    item_map = padded_item_map[
        max_room_size:-max_room_size, max_room_size:-max_room_size
    ]

    # Visual stuff
    c_path_map = map != BlockType.WALL.value
    z = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    adj_path_map = jsp.signal.convolve(c_path_map, z, mode="same")
    adj_path_map = adj_path_map > 0.5

    rare_map = jax.random.choice(
        k_rare_map,
        jnp.array([False, True]),
        static_params.map_size,
        p=jnp.array([0.9, 0.1]),
    )

    wall_map = (
        rare_map * BlockType.WALL_MOSS.value + (1 - rare_map) * BlockType.WALL.value
    )

    rare_map = jnp.logical_and(rare_map, map == BlockType.PATH.value)
    rare_map = jnp.logical_and(rare_map, item_map == ItemType.NONE.value)
    path_map = rare_map * config.rare_path_replacement_block + (1 - rare_map) * map

    is_wall_map = jnp.logical_and(map == BlockType.WALL.value, adj_path_map)
    is_darkness_map = jnp.logical_not(adj_path_map)
    is_path_map = jnp.logical_not(jnp.logical_or(is_wall_map, is_darkness_map))

    map = (
        is_path_map * path_map
        + is_wall_map * wall_map
        + is_darkness_map * BlockType.DARKNESS.value
    )

    light_map = jnp.ones(static_params.map_size, dtype=jnp.float32)

    # Ladders
    valid_ladder_down = (map.flatten() == BlockType.PATH.value).astype(jnp.float32)
    ladder_index = jax.random.choice(
        k_ladder_down,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down / valid_ladder_down.sum(),
    )
    ladder_down_position = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    item_map = item_map.at[ladder_down_position[0], ladder_down_position[1]].set(
        ItemType.LADDER_DOWN.value
    )

    valid_ladder_up = map.flatten() == BlockType.PATH.value
    ladder_index = jax.random.choice(
        k_ladder_up,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_up,
    )
    ladder_up_position = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )
    item_map = item_map.at[ladder_up_position[0], ladder_up_position[1]].set(
        ItemType.LADDER_UP.value
    )

    return map, item_map, light_map, ladder_down_position, ladder_up_position


def generate_smoothworld(rng, static_params, player_position, config, params):
    # ONE monolithic split for all randomness
    # Key allocation:
    #   [0]     = water noise
    #   [1]     = mountain noise
    #   [2]     = path_x noise
    #   [3]     = tree_noise
    #   [4]     = tree_uniform
    #   [5:10]  = _add_ore (5 ores * 1 key each)
    #   [10]    = diamond
    #   [11]    = ladder_down
    #   [12]    = ladder_up
    total_keys = 13
    keys = jax.random.split(rng, total_keys)

    k_water = keys[0]
    k_mountain = keys[1]
    k_path_x = keys[2]
    k_tree_noise = keys[3]
    k_tree_uniform = keys[4]
    ore_keys = keys[5:10]
    k_diamond = keys[10]
    k_ladder_down = keys[11]
    k_ladder_up = keys[12]

    player_proximity_map = get_distance_map(
        player_position, static_params.map_size
    ).astype(jnp.float32)
    player_proximity_map_water = (
        player_proximity_map / config.player_proximity_map_water_strength
    )
    player_proximity_map_water = jnp.clip(
        player_proximity_map_water, 0.0, config.player_proximity_map_water_max
    )

    player_proximity_map_mountain = (
        player_proximity_map / config.player_proximity_map_mountain_strength
    )
    player_proximity_map_mountain = jnp.clip(
        player_proximity_map_mountain,
        0.0,
        config.player_proximity_map_mountain_max,
    )

    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    water = generate_fractal_noise_2d(
        k_water,
        static_params.map_size,
        small_res,
        octaves=1,
        override_angles=params.fractal_noise_angles[0],
    )
    water = water + player_proximity_map_water - 1.0

    # Water
    map = jnp.where(
        water > config.water_threshold, config.sea_block, config.default_block
    )

    sand_map = jnp.logical_and(
        water > config.sand_threshold,
        map != config.sea_block,
    )

    map = jnp.where(sand_map, config.coast_block, map)

    # Mountain vs grass
    mountain_threshold = 0.7

    mountain = (
        generate_fractal_noise_2d(
            k_mountain,
            static_params.map_size,
            small_res,
            octaves=1,
            override_angles=params.fractal_noise_angles[1],
        )
        + 0.05
    )
    mountain = mountain + player_proximity_map_mountain - 1.0
    map = jnp.where(mountain > mountain_threshold, config.mountain_block, map)

    # Paths
    path_x = generate_fractal_noise_2d(
        k_path_x,
        static_params.map_size,
        x_res,
        octaves=1,
        override_angles=params.fractal_noise_angles[2],
    )
    path = jnp.logical_and(mountain > mountain_threshold, path_x > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    path_y = path_x.T
    path = jnp.logical_and(mountain > mountain_threshold, path_y > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    # Caves
    caves = jnp.logical_and(mountain > 0.85, water > 0.4)
    map = jnp.where(caves > 0.5, config.inner_mountain_block, map)

    # Trees
    tree_noise = generate_fractal_noise_2d(
        k_tree_noise,
        static_params.map_size,
        larger_res,
        octaves=1,
        override_angles=params.fractal_noise_angles[3],
    )
    tree = (tree_noise > config.tree_threshold_perlin) * jax.random.uniform(
        k_tree_uniform, shape=static_params.map_size
    ) > config.tree_threshold_uniform
    tree = jnp.logical_and(tree, map == config.tree_requirement_block)
    map = jnp.where(tree, config.tree, map)

    # Ores
    def _add_ore(map, index):
        k_ore = ore_keys[index]
        ore_map = jnp.logical_and(
            map == config.ore_requirement_blocks[index],
            jax.random.uniform(k_ore, static_params.map_size)
            < config.ore_chances[index],
        )
        map = jnp.where(ore_map, config.ores[index], map)
        return map, None

    map, _ = jax.lax.scan(_add_ore, map, jnp.arange(5))

    # Lava
    lava_map = jnp.logical_and(
        mountain > 0.85,
        tree_noise > 0.7,
    )
    map = jnp.where(lava_map, config.lava, map)

    # Add diamond if always_diamond flag is set
    adding_diamond = jnp.logical_and(
        config.default_block == BlockType.GRASS.value,  # Hacky check for overworld
        params.always_diamond,
    )
    valid_diamond = (map.flatten() == BlockType.STONE.value).astype(jnp.float32)
    diamond_index = jax.random.choice(
        k_diamond,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_diamond / valid_diamond.sum(),
    )
    diamond_position = jnp.array(
        [
            diamond_index // static_params.map_size[0],
            diamond_index % static_params.map_size[0],
        ]
    )
    diamond_replace_block = jax.lax.select(
        adding_diamond, BlockType.DIAMOND.value, BlockType.STONE.value
    )
    map = map.at[diamond_position[0], diamond_position[1]].set(diamond_replace_block)

    # Light map
    light_map = (
        jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light
    )

    # Make sure player spawns on grass
    map = map.at[player_position[0], player_position[1]].set(config.player_spawn)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)

    valid_ladder_down = map.flatten() == config.valid_ladder
    ladder_index = jax.random.choice(
        k_ladder_down,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down,
    )
    ladder_down = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    item_map = item_map.at[ladder_down[0], ladder_down[1]].set(
        ItemType.LADDER_DOWN.value * config.ladder_down
        + item_map[ladder_down[0], ladder_down[1]] * (1 - config.ladder_down)
    )

    valid_ladder_up = map.flatten() == config.valid_ladder
    ladder_index = jax.random.choice(
        k_ladder_up,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_up,
    )
    ladder_up = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
        1 - config.default_light
    ) + config.default_light * jnp.ones((9, 9))

    light_map = jax.lax.dynamic_update_slice(
        light_map, LIGHT_MAP_AROUND_LADDER, ladder_up - jnp.array([4, 4])
    )

    z = jnp.array([[0.2, 0.7, 0.2], [0.7, 1, 0.7], [0.2, 0.7, 0.2]]) * (
        config.lava == BlockType.LAVA.value
    )
    light_map += jsp.signal.convolve(lava_map, z, mode="same")
    light_map = jnp.clip(light_map, 0.0, 1.0)

    item_map = item_map.at[ladder_up[0], ladder_up[1]].set(
        ItemType.LADDER_UP.value * config.ladder_up
        + item_map[ladder_up[0], ladder_up[1]] * (1 - config.ladder_up)
    )

    return map, item_map, light_map, ladder_down, ladder_up


def generate_world(rng, params, static_params):
    # ONE monolithic split for all randomness
    # Key allocation:
    #   [0:6]   = smoothgens (6 worlds)
    #   [6:9]   = dungeons (3 dungeons)
    #   [9]     = potion_mapping
    #   [10]    = state_rng
    total_keys = 11
    keys = jax.random.split(rng, total_keys)

    smoothgen_keys = keys[0:6]
    dungeon_keys = keys[6:9]
    k_potion = keys[9]
    k_state_rng = keys[10]

    player_position = jnp.array(
        [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
    )

    # Generate smoothgens (overworld, caves, elemental levels, boss level)
    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0, None))(
        smoothgen_keys, static_params, player_position, ALL_SMOOTHGEN_CONFIGS, params
    )

    # Generate dungeons
    dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        dungeon_keys, static_params, ALL_DUNGEON_CONFIGS
    )

    # Splice smoothgens and dungeons in order of levels
    map, item_map, light_map, ladders_down, ladders_up = jax.tree_util.tree_map(
        lambda x, y: jnp.stack(
            (x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0
        ),
        smoothgens,
        dungeons,
    )

    # Mobs
    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (static_params.num_levels, max_mobs, 2), dtype=jnp.int32
            ),
            health=jnp.ones((static_params.num_levels, max_mobs), dtype=jnp.float32),
            mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_mobs), dtype=jnp.int32
            ),
            type_id=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
        )

    melee_mobs = generate_empty_mobs(static_params.max_melee_mobs)
    ranged_mobs = generate_empty_mobs(static_params.max_ranged_mobs)
    passive_mobs = generate_empty_mobs(static_params.max_passive_mobs)

    # Projectiles
    def _create_projectiles(max_num):
        projectiles = generate_empty_mobs(max_num)

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        return projectiles, projectile_directions

    mob_projectiles, mob_projectile_directions = _create_projectiles(
        static_params.max_mob_projectiles
    )
    player_projectiles, player_projectile_directions = _create_projectiles(
        static_params.max_player_projectiles
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    #looking at map 0 since player spawns in overworld
    padded_grid = jnp.pad(
        map[0],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    # Pad item_map the same way
    padded_item_map = jnp.pad(
        item_map[0],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )
    tl_corner = player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    # Compute closest_blocks only if include_relative_positions is enabled
    # num_blocks = len(BlockType) + 3 ladder channels
    num_blocks = len(BlockType) + 3
    k = 5
    if static_params.include_relative_positions:
        map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
        map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))
        # Include item overlay channels for ladders so they appear in closest_blocks
        item_view = jax.lax.dynamic_slice(padded_item_map, tl_corner, OBS_DIM)
        item_one_hot = jax.nn.one_hot(item_view, num_classes=len(ItemType))
        ladder_down_ch = item_one_hot[:, :, ItemType.LADDER_DOWN.value : ItemType.LADDER_DOWN.value + 1]
        ladder_up_ch = item_one_hot[:, :, ItemType.LADDER_UP.value : ItemType.LADDER_UP.value + 1]
        ladder_down_blocked_ch = item_one_hot[:, :, ItemType.LADDER_DOWN_BLOCKED.value : ItemType.LADDER_DOWN_BLOCKED.value + 1]
        semantic_map = jnp.concatenate([map_view_one_hot, ladder_down_ch, ladder_up_ch, ladder_down_blocked_ch], axis=-1)
        closest_blocks = find_closest_blocks(obs_dim_array // 2, semantic_map)

        # Initialize per-floor storage for floor-aware closest blocks
        # All floors start as sentinel (30,30) except floor 0 which gets initial closest_blocks
        closest_blocks_per_floor = jnp.full(
            (static_params.num_levels, num_blocks, 2, k),
            30,
            dtype=jnp.int32
        )
        closest_blocks_per_floor = closest_blocks_per_floor.at[0].set(closest_blocks)
    else:
        # Skip relative position computation - initialize with sentinel values
        closest_blocks = jnp.full((num_blocks, 2, k), 30, dtype=jnp.int32)
        closest_blocks_per_floor = jnp.full(
            (static_params.num_levels, num_blocks, 2, k),
            30,
            dtype=jnp.int32
        )

    # Potion mapping for episode
    potion_mapping = jax.random.permutation(k_potion, jnp.arange(6))

    # Inventory
    inventory = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(),
        get_new_empty_inventory(),
    )

    state = EnvState(
        map=map,
        item_map=item_map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),  # First ladder starts open
        player_position=player_position,
        player_direction=jnp.asarray(Action.UP.value, dtype=jnp.int32),
        player_level=jnp.asarray(0, dtype=jnp.int32),
        player_health=jnp.asarray(9.0, dtype=jnp.float32),
        player_food=jnp.asarray(9, dtype=jnp.int32),
        player_drink=jnp.asarray(9, dtype=jnp.int32),
        player_energy=jnp.asarray(9, dtype=jnp.int32),
        player_mana=jnp.asarray(9, dtype=jnp.int32),
        player_recover=jnp.asarray(0.0, dtype=jnp.float32),
        player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
        player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
        player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
        player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
        is_sleeping=False,
        is_resting=False,
        player_xp=jnp.asarray(0, dtype=jnp.int32),
        player_dexterity=jnp.asarray(1, dtype=jnp.int32),
        player_strength=jnp.asarray(1, dtype=jnp.int32),
        player_intelligence=jnp.asarray(1, dtype=jnp.int32),
        inventory=inventory,
        sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
        bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
        armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.array([False, False], dtype=bool),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(
            BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32
        ),
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=jnp.asarray(calculate_light_level(0, params), dtype=jnp.float32),
        state_rng=k_state_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
        closest_blocks=closest_blocks,
        closest_blocks_per_floor=closest_blocks_per_floor,
        player_state=0,
        player_state_diff=0,
        inventory_diff=get_new_empty_inventory(),
        intrinsics_diff=jnp.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=jnp.int32,
        ),
        task_done = False,
        achievements_diff=jnp.zeros((len(Achievement),), dtype=jnp.bool),
        closest_blocks_prev=jnp.zeros_like(closest_blocks, dtype=jnp.int32),
    )

    return state
