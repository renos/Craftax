import jax.numpy as jnp
import jax

def find_closest_blocks(player_position, semantic_map, k=5):

    height, width, num_blocks = semantic_map.shape

    y_grid, x_grid = jnp.mgrid[0:height, 0:width]
    # distances = jnp.sqrt(
    #     (x_grid - agent_position[0]) ** 2 + (y_grid - agent_position[1]) ** 2
    # )
    distances = jnp.abs(x_grid - player_position[1]) + jnp.abs(
        y_grid - player_position[0]
    )

    # Prepare for the output
    closest_blocks_positions = jnp.full((k, 2, num_blocks), jnp.nan)

    def process_block_type(block_id):
        block_map = semantic_map[:, :, block_id]
        block_distances = jnp.where(block_map == 1, distances, jnp.inf)

        # Flatten for easy indexing
        flat_distances = block_distances.ravel()

        # Argsort distances and select the k smallest
        sorted_indices = jnp.argsort(flat_distances)[:k]
        sorted_y, sorted_x = jnp.unravel_index(sorted_indices, block_distances.shape)

        sorted_y_relative = sorted_y - player_position[0]
        sorted_x_relative = sorted_x - player_position[1]

        # Create a mask to find which of the sorted indices are inf (indicating no block present)
        is_inf_mask = jnp.isinf(flat_distances[sorted_indices])

        # Replace positions corresponding to inf distances with the predetermined far away value
        # Assuming (20, 20) is outside your map and used as a placeholder for 'far away'
        far_away_value = jnp.array([30, 30])
        closest_y = jnp.where(is_inf_mask, far_away_value[0], sorted_y_relative)
        closest_x = jnp.where(is_inf_mask, far_away_value[1], sorted_x_relative)

        return jnp.stack([closest_x, closest_y])

    # Vectorize the operation for each block type
    closest_blocks_positions = jax.vmap(process_block_type)(jnp.arange(num_blocks))

    return closest_blocks_positions


def update_relative_positions(closest_blocks, dx, dy, height, width):
    num_blocks, _, k = closest_blocks.shape

    def update_block_positions(block_id):
        # new_block_positions = closest_blocks[block_id] - jnp.array([dx, dy]).reshape(
        #     (2, 1)
        # )
        block_positions = closest_blocks[block_id]
        not_nan_mask = jnp.logical_or(
            block_positions[0, :] != 30, block_positions[1, :] != 30
        )
        subtraction_array = jnp.array([dy, dx]).reshape((2, 1))

        # Apply subtraction only to entries that are not (30, 30)
        new_block_positions = jnp.where(
            not_nan_mask, block_positions - subtraction_array, block_positions
        )

        # remove
        within_bounds_y = jnp.abs(new_block_positions[0, :]) <= width
        within_bounds_x = jnp.abs(new_block_positions[1, :]) <= height
        within_bounds = within_bounds_y & within_bounds_x

        # Use jnp.where to replace entries within bounds with (20, 20), respecting the y-first order
        replaced_block_positions_y = jnp.where(
            within_bounds, 30, new_block_positions[0, :]
        )
        replaced_block_positions_x = jnp.where(
            within_bounds, 30, new_block_positions[1, :]
        )

        # Reconstruct the positions array with replaced values
        replaced_block_positions = jnp.stack(
            (replaced_block_positions_y, replaced_block_positions_x), axis=0
        )

        return replaced_block_positions  # replaced_block_positions

    block_ids = jnp.arange(num_blocks)
    closest_blocks = jax.vmap(update_block_positions)(block_ids)

    return closest_blocks


# {diamond : [-1, -1]}
# dx,dy = -1, 0
# {diamond : [-2, -1]}


def merge_old_new(closest_blocks, k):
    num_blocks, _, _ = closest_blocks.shape

    def update_block_positions(block_id):
        closest_block_i = closest_blocks[block_id]
        # Calculate L1 distances from the origin
        l1_distances = jnp.sum(jnp.abs(closest_block_i), axis=0)

        # Sort based on L1 distance
        sorted_indices = jnp.argsort(l1_distances)[:k]
        sorted_block_positions = closest_block_i[:, sorted_indices]
        return sorted_block_positions

    block_ids = jnp.arange(num_blocks)
    closest_blocks = jax.vmap(update_block_positions)(block_ids)

    return closest_blocks


def update_closest_blocks(state, old_pos, new_pos, OBS_DIM, MAX_OBS_DIM, BlockType):
    k = 5
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
    padded_grid = jnp.pad(
        state.map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))
    player_pos = state.player_position
    new_closest_blocks = find_closest_blocks(obs_dim_array // 2, map_view_one_hot)

    old_blocks = state.closest_blocks
    dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]

    height, width = obs_dim_array // 2
    old_blocks_updated = update_relative_positions(old_blocks, dx, dy, height, width)

    combined = jnp.concatenate([old_blocks_updated, new_closest_blocks], axis=-1)

    new_closest_blocks = merge_old_new(combined, k)

    state = state.replace(closest_blocks=new_closest_blocks)

    return state
