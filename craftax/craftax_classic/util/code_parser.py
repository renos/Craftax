from jax import numpy as jnp

import jax

import re


def return_task_reward_func(module_dict):
    pattern = r"^task_\d+_reward$"
    # reward_functions = {
    #     name: obj
    #     for name, obj in module_dict.items()
    #     if re.match(pattern, name) and callable(obj)
    # }
    reward_functions = sorted(
        [
            obj
            for name, obj in module_dict.items()
            if re.match(pattern, name) and callable(obj)
        ],
        key=lambda f: int(re.search(r"\d+", f.__name__).group()),
    )

    def check_task_reward(
        task_num,
        intrinsics_diff,
        inventory_diff,
        closest_blocks,
        closest_blocks_prev,
        env_reward,
        in_lava_pen,
        achievements_diff,
    ):
        health_diff = intrinsics_diff[
            0
        ]  # Update this index based on actual health position

        health_penalty = 0.1 * health_diff + in_lava_pen + 0.1 * intrinsics_diff[1:].sum()

        # Dynamically select the reward function based on task_num
        # Construct the function name as a string
        inputs = (
            inventory_diff,
            closest_blocks,
            closest_blocks_prev,
            intrinsics_diff,
            achievements_diff,
            # env_reward,
            health_penalty,
            # closest_blocks_prev,
        )

        # total_reward = jax.lax.switch(
        #     task_num, reward_functions, [inputs for _ in range(len(reward_functions))]
        # )
        all_rewards = jnp.array([f(*inputs) for f in reward_functions])
        total_reward = all_rewards[task_num]

        return total_reward

    return check_task_reward, len(reward_functions)


def return_check_task_completion_func(module_dict):

    # Regular expression to match "is done" functions based on naming convention
    pattern = r"^task_\d+_is_done$"

    # Find all matching functions and sort them by task number
    is_done_functions = sorted(
        [
            obj
            for name, obj in module_dict.items()
            if re.match(pattern, name) and callable(obj)
        ],
        key=lambda f: int(re.search(r"\d+", f.__name__).group()),
    )
    num_done_funcs = len(is_done_functions)

    def check_task_completion(
        current_task,
        closest_blocks,
        closest_blocks_prev,
        inventory,
        inventory_diff,
        player_intrinsics,
        player_intrinsics_diff,
        achievements,
    ):
        # Assuming a maximum task number; adjust as needed
        conditions = jnp.array(
            [
                func(
                    inventory,
                    inventory_diff,
                    closest_blocks,
                    closest_blocks_prev,
                    player_intrinsics,
                    player_intrinsics_diff,
                    achievements,
                )
                for func in is_done_functions
            ]
        )

        # Ensure there's a default "is done" status for tasks beyond those defined
        conditions = jnp.concatenate([conditions, jnp.array([False])])

        # Determine if the current task is complete
        task_complete = conditions[current_task]
        # Decide on the next task based on the completion status
        next_task = jnp.where(task_complete, current_task + 1, current_task)

        return next_task, task_complete

    return check_task_completion, num_done_funcs


# def return_task_heads(module_dict, correct_num_reward_funcs):
#     pattern = r"^task_\d+_heads$"
#     head_functions = sorted(
#         [
#             obj
#             for name, obj in module_dict.items()
#             if re.match(pattern, name) and callable(obj)
#         ],
#         key=lambda f: int(re.search(r"\d+", f.__name__).group()),
#     )
#     common_heads = []
#     task_specific_heads = []

#     # Execute each function to get the heads and append them to the lists
#     for func in head_functions:
#         joint_head, task_specific_head = func()
#         common_heads.append(joint_head)
#         task_specific_heads.append(task_specific_head)

#     # Convert lists to jax arrays
#     common_heads_array = jnp.array(common_heads)
#     task_specific_heads_array = jnp.array(task_specific_heads)

#     if len(common_heads_array) != correct_num_reward_funcs:
#         common_heads_array = jnp.arange(correct_num_reward_funcs)
#         task_specific_heads_array = jnp.arange(correct_num_reward_funcs)

#     return (
#         common_heads_array,
#         task_specific_heads_array,
#         jnp.max(common_heads_array) + 1,  # taking into account zero indexing
#         jnp.max(task_specific_heads_array) + 1,
#     )


def return_task_heads(correct_num_reward_funcs):
    common_heads_array = jnp.arange(correct_num_reward_funcs)

    return (
        common_heads_array,
        jnp.max(common_heads_array) + 1,  # taking into account zero indexing
    )


def task_to_skill(module_dict):
    pattern = r"^task_\d+_network_number$"  # e.g. task_0_network_number
    head_functions = sorted(
        [
            obj
            for name, obj in module_dict.items()
            if re.match(pattern, name) and callable(obj)
        ],
        key=lambda f: int(re.search(r"\d+", f.__name__).group()),
    )
    # Map each network number to its corresponding skill function name
    network_to_skill = []
    for func in head_functions:
        network_number = func()  # Call the function to get its network number
        network_to_skill.append(network_number)

    #assert max(network_to_skill) < len(network_to_skill)

    return network_to_skill


def task_and_reward_funcs(module_dict):
    check_task_reward, num_reward_funcs = return_task_reward_func(module_dict)
    check_task_completion, num_complete_funcs = return_check_task_completion_func(
        module_dict
    )
    task_to_skill_index = task_to_skill(module_dict)
    num_skills = (
        max(task_to_skill_index) + 1
    )  # 0 indexing so if skills [0] -> we have 1 skill

    common_heads, num_common_heads = (
        return_task_heads(num_skills)
    )

    assert (
        num_reward_funcs == num_complete_funcs
    ), "Number of task completion functions must match number of reward functions"
    return (
        check_task_completion,
        check_task_reward,
        num_reward_funcs,
        common_heads,
        num_common_heads,
        task_to_skill_index,
    )
