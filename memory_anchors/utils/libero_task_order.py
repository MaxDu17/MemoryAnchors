from __future__ import annotations

from typing import Optional

from omegaconf import OmegaConf


TASK_ORDERS: list[list[int]] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


def _as_list(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


def _validate_permutation(permutation: list[int], n_tasks: int) -> None:
    assert len(permutation) == n_tasks, (
        f"Expected permutation length {n_tasks}, got {len(permutation)}."
    )
    expected_values = set(range(n_tasks))
    actual_values = set(permutation)
    assert actual_values == expected_values, (
        f"Permutation must contain each integer from 0 to {n_tasks - 1} exactly once. "
        f"Got: {permutation}"
    )


def resolve_task_order(
    tasks_to_evaluate,
    task_order_index: Optional[int] = None,
    task_order_permutation=None,
) -> list[str]:
    tasks = _as_list(tasks_to_evaluate)
    permutation = _as_list(task_order_permutation)
    index = None if task_order_index is None else int(task_order_index)

    assert tasks is not None, "tasks_to_evaluate must be provided."
    n_tasks = len(tasks)

    assert not (index is not None and permutation is not None), (
        "Only one of task_order_index or task_order_permutation may be set."
    )

    if index is not None:
        assert 0 <= index < len(TASK_ORDERS), (
            f"task_order_index out of range. Got {index}, valid range is "
            f"[0, {len(TASK_ORDERS) - 1}]."
        )
        permutation = TASK_ORDERS[index]
    elif permutation is None:
        permutation = list(range(n_tasks))

    assert isinstance(permutation, list), "task_order_permutation must be a list."
    permutation = [int(x) for x in permutation]
    _validate_permutation(permutation, n_tasks=n_tasks)
    return [tasks[i] for i in permutation]


def register_libero_task_order_resolver() -> None:
    OmegaConf.register_new_resolver(
        "libero_task_order",
        resolve_task_order,
        replace=True,
    )
