"""Shared constants of the "NBT High Conflict" pipeline (paper Sec. 5.2 / App. B.4).

The K task-task pairs with the largest moving-anchor forgetting Delta_{i,j} on the reference
(RandER) runs are the "high-conflict" pairs; every method is then scored on those pairs.
"""

WORST_POSITIVE_BT_PAIR_TOP_K = 6  # paper: n = 6

PAIRWISE_CORRELATIONS_TOP_POSITIVE_BT_KEY = f"top{WORST_POSITIVE_BT_PAIR_TOP_K}_most_positive_bt_pairs"
PAIRWISE_CORRELATIONS_TOP_NEGATIVE_KEY = f"top{WORST_POSITIVE_BT_PAIR_TOP_K}_most_negative_pairs"
TARGET_TASK_EXTRACTION_FILENAME = f"top{WORST_POSITIVE_BT_PAIR_TOP_K}_bt_pair_delta_extraction.json"
TARGET_TASK_EXTRACTION_AGGREGATE_PAIRS_KEY = f"aggregate_top{WORST_POSITIVE_BT_PAIR_TOP_K}_pairs"
