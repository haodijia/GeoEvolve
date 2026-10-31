"""Equivalence of frozen observation values, without model or file digests."""
import filecmp
from pathlib import Path


def equivalent_results(left, right, root=None):
    if left == right:
        return True
    if isinstance(left, str) and isinstance(right, str):
        first = Path(root) / left if root else Path(left)
        second = Path(root) / right if root else Path(right)
        return first.is_file() and second.is_file() and filecmp.cmp(first, second, shallow=False)
    return False
