from __future__ import annotations

import math
from typing import Iterable

from .models import Vector


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot / (left_norm * right_norm)


def average_vectors(vectors: Iterable[Vector]) -> Vector:
    vector_list = list(vectors)
    if not vector_list:
        raise ValueError("At least one vector is required.")
    length = len(vector_list[0])
    if any(len(vector) != length for vector in vector_list):
        raise ValueError("All vectors must have the same length.")
    totals = [0.0] * length
    for vector in vector_list:
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(vector_list))
    return tuple(value / count for value in totals)


def newcomer_index(name: str, prefix: str = "新人") -> int | None:
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)
