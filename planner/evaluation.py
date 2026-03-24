from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int, int]


def compute_path_metrics(path: Sequence[GridPoint], env_data_sequence, config: Dict[str, object], stats=None):
    if not path:
        return {}

    metrics = {}

    total_dist = 0.0
    for i in range(1, len(path)):
        total_dist += float(np.linalg.norm(np.asarray(path[i], dtype=float) - np.asarray(path[i - 1], dtype=float)))
    metrics["path_length"] = total_dist

    energy_total = 0.0
    for i in range(1, len(path)):
        env = env_data_sequence[min(i, len(env_data_sequence) - 1)]
        cur_u, cur_v, cur_w = env["current_field"]
        x, y, z = path[i]
        current_vec = np.array([cur_u[x, y, z], cur_v[x, y, z], cur_w[x, y, z]], dtype=float)
        direction = np.asarray(path[i], dtype=float) - np.asarray(path[i - 1], dtype=float)
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            continue
        cos_angle = np.dot(current_vec, direction) / (np.linalg.norm(current_vec) * direction_norm + 1e-8)
        energy_total += max(1 - cos_angle, 0.1)
    metrics["total_energy"] = energy_total

    signal_field = env_data_sequence[0]["signal_field"]
    compliant = sum(1 for x, y, z in path if signal_field[x, y, z] >= float(config["S_th"]))
    metrics["signal_compliance"] = compliant / max(len(path), 1)

    min_obs_dist = float("inf")
    for i, node in enumerate(path):
        env = env_data_sequence[min(i, len(env_data_sequence) - 1)]
        for obs in env["obstacles_t"]:
            min_obs_dist = min(min_obs_dist, float(np.linalg.norm(np.asarray(node) - np.asarray(obs))))
    metrics["min_obstacle_dist"] = min_obs_dist if np.isfinite(min_obs_dist) else None

    angles = []
    for i in range(2, len(path)):
        v0 = np.asarray(path[i - 1], dtype=float) - np.asarray(path[i - 2], dtype=float)
        v1 = np.asarray(path[i], dtype=float) - np.asarray(path[i - 1], dtype=float)
        cos_t = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-8)
        angles.append(float(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    avg_turn_angle = float(np.mean(angles)) if angles else 0.0
    metrics["avg_turn_angle"] = avg_turn_angle
    metrics["smoothness"] = 1.0 - avg_turn_angle / np.pi

    if stats is not None:
        metrics["total_nodes_expanded"] = stats.get("total_nodes_expanded", 0)
        metrics["replan_count"] = stats.get("replan_count", 0)

    return metrics
