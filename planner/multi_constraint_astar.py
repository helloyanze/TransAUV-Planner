from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int, int]


def euclidean_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def estimate_return_energy(current_node: GridPoint, home_point: GridPoint) -> float:
    return float(sum(abs(a - b) for a, b in zip(current_node, home_point)) * 1.2)


def compute_turn_angle(prev_node: Optional[GridPoint], current_node: GridPoint, next_node: GridPoint) -> float:
    if prev_node is None:
        return 0.0

    v0 = np.asarray(current_node, dtype=float) - np.asarray(prev_node, dtype=float)
    v1 = np.asarray(next_node, dtype=float) - np.asarray(current_node, dtype=float)
    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)
    if norm0 == 0 or norm1 == 0:
        return 0.0

    cos_theta = np.dot(v0, v1) / (norm0 * norm1 + 1e-8)
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def in_bounds(node: GridPoint, grid_size: int) -> bool:
    x, y, z = node
    return 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size


def check_hard_constraints(
    node: GridPoint,
    parent_node: Optional[GridPoint],
    grandparent_node: Optional[GridPoint],
    terrain: np.ndarray,
    signal_field: np.ndarray,
    energy_used: float,
    obstacles_t: Sequence[GridPoint],
    config: Dict[str, object],
):
    x, y, z = node

    if terrain[x, y, z] == 1:
        return False, "terrain_collision"

    if signal_field[x, y, z] < float(config["S_crit"]):
        return False, "comm_blackout"

    energy_to_home = estimate_return_energy(node, tuple(config["home_point"]))
    if energy_used + energy_to_home > float(config["energy_budget"]) * 0.95:
        return False, "energy_critical"

    for obs in obstacles_t:
        if euclidean_dist(node, obs) < float(config["R_hard"]):
            return False, "obstacle_collision"

    if parent_node is not None:
        turn_angle = compute_turn_angle(grandparent_node, parent_node, node)
        if turn_angle > float(config["max_turn_angle"]):
            return False, "turn_radius_violation"

    return True, "pass"


def compute_safety_cost(
    node: GridPoint,
    obstacles_t: Sequence[GridPoint],
    signal_field: np.ndarray,
    config: Dict[str, object],
):
    x, y, z = node
    r_hard = float(config["R_hard"])
    r_warn = float(config["R_warn"])
    lam = float(config["lambda_obs"])

    c_obstacle_soft = 0.0
    for obs in obstacles_t:
        dist = euclidean_dist(node, obs)
        if r_hard <= dist < r_warn:
            c_obstacle_soft = max(c_obstacle_soft, float(np.exp(-lam * (dist - r_hard))))

    s = float(signal_field[x, y, z])
    s_crit = float(config["S_crit"])
    s_th = float(config["S_th"])
    if s >= s_th:
        c_signal_soft = 0.0
    else:
        c_signal_soft = ((s_th - s) / max(s_th - s_crit, 1e-8)) ** 2
        c_signal_soft = min(c_signal_soft, 1.0)

    return c_obstacle_soft, c_signal_soft


def compute_efficiency_cost(
    node: GridPoint,
    parent_node: Optional[GridPoint],
    grandparent_node: Optional[GridPoint],
    current_field,
    step_count: int,
    config: Dict[str, object],
):
    x, y, z = node
    cur_u, cur_v, cur_w = current_field
    v_current = np.array([cur_u[x, y, z], cur_v[x, y, z], cur_w[x, y, z]], dtype=float)

    if parent_node is not None:
        direction = np.asarray(node, dtype=float) - np.asarray(parent_node, dtype=float)
        direction_norm = np.linalg.norm(direction)
        d_hat = direction / direction_norm if direction_norm > 0 else np.zeros(3, dtype=float)

        current_norm = np.linalg.norm(v_current)
        v_hat = v_current / current_norm if current_norm > 0 else np.zeros(3, dtype=float)
        direction_cost = (1 - np.dot(v_hat, d_hat)) / 2.0
        strength_cost = min(current_norm / max(float(config["v_current_max"]), 1e-8), 1.0)
        beta = float(config.get("beta_strength", 0.3))
        c_energy = (1 - beta) * direction_cost + beta * strength_cost
    else:
        c_energy = 0.0

    delta_theta = compute_turn_angle(grandparent_node, parent_node, node) if parent_node is not None else 0.0
    c_motion = (delta_theta / np.pi) ** 2
    c_length = min(step_count / max(float(config["D_max"]), 1.0), 1.0)

    depth_factor = z / max(float(config["grid_size"]) - 1, 1.0)
    dist_from_beacon = euclidean_dist(node, tuple(config["acoustic_beacon"]))
    beacon_range = max(float(config.get("beacon_range", 15.0)), 1.0)
    range_factor = min(dist_from_beacon / beacon_range, 1.0)
    c_localize = 0.4 * depth_factor + 0.6 * range_factor

    return c_energy, c_motion, c_length, c_localize


def compute_transition_cost(
    node: GridPoint,
    parent_node: Optional[GridPoint],
    grandparent_node: Optional[GridPoint],
    env_data: Dict[str, object],
    config: Dict[str, object],
    weights: Dict[str, float],
    energy_used: float,
    step_count: int,
):
    valid, reason = check_hard_constraints(
        node=node,
        parent_node=parent_node,
        grandparent_node=grandparent_node,
        terrain=env_data["terrain"],
        signal_field=env_data["signal_field"],
        energy_used=energy_used,
        obstacles_t=env_data["obstacles_t"],
        config=config,
    )
    if not valid:
        return float("inf"), reason

    c_obs, c_sig = compute_safety_cost(node, env_data["obstacles_t"], env_data["signal_field"], config)
    c_eng, c_mot, c_len, c_loc = compute_efficiency_cost(
        node, parent_node, grandparent_node, env_data["current_field"], step_count, config
    )

    safety_cost = weights["w_obstacle"] * c_obs + weights["w_signal"] * c_sig
    efficiency_cost = (
        weights["w_energy"] * c_eng
        + weights["w_motion"] * c_mot
        + weights["w_length"] * c_len
        + weights["w_localize"] * c_loc
    )
    step_distance = euclidean_dist(node, parent_node) if parent_node is not None else 0.0
    g_step = step_distance + float(config.get("safety_priority", 2.0)) * safety_cost + efficiency_cost
    return g_step, "pass"


class AdaptiveWeightManager:
    WEIGHT_TEMPLATES = {
        "open_water": {
            "w_obstacle": 0.20,
            "w_signal": 0.30,
            "w_energy": 0.25,
            "w_motion": 0.10,
            "w_length": 0.10,
            "w_localize": 0.05,
        },
        "dense_reef": {
            "w_obstacle": 0.45,
            "w_signal": 0.20,
            "w_energy": 0.15,
            "w_motion": 0.10,
            "w_length": 0.05,
            "w_localize": 0.05,
        },
        "strong_current": {
            "w_obstacle": 0.20,
            "w_signal": 0.15,
            "w_energy": 0.45,
            "w_motion": 0.08,
            "w_length": 0.07,
            "w_localize": 0.05,
        },
        "comm_edge": {
            "w_obstacle": 0.15,
            "w_signal": 0.50,
            "w_energy": 0.15,
            "w_motion": 0.05,
            "w_length": 0.05,
            "w_localize": 0.10,
        },
        "low_battery": {
            "w_obstacle": 0.20,
            "w_signal": 0.10,
            "w_energy": 0.50,
            "w_motion": 0.05,
            "w_length": 0.10,
            "w_localize": 0.05,
        },
    }

    def __init__(self, config: Dict[str, object]):
        self.config = config
        self.current_weights = self.WEIGHT_TEMPLATES["open_water"].copy()

    def detect_scenario(self, node: GridPoint, env_data: Dict[str, object], energy_used: float):
        x, y, z = node
        terrain = env_data["terrain"]
        cur_u, cur_v, cur_w = env_data["current_field"]
        gs = int(self.config["grid_size"])
        radius = int(self.config.get("perception_range", 5))

        obstacle_count = 0
        total_cells = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < gs and 0 <= ny < gs and 0 <= nz < gs:
                        total_cells += 1
                        if terrain[nx, ny, nz] == 1:
                            obstacle_count += 1
        obstacle_density = obstacle_count / max(total_cells, 1)

        local_speed = float(np.sqrt(cur_u[x, y, z] ** 2 + cur_v[x, y, z] ** 2 + cur_w[x, y, z] ** 2))
        signal_strength = float(env_data["signal_field"][x, y, z])
        energy_ratio = 1.0 - energy_used / max(float(self.config["energy_budget"]), 1.0)

        if energy_ratio < 0.25:
            return "low_battery", 0.9
        if signal_strength < float(self.config["S_th"]) * 1.5:
            return "comm_edge", 0.8
        if obstacle_density > 0.25:
            return "dense_reef", 0.85
        if local_speed > float(self.config["v_current_max"]) * 0.6:
            return "strong_current", 0.8
        return "open_water", 0.7

    def get_weights(self, node: GridPoint, env_data: Dict[str, object], energy_used: float):
        scenario, confidence = self.detect_scenario(node, env_data, energy_used)
        target = self.WEIGHT_TEMPLATES[scenario]
        smooth_factor = 0.3
        for key, value in target.items():
            self.current_weights[key] = (1 - smooth_factor) * self.current_weights[key] + smooth_factor * value
        return self.current_weights.copy(), scenario, confidence


@dataclass
class SearchResult:
    path: Optional[List[GridPoint]]
    stats: Dict[str, object]


class MultiConstraintAStar:
    DIRECTIONS_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(self, config: Dict[str, object], transformer_model=None):
        self.config = config
        self.grid_size = int(config["grid_size"])
        self.transformer = transformer_model
        self.weight_manager = AdaptiveWeightManager(config)
        self.replan_interval = int(config.get("replan_interval", 3))
        self.alpha = float(config.get("alpha", 1.2))

    def heuristic(self, node: GridPoint, goal: GridPoint, env_data: Dict[str, object], t: int) -> float:
        if self.transformer is not None and hasattr(self.transformer, "predict_cost_to_goal"):
            return float(self.transformer.predict_cost_to_goal(node, goal, env_data, t)) * 0.85
        return euclidean_dist(node, goal)

    def reconstruct_path(self, came_from: Dict[GridPoint, GridPoint], current: GridPoint) -> List[GridPoint]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _single_search(
        self,
        start: GridPoint,
        goal: GridPoint,
        env_data: Dict[str, object],
        current_time: int,
        total_energy: float,
    ) -> SearchResult:
        open_heap: List[Tuple[float, float, GridPoint]] = []
        heapq.heappush(open_heap, (self.heuristic(start, goal, env_data, current_time), 0.0, start))

        came_from: Dict[GridPoint, GridPoint] = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0.0
        closed = set()
        nodes_expanded = 0
        scenario_trace: List[Tuple[int, str, float]] = []
        reject_counter = defaultdict(int)

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1

            if current == goal:
                return SearchResult(
                    path=self.reconstruct_path(came_from, current),
                    stats={
                        "nodes_expanded": nodes_expanded,
                        "scenario_trace": scenario_trace,
                        "reject_counter": dict(reject_counter),
                    },
                )

            weights, scenario, confidence = self.weight_manager.get_weights(current, env_data, total_energy + current_g)
            scenario_trace.append((current_time, scenario, confidence))

            parent = came_from.get(current)
            grandparent = came_from.get(parent) if parent is not None else None

            for dx, dy, dz in self.DIRECTIONS_26:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not in_bounds(neighbor, self.grid_size) or neighbor in closed:
                    continue

                transition_cost, reason = compute_transition_cost(
                    neighbor,
                    current,
                    parent,
                    env_data,
                    self.config,
                    weights,
                    total_energy + current_g,
                    int(round(current_g)),
                )
                if not np.isfinite(transition_cost):
                    reject_counter[reason] += 1
                    continue

                tentative_g = current_g + transition_cost
                if tentative_g >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self.alpha * self.heuristic(neighbor, goal, env_data, current_time)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        return SearchResult(
            path=None,
            stats={
                "nodes_expanded": nodes_expanded,
                "scenario_trace": scenario_trace,
                "reject_counter": dict(reject_counter),
            },
        )

    def search(self, start: GridPoint, goal: GridPoint, env_data_sequence: Sequence[Dict[str, object]], start_time: int = 0):
        full_path = [start]
        current_start = start
        current_time = start_time
        total_energy = 0.0
        stats = {
            "total_nodes_expanded": 0,
            "replan_count": 0,
            "scenario_log": [],
            "segment_summaries": [],
        }

        max_iterations = int(self.config.get("max_replan", len(env_data_sequence) * self.grid_size))
        iterations = 0

        while current_start != goal and iterations < max_iterations:
            env_index = min(current_time, len(env_data_sequence) - 1)
            env_data = env_data_sequence[env_index]
            result = self._single_search(current_start, goal, env_data, current_time, total_energy)
            stats["total_nodes_expanded"] += result.stats["nodes_expanded"]
            stats["scenario_log"].extend(result.stats["scenario_trace"])
            stats["segment_summaries"].append(result.stats)

            if result.path is None or len(result.path) < 2:
                stats["failure_time"] = current_time
                return None, stats

            segment = result.path[1 : 1 + self.replan_interval]
            for node in segment:
                total_energy += euclidean_dist(full_path[-1], node)
                full_path.append(node)
            current_start = full_path[-1]
            current_time += len(segment)
            stats["replan_count"] += 1
            iterations += 1

        if current_start != goal:
            stats["failure_reason"] = "max_replan_reached"
            return None, stats

        stats["path_length"] = len(full_path)
        stats["total_energy_proxy"] = total_energy
        return full_path, stats
