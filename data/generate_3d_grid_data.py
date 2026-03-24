from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


GridPoint = Tuple[int, int, int]


@dataclass
class MapConfig:
    grid_size: int = 20
    time_steps: int = 12
    terrain_density: float = 0.012
    terrain_inflation_radius: int = 2
    safe_clearance_margin: int = 2
    signal_warning_threshold: float = 0.30
    signal_critical_threshold: float = 0.05
    hard_obstacle_radius: float = 1.0
    warning_obstacle_radius: float = 4.0
    max_turn_angle_deg: float = 120.0
    current_speed_scale: float = 1.5
    vertical_current_scale: float = 0.3
    current_speed_max: float = 2.0
    energy_budget: float = 120.0
    home_point: GridPoint = (1, 1, 1)
    acoustic_beacon: GridPoint = (5, 5, 2)
    beacon_range: float = 15.0
    perception_range: int = 5
    replan_interval: int = 3
    alpha: float = 1.2
    beta_strength: float = 0.3
    lambda_obs: float = 0.8
    safety_priority: float = 2.0
    dynamic_obstacle_count: int = 10
    random_seed: int = 7
    start: GridPoint = (1, 1, 1)
    goal: GridPoint = (18, 18, 18)
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def d_max(self) -> int:
        return self.grid_size * 3

    @property
    def max_turn_angle_rad(self) -> float:
        return np.deg2rad(self.max_turn_angle_deg)

    def to_search_config(self) -> Dict[str, object]:
        config = asdict(self)
        config["S_th"] = self.signal_warning_threshold
        config["S_crit"] = self.signal_critical_threshold
        config["R_hard"] = self.hard_obstacle_radius
        config["R_warn"] = self.warning_obstacle_radius
        config["v_current_max"] = self.current_speed_max
        config["max_turn_angle"] = self.max_turn_angle_rad
        config["D_max"] = self.d_max
        return config


def euclidean_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _inflate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()

    inflated = mask.copy()
    occupied = np.argwhere(mask == 1)
    grid_size = mask.shape[0]

    for x, y, z in occupied:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > radius * radius:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                        inflated[nx, ny, nz] = 1
    return inflated


def _clear_points(terrain: np.ndarray, points: Sequence[GridPoint], margin: int) -> None:
    grid_size = terrain.shape[0]
    for x, y, z in points:
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                for dz in range(-margin, margin + 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                        terrain[nx, ny, nz] = 0


def generate_static_terrain(config: MapConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    terrain = (
        rng.random((config.grid_size, config.grid_size, config.grid_size)) < config.terrain_density
    ).astype(np.uint8)
    inflated = _inflate_binary_mask(terrain, config.terrain_inflation_radius)
    _clear_points(
        inflated,
        [config.start, config.goal, config.home_point, config.acoustic_beacon],
        config.safe_clearance_margin,
    )
    return inflated


def generate_dynamic_current(config: MapConfig, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gs = config.grid_size
    x = np.arange(gs)
    y = np.arange(gs)
    z = np.arange(gs)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    u = config.current_speed_scale * np.sin(0.2 * X + 0.12 * t)
    v = config.current_speed_scale * np.cos(0.18 * Y + 0.09 * t)
    w = config.vertical_current_scale * np.sin(0.15 * Z + 0.06 * t)
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)


def generate_dynamic_obstacles(config: MapConfig, t: int) -> List[GridPoint]:
    gs = config.grid_size
    margin = int(np.ceil(config.warning_obstacle_radius)) + 1
    rng = np.random.default_rng(config.random_seed + 101)

    bases = []
    for _ in range(config.dynamic_obstacle_count):
        bases.append(
            (
                int(rng.integers(margin, gs - margin)),
                int(rng.integers(margin, gs - margin)),
                int(rng.integers(margin, gs - margin)),
            )
        )

    obstacles: List[GridPoint] = []
    for index, (bx, by, bz) in enumerate(bases):
        ox = int(np.clip(round(bx + 2 * np.sin(0.30 * t + index)), 0, gs - 1))
        oy = int(np.clip(round(by + 2 * np.cos(0.22 * t + 0.5 * index)), 0, gs - 1))
        oz = int(np.clip(round(bz + np.sin(0.12 * t + index)), 0, gs - 1))
        point = (ox, oy, oz)
        if euclidean_dist(point, config.start) > config.safe_clearance_margin and euclidean_dist(
            point, config.goal
        ) > config.safe_clearance_margin:
            obstacles.append(point)

    return obstacles


def generate_signal_field(config: MapConfig) -> np.ndarray:
    gs = config.grid_size
    cx, cy, cz = config.acoustic_beacon
    x = np.arange(gs)
    y = np.arange(gs)
    z = np.arange(gs)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    signal = np.exp(-0.08 * dist) * np.maximum(1 - 0.03 * Z, 0.0)
    return np.clip(signal, 0.0, 1.0).astype(np.float32)


def build_step_environment(
    t: int,
    terrain: np.ndarray,
    signal_field: np.ndarray,
    current_field: Tuple[np.ndarray, np.ndarray, np.ndarray],
    obstacles_t: List[GridPoint],
) -> Dict[str, object]:
    return {
        "time_step": t,
        "terrain": terrain,
        "current_field": current_field,
        "obstacles_t": obstacles_t,
        "signal_field": signal_field,
    }


def generate_ocean_map(config: MapConfig | None = None, save_path: str | Path | None = None):
    config = config or MapConfig()
    terrain = generate_static_terrain(config)
    signal_field = generate_signal_field(config)

    env_data_sequence: List[Dict[str, object]] = []
    for t in range(config.time_steps):
        current_field = generate_dynamic_current(config, t)
        obstacles_t = generate_dynamic_obstacles(config, t)
        env_data_sequence.append(
            build_step_environment(t, terrain, signal_field, current_field, obstacles_t)
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, np.asarray(env_data_sequence, dtype=object), allow_pickle=True)

    return env_data_sequence, config


def summarize_environment(env_data_sequence: Sequence[Dict[str, object]], config: MapConfig) -> Dict[str, float]:
    signal_field = env_data_sequence[0]["signal_field"]
    terrain = env_data_sequence[0]["terrain"]
    return {
        "grid_size": float(config.grid_size),
        "time_steps": float(config.time_steps),
        "terrain_occupancy": float(np.mean(terrain)),
        "signal_min": float(np.min(signal_field)),
        "signal_max": float(np.max(signal_field)),
        "signal_below_warning_ratio": float(np.mean(signal_field < config.signal_warning_threshold)),
        "dynamic_obstacle_count": float(len(env_data_sequence[0]["obstacles_t"])),
    }


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "ocean_dynamic_map.npy"
    env_data_sequence, config = generate_ocean_map(save_path=output_path)
    summary = summarize_environment(env_data_sequence, config)

    print("Dynamic ocean map generated.")
    print(f"Saved to: {output_path}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
