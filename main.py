from __future__ import annotations

from pathlib import Path

from data.generate_3d_grid_data import MapConfig, generate_ocean_map, summarize_environment
from planner import MultiConstraintAStar, compute_path_metrics


def run_demo():
    config = MapConfig()
    save_path = Path("data") / "ocean_dynamic_map.npy"
    env_data_sequence, config = generate_ocean_map(config=config, save_path=save_path)

    planner = MultiConstraintAStar(config.to_search_config())
    path, stats = planner.search(config.start, config.goal, env_data_sequence)

    print("=== Environment Summary ===")
    for key, value in summarize_environment(env_data_sequence, config).items():
        print(f"{key}: {value:.4f}")

    print("\n=== Planning Summary ===")
    if path is None:
        print("Path planning failed.")
        print(stats)
        return

    metrics = compute_path_metrics(path, env_data_sequence, config.to_search_config(), stats)
    print(f"path_nodes: {len(path)}")
    print(f"start: {config.start}")
    print(f"goal: {config.goal}")
    print(f"replans: {stats['replan_count']}")
    print(f"nodes_expanded: {stats['total_nodes_expanded']}")

    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    run_demo()
