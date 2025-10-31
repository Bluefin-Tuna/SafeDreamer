import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# === CONFIG ===
root_dirs = [
    "/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo/logdir/evals_finegrained",
    "/coc/flash5/gzollicoffer3/SafeDreamer/HRSSM/HRSSM/logdir/evals_finegrained"
]
save_folder = "surface_plots"
os.makedirs(save_folder, exist_ok=True)

# Select what to include
selected_methods = ["filter", "reject5", "masked", "None"]  # Added 'masked'
selected_tasks = ["carla_four_lane", "carla_right_turn_simple", "carla_stop_sign"]
selected_augs = ["jitter", "occlusion", "chrome", "gaussian", "glare"]
selected_intensities = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.50, 2.0, 3.0, 4.0, 5.0]

task_baselines = {
    "carla_four_lane": 1100,
    "carla_right_turn_simple": 380,
    "carla_stop_sign": 260,
}

# === Regex patterns ===
# Pattern 1: task_bev_task_aug_[method]_proportion{prop}_timestep10_{intensity}
pattern_bev = re.compile(
    r'^(?P<task>[a-zA-Z0-9_]+)_bev_(?P=task)_(?P<aug>[a-zA-Z0-9]+)(?:_(?P<method>[a-zA-Z0-9]+))?_proportion(?P<prop>[0-9.]+)_timestep10_(?P<intensity>[0-9.]+)$'
)

# Pattern 2: task_task_aug_masked_proportion{prop}_timestep10_{intensity}
pattern_masked = re.compile(
    r'^(?P<task>[a-zA-Z0-9_]+)_(?P=task)_(?P<aug>[a-zA-Z0-9]+)_masked_proportion(?P<prop>[0-9.]+)_timestep10_(?P<intensity>[0-9.]+)$'
)

records = []

for root_dir in root_dirs:
    if not os.path.exists(root_dir):
        print(f"⚠️ Directory not found: {root_dir}")
        continue
    
    print(f"\n📁 Processing directory: {root_dir}")
    
    for folder_name in os.listdir(root_dir):
        # Try pattern_bev first
        match = pattern_bev.match(folder_name)
        if match:
            task = match.group("task")
            aug = match.group("aug")
            method = match.group("method") if match.group("method") else "None"
            proportion = float(match.group("prop"))
            intensity = float(match.group("intensity"))
        else:
            # Try pattern_masked
            match = pattern_masked.match(folder_name)
            if match:
                task = match.group("task")
                aug = match.group("aug")
                method = "masked"  # Set method to 'masked'
                proportion = float(match.group("prop"))
                intensity = float(match.group("intensity"))
            else:
                continue

        # Apply filters
        if selected_tasks and task not in selected_tasks:
            continue
        if selected_augs and aug not in selected_augs:
            continue
        if selected_methods and method not in selected_methods:
            continue
        if selected_intensities and intensity not in selected_intensities:
            continue

        metrics_path = os.path.join(root_dir, folder_name, "metrics.jsonl")
        if not os.path.exists(metrics_path):
            continue

        # Load latest score from metrics.jsonl
        try:
            with open(metrics_path) as f:
                scores = [json.loads(line) for line in f if line.strip()]
            if not scores:
                continue

            score_values = []
            for s in scores:
                val = s.get("episode/score") or s.get("score", None)
                if val is not None:
                    score_values.append(val)
            if not score_values:
                continue

            expected_score = sum(score_values) / len(score_values)
            records.append((task, aug, method, proportion, intensity, expected_score))

        except Exception as e:
            print(f"⚠️ Skipping {folder_name}: {e}")

# === Create DataFrame ===
df = pd.DataFrame(records, columns=["task", "aug", "method", "proportion", "intensity", "score"])

if df.empty:
    raise RuntimeError("No matching data found — check your filters or folder pattern!")

print(f"\n✅ Loaded {len(df)} valid records before baselines")
print(f"Methods found: {sorted(df['method'].unique())}")
print(f"Augs found: {sorted(df['aug'].unique())}")
print(f"Tasks found: {sorted(df['task'].unique())}")

# === Add baseline edges ===
for task, base_score in task_baselines.items():
    for aug in df["aug"].unique():
        for method in df["method"].unique():
            if task not in selected_tasks:
                continue

            subset = df[(df["task"] == task) & (df["aug"] == aug) & (df["method"] == method)]
            if subset.empty:
                continue

            intensities = sorted(subset["intensity"].unique())
            proportions = sorted(subset["proportion"].unique())

            baseline_rows = []
            for intensity in intensities:
                baseline_rows.append({
                    "task": task,
                    "aug": aug,
                    "method": method,
                    "proportion": 0.0,
                    "intensity": intensity,
                    "score": base_score
                })
            for proportion in proportions:
                baseline_rows.append({
                    "task": task,
                    "aug": aug,
                    "method": method,
                    "proportion": proportion,
                    "intensity": 0.0,
                    "score": base_score
                })

            df = pd.concat([df, pd.DataFrame(baseline_rows)], ignore_index=True)

print(f"✅ Total records after baselines: {len(df)}")

def plot_surface_discrete(df, task_name, aug_name, method_name, save_folder, z_limits=None):
    subset = df[(df["task"] == task_name) & (df["aug"] == aug_name) & (df["method"] == method_name)]
    if subset.empty:
        print(f"⚠️ No data for {task_name} / {aug_name} / {method_name}, skipping.")
        return

    intensity_levels = sorted(subset["intensity"].unique())
    proportion_levels = sorted(subset["proportion"].unique(), reverse=True)

    pivot = subset.pivot_table(index='proportion', columns='intensity', values='score', aggfunc='mean')
    pivot = pivot.reindex(index=proportion_levels, columns=intensity_levels)

    Z = pivot.values
    X_idx = np.arange(len(intensity_levels))
    Y_idx = np.arange(len(proportion_levels))
    X, Y = np.meshgrid(X_idx, Y_idx)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Set vmin and vmax for consistent colorbar if z_limits provided
    if z_limits:
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', 
                               vmin=z_limits[0], vmax=z_limits[1])
        ax.set_zlim(z_limits[0], z_limits[1])
    else:
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

    ax.set_xticks(X_idx)
    ax.set_xticklabels([f"{v:g}" for v in intensity_levels])
    ax.set_yticks(Y_idx)
    ax.set_yticklabels([f"{v:g}" for v in proportion_levels])

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Proportion")
    ax.set_zlabel("Score")
    ax.set_title(f"{task_name} - {aug_name} - {method_name}")
    
    fig.colorbar(surf, shrink=0.5, aspect=5)

    out_path = os.path.join(save_folder, f"{task_name}_{aug_name}_{method_name}_surface.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"💾 Saved: {out_path}")

# === Generate plots with consistent z-axis per (task, aug) ===
for task in df["task"].unique():
    for aug in df["aug"].unique():
        # Get all data for this task and aug across all methods
        task_aug_subset = df[(df["task"] == task) & (df["aug"] == aug)]
        
        if task_aug_subset.empty:
            continue
        
        # Calculate global min/max for z-axis
        z_min = task_aug_subset["score"].min()
        z_max = task_aug_subset["score"].max()
        z_limits = (z_min, z_max)
        
        print(f"📊 {task} - {aug}: Z-axis range [{z_min:.2f}, {z_max:.2f}]")
        
        # Plot each method with the same z-axis limits
        for method in df["method"].unique():
            plot_surface_discrete(df, task, aug, method, save_folder, z_limits=z_limits)