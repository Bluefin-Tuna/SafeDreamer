import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# === CONFIG ===
root_dir = "/coc/flash5/gzollicoffer3/SafeDreamer/HRSSM/HRSSM/logdir/evals_finegrained"
save_folder = "surface_plots"
os.makedirs(save_folder, exist_ok=True)

# Select what to include
selected_tasks = ["carla_four_lane", "carla_right_turn_simple", "carla_stop_sign"]
selected_augs = ["jitter", "occlusion", "chrome", "gaussian", "glare"]  # e.g. ["chrome", "occlusion", "jitter"]
selected_intensities = [0.625, 0.75, 0.875, 1.0, 2.0, 3.0, 4.0]  # you can restrict this list

task_baselines = {
    "carla_four_lane": 1100,
    "carla_right_turn_simple": 950,
    "carla_stop_sign": 1000,
}
# === Regex pattern ===
pattern = re.compile(
    r'^(?P<task>[a-zA-Z0-9_]+)_(?P=task)_(?P<aug>[a-zA-Z0-9]+)_masked_proportion(?P<prop>[0-9.]+)_timestep10_(?P<intensity>[0-9.]+)$'
)

records = []

for folder_name in os.listdir(root_dir):
    match = pattern.match(folder_name)
    if not match:
        continue

    task, aug = match["task"], match["aug"]
    proportion = float(match["prop"])
    intensity = float(match["intensity"])

    # Apply filters
    if selected_tasks and task not in selected_tasks:
        continue
    if selected_augs and aug not in selected_augs:
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

        # You can average over all scores, or just take the last one:
        score_values = []
        for s in scores:
            val = s.get("episode/score") or s.get("score", None)
            if val is not None:
                score_values.append(val)
        if not score_values:
            continue

        # Expected (mean) score across entries
        expected_score = sum(score_values) / len(score_values)

        records.append((task, aug, proportion, intensity, expected_score))



    except Exception as e:
        print(f"Skipping {folder_name}: {e}")

# === Create DataFrame ===
df = pd.DataFrame(records, columns=["task", "aug", "proportion", "intensity", "score"])

# === Add baseline edges (proportion=0 or intensity=0) for each task & aug ===
for task, base_score in task_baselines.items():
    for aug in df["aug"].unique():
        if task not in selected_tasks:
            continue

        # Get existing values for this (task, aug)
        subset = df[(df["task"] == task) & (df["aug"] == aug)]
        if subset.empty:
            continue

        # Unique intensity and proportion levels in data
        intensities = sorted(subset["intensity"].unique())
        proportions = sorted(subset["proportion"].unique())

        # Create baseline points: one full row (proportion=0) and one full column (intensity=0)
        baseline_rows = []
        for intensity in intensities:
            baseline_rows.append({
                "task": task,
                "aug": aug,
                "proportion": 0.0,
                "intensity": intensity,
                "score": base_score
            })
        for proportion in proportions:
            baseline_rows.append({
                "task": task,
                "aug": aug,
                "proportion": proportion,
                "intensity": 0.0,
                "score": base_score
            })

        df = pd.concat([df, pd.DataFrame(baseline_rows)], ignore_index=True)

if df.empty:
    raise RuntimeError("No matching data found — check your filters or folder pattern!")

print(f"✅ Loaded {len(df)} valid records")

def plot_surface_discrete(df, task_name, aug_name, save_folder):
    subset = df[(df["task"] == task_name) & (df["aug"] == aug_name)]
    if subset.empty:
        print(f"⚠️ No data for {task_name} / {aug_name}, skipping.")
        return

    # Sort unique levels (with baseline if present)
    intensity_levels = sorted(subset["intensity"].unique())
    proportion_levels = sorted(subset["proportion"].unique(), reverse=True)

    # Build pivot table
    pivot = subset.pivot_table(index='proportion', columns='intensity', values='score', aggfunc='mean')

    # Ensure all levels exist (fill missing cells as NaN)
    pivot = pivot.reindex(index=proportion_levels, columns=intensity_levels)

    Z = pivot.values
    # Use integer index positions for even spacing
    X_idx = np.arange(len(intensity_levels))
    Y_idx = np.arange(len(proportion_levels))
    X, Y = np.meshgrid(X_idx, Y_idx)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot using index-based coordinates
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

    # Set custom tick labels for discrete positions
    ax.set_xticks(X_idx)
    ax.set_xticklabels([f"{v:g}" for v in intensity_levels])
    ax.set_yticks(Y_idx)
    ax.set_yticklabels([f"{v:g}" for v in proportion_levels])

    ax.set_xlabel("Intensity")
    ax.set_ylabel("Proportion")
    ax.set_zlabel("Score")
    ax.set_title(f"{task_name} - {aug_name}")
    # ax.invert_yaxis()  # keep flipped direction
    fig.colorbar(surf, shrink=0.5, aspect=5)

    out_path = os.path.join(save_folder, f"{task_name}_{aug_name}_surface.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"💾 Saved: {out_path}")



# === Generate plots ===
for task in df["task"].unique():
    for aug in df["aug"].unique():
        plot_surface_discrete(df, task, aug, save_folder)
