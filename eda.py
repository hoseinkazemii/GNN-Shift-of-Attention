#######################################################################
# ---------------------- EDA ---------------------------------------
#########################################################################
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_powerline_attention_timeline(df, participant_label, output_dir="plots"):
    ensure_dir(output_dir)
    plt.figure(figsize=(12, 3))
    plt.plot(df["Timeframe"], df["label_powerline_future_1p5s"], label='Future Attention to Powerline')
    plt.xlabel("Time (s)")
    plt.ylabel("Powerline Label (next 1.5s)")
    plt.title(f"Powerline Future Attention Timeline — {participant_label}")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{participant_label}_powerline_timeline.png")
    plt.savefig(filename)
    plt.show()

def plot_distance_to_powerline_timeline(df, participant_label, output_dir="plots"):
    ensure_dir(output_dir)
    plt.figure(figsize=(12, 3))
    plt.plot(df["Timeframe"], df["Min Distance to Powerline (Grabbable Object)"], label="Distance to Powerline")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title(f"Distance to Powerline vs. Time — {participant_label}")
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{participant_label}_distance_to_powerline.png")
    plt.savefig(filename)
    plt.show()

def plot_gaze_heatmap(df, participant_label, output_dir="plots"):
    ensure_dir(output_dir)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=df["Gaze Position X"],
        y=df["Gaze Position Z"],
        hue=df["Name"],
        alpha=0.6,
        s=10,
        palette="tab10",
        legend=False
    )
    plt.xlabel("Gaze X")
    plt.ylabel("Gaze Z")
    plt.title(f"Gaze Position Heatmap (XZ) — {participant_label}")
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{participant_label}_gaze_heatmap.png")
    plt.savefig(filename)
    plt.show()

def plot_fixation_durations(df, participant_label, output_dir="plots"):
    ensure_dir(output_dir)
    durations = []
    current_obj = None
    start_time = None

    for idx, row in df.iterrows():
        obj = row["Name"]
        time = row["Timeframe"]
        
        if obj != current_obj:
            if current_obj is not None:
                durations.append((current_obj, time - start_time))
            current_obj = obj
            start_time = time

    fixation_df = pd.DataFrame(durations, columns=["Object", "Fixation Duration (s)"])
    fixation_summary = fixation_df.groupby("Object")["Fixation Duration (s)"].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    fixation_summary.plot(kind="bar")
    plt.ylabel("Avg Fixation Duration (s)")
    plt.title(f"Average Fixation Duration per Object — {participant_label}")
    plt.grid(axis="y")
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{participant_label}_fixation_durations.png")
    plt.savefig(filename)
    plt.show()

def plot_transition_counts_to_powerline(df, participant_label, powerline_name="PowerLine1", output_dir="plots"):
    ensure_dir(output_dir)
    transitions = []

    name_series = df["Name"].astype(str).values
    for i in range(1, len(name_series)):
        prev_obj = name_series[i-1]
        current_obj = name_series[i]
        if current_obj == powerline_name and prev_obj != powerline_name:
            transitions.append(prev_obj)

    transition_counts = Counter(transitions)
    if not transition_counts:
        print("No transitions to powerline found.")
        return

    labels, values = zip(*transition_counts.items())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel("Count")
    plt.title(f"Fixation Transitions to Powerline — {participant_label}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{participant_label}_transitions_to_powerline.png")
    plt.savefig(filename)
    plt.show()



# Load the combined labeled data (already preprocessed and labeled)
df_combined = pd.read_csv("combined_labeled_data.csv")
df_one = df_combined[df_combined["source_file"] == "VREyeTracking_PowerLineScenario_20241219155732.csv"]
# print(df_one.head())
# print(df_one.columns)

# Plotting for the first participant
plot_powerline_attention_timeline(df_one, "Participant 5")
plot_distance_to_powerline_timeline(df_one, "Participant 5")
plot_gaze_heatmap(df_one, "Participant 5")
plot_fixation_durations(df_one, "Participant 5")
plot_transition_counts_to_powerline(df_one, "Participant 5")