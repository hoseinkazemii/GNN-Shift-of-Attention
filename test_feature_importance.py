import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ----------------------------------------
# CONFIG: Feature combinations to test
# ----------------------------------------
# These are the possible features in your dataset
base_features = [
    'Min Distance to Powerline (Hit Object)',    # dist
    'Powerline and Grabbable X Distance',         # dx
    'Powerline and Grabbable Y Distance',         # dy
    'Powerline and Grabbable Z Distance'          # dz
]

gaze_direction_features = [
    'Gaze Direction X',
    'Gaze Direction Y',
    'Gaze Direction Z'
]

gaze_position_features = [
    'Gaze Position X',
    'Gaze Position Y',
    'Gaze Position Z'
]

# Define feature sets to test
feature_sets_to_test = {
    'Base only (distance features)': base_features,
    'Base + Gaze Direction': base_features + gaze_direction_features,
    'Base + Gaze Position': base_features + gaze_position_features,
    'Base + Gaze Direction + Gaze Position': base_features + gaze_direction_features + gaze_position_features
}

# ----------------------------------------
# Load the critical data (already filtered)
# ----------------------------------------
df = pd.read_csv('combined_labeled_data.csv')

CRITICAL_DIST_THRESHOLD = 50

# Filtering critical data (same as your STGCN pipeline)
df_critical = df[
    (df["Min Distance to Powerline (Grabbable Object)"] < CRITICAL_DIST_THRESHOLD) &
    (df["LoadingStarted"] == 1)
].copy()

# Only keep needed columns
columns_needed = base_features + gaze_direction_features + gaze_position_features + [
    "Name", "Timeframe", "label_powerline_future_1p5s"
]
df_critical = df_critical[columns_needed].copy()

print(f"Critical dataset size: {len(df_critical)} samples")

# ----------------------------------------
# Build simple sequences (no graph)
# ----------------------------------------

# We will group by sequences of length WINDOW_SIZE timesteps
WINDOW_SIZE = 10  # Same as STGCN
delta_t = 0.02
lookahead_seconds = 1.5
lookahead_steps = int(lookahead_seconds / delta_t)

# Sort by timeframe
df_critical = df_critical.sort_values("Timeframe").reset_index(drop=True)
timeframes = df_critical["Timeframe"].unique()

X_all = []
y_all = []

for i in tqdm(range(len(timeframes) - WINDOW_SIZE - lookahead_steps), desc="Sequences"):
    window_times = timeframes[i:i+WINDOW_SIZE]
    label_times = timeframes[i+WINDOW_SIZE:i+WINDOW_SIZE+lookahead_steps]

    window_df = df_critical[df_critical["Timeframe"].isin(window_times)]
    future_df = df_critical[df_critical["Timeframe"].isin(label_times)]

    if len(window_df) != WINDOW_SIZE:
        continue  # Skip if window is incomplete

    # Label: whether 'PowerLine1' appears in future window
    label = int((future_df["Name"] == "PowerLine1").any())

    X_all.append(window_df)
    y_all.append(label)


print(f"Total sequences generated: {len(X_all)}")

# ----------------------------------------
# Function to flatten features for Logistic Regression
# ----------------------------------------
def flatten_features(X_seq, selected_features):
    """
    X_seq: a DataFrame with WINDOW_SIZE timesteps
    selected_features: list of feature names
    Output: 1D flattened feature vector
    """
    features = X_seq[selected_features].values
    flat = features.flatten()
    return flat

# ----------------------------------------
# Main Testing Loop
# ----------------------------------------

for test_name, selected_features in feature_sets_to_test.items():
    print("="*60)
    print(f"Testing Feature Set: {test_name}")
    print(f"Selected features: {selected_features}")
    print("="*60)

    # Flatten X for selected features
    X_flat = np.array([flatten_features(seq, selected_features) for seq in X_all])
    y = np.array(y_all)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, stratify=y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Logistic Regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',   # important because of class imbalance
        solver='lbfgs'
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=["No Powerline", "Powerline"],
        digits=4
    )
    print(report)
    print("\n\n")
