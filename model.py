import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy import stats

# Load and clean dataset
def load_and_prepare_data(data_dir, feature_columns):
    selected_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv"
    ]
    dataframes = []
    for file in selected_files:
        path = os.path.join(data_dir, file)
        print(f"Reading {file}...")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        df.columns = df.columns.str.strip()
        required_cols = feature_columns + ['Label']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file}: missing required columns")
            continue

        df = df[required_cols].dropna()

        df['Label'] = df['Label'].astype(str).apply(
            lambda x: 'BENIGN' if 'BENIGN' in x.upper() else 'MALICIOUS'
        )

        dataframes.append(df)

    if not dataframes:
        raise ValueError("No usable data files found.")

    return pd.concat(dataframes, ignore_index=True)

# Train random forest for a given context length
def train_rf(X, y, min_f1_score=0.85):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=16, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    return clf if score >= min_f1_score else None, score

# Compute certainty of prediction
def compute_certainty(model, X):
    X_array = X.values if hasattr(X, 'values') else X
    tree_preds = np.array([tree.predict(X_array) for tree in model.estimators_])
    pred_labels, counts = stats.mode(tree_preds, axis=0, keepdims=True)
    certainty = counts[0] / len(model.estimators_)
    return pred_labels[0], certainty[0]

# Simulate ASAP inference for a single flow
def simulate_asap_inference(flow_features_dict, context_models, certainty_threshold=0.9):
    for n in sorted(context_models):
        model = context_models[n]
        features = flow_features_dict[n]
        label, certainty = compute_certainty(model, features)
        if certainty >= certainty_threshold:
            return label, certainty, n
    return 'UNKNOWN', 0.0, None
