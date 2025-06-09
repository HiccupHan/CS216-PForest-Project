import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from scipy import stats
import joblib
import json

# Configuration
MIN_F1_SCORE = 0.85
CERTAINTY_THRESHOLD = 0.9
CONTEXT_LENGTHS = [3, 5, 10]  # Packet counts
DATA_DIR = "./data"
MODEL_DIR = "./models"
RULES_DIR = "./rules"
TCAM_DIR = "./tcam_rules"
RMT_DIR = "./rmt_input"

FEATURE_COLUMNS = [
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'SYN Flag Count',
    'ACK Flag Count'
]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RULES_DIR, exist_ok=True)
os.makedirs(TCAM_DIR, exist_ok=True)
os.makedirs(RMT_DIR, exist_ok=True)

# Load and clean dataset
def load_and_prepare_data():
    selected_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv"
    ]
    dataframes = []
    for file in selected_files:
        path = os.path.join(DATA_DIR, file)
        print(f"Reading {file}...")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue

        df.columns = df.columns.str.strip()
        required_cols = FEATURE_COLUMNS + ['Label']
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
def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=16, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    return clf if score >= MIN_F1_SCORE else None, score

# Compute certainty of prediction
def compute_certainty(model, X):
    X_array = X.values if hasattr(X, 'values') else X
    tree_preds = np.array([tree.predict(X_array) for tree in model.estimators_])
    pred_labels, counts = stats.mode(tree_preds, axis=0, keepdims=True)
    certainty = counts[0] / len(model.estimators_)
    return pred_labels[0], certainty[0]

# Simulate ASAP inference for a single flow
def simulate_asap_inference(flow_features_dict, context_models):
    for n in sorted(context_models):
        model = context_models[n]
        features = flow_features_dict[n]
        label, certainty = compute_certainty(model, features)
        if certainty >= CERTAINTY_THRESHOLD:
            return label, certainty, n
    return 'UNKNOWN', 0.0, None

# Export a single decision tree to text format and match-action JSON
def export_decision_tree(model, context_len, feature_names):
    tree = model.estimators_[0]
    tree_rules = export_text(tree, feature_names=feature_names)
    rules_path = os.path.join(RULES_DIR, f"tree_rules_ctx{context_len}.txt")
    with open(rules_path, 'w') as f:
        f.write(tree_rules)
    print(f"Exported tree rules for context {context_len} to {rules_path}")

    def parse_tree_to_rules(tree):
        tree_ = tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold
        children_left = tree_.children_left
        children_right = tree_.children_right

        rules = []

        def recurse(node, path):
            if children_left[node] == children_right[node]:
                class_val = tree.classes_[np.argmax(tree_.value[node])]
                rules.append({"path": path.copy(), "label": str(class_val)})
            else:
                fname = feature_names[feature[node]]
                thresh = threshold[node]
                path.append((fname, "<=", thresh))
                recurse(children_left[node], path)
                path.pop()
                path.append((fname, ">", thresh))
                recurse(children_right[node], path)
                path.pop()

        recurse(0, [])
        return rules

    tcam_rules = parse_tree_to_rules(tree)
    json_path = os.path.join(TCAM_DIR, f"tcam_ctx{context_len}.json")
    with open(json_path, 'w') as f:
        json.dump(tcam_rules, f, indent=2)
    print(f"Exported TCAM rules for context {context_len} to {json_path}")

    # Generate RMT input format
    rmt_entries = []
    for idx, rule in enumerate(tcam_rules):
        rmt_entries.append({
            "id": f"ctx{context_len}_rule_{idx}",
            "step": 0,
            "match": "ternary",
            "key_size": 44,
            "entries": 1
        })
    rmt_path = os.path.join(RMT_DIR, f"rmt_input_ctx{context_len}.json")
    with open(rmt_path, 'w') as f:
        json.dump(rmt_entries, f, indent=2)
    print(f"Exported RMT input for context {context_len} to {rmt_path}")

# Main pipeline
def main():
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} flow records.")

    context_models = {}

    for context_len in CONTEXT_LENGTHS:
        print(f"Training for context length: {context_len}")

        df_sub = df.copy()
        X = df_sub[FEATURE_COLUMNS]
        y = df_sub['Label']

        model, score = train_rf(X, y)
        if model:
            print(f"Model trained for context {context_len} with F1 score: {score:.3f}")
            model_path = os.path.join(MODEL_DIR, f"rf_ctx{context_len}.pkl")
            joblib.dump(model, model_path)
            context_models[context_len] = model
            export_decision_tree(model, context_len, FEATURE_COLUMNS)
        else:
            print(f"Model rejected for context {context_len} (F1 < {MIN_F1_SCORE})")

    print("\nRunning simulated ASAP inference on sample flows:")
    sample_flows = df.sample(n=5)
    for idx, row in sample_flows.iterrows():
        features_dict = {n: pd.DataFrame([row[FEATURE_COLUMNS]]) for n in CONTEXT_LENGTHS}
        label, certainty, ctx = simulate_asap_inference(features_dict, context_models)
        label_str = 'MALICIOUS' if label == 1 else 'BENIGN'
        print(f"Flow {idx}: Predicted={label_str}, Certainty={certainty:.2f}, Context={ctx}")

if __name__ == "__main__":
    main()