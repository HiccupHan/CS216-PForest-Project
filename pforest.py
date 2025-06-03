import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict

# === Configuration ===
MIN_F1_SCORE = 0.85  # τs
CERTAINTY_THRESHOLD = 0.9  # τc
PACKET_LIMITS = [3, 5, 10]  # different subflow lengths

# === Placeholder for feature extraction ===
def extract_features(df, n_packets):
    # Aggregate the first n_packets for each flow (by FlowID)
    features = []
    labels = []
    grouped = df.groupby('FlowID')
    for _, flow in grouped:
        if len(flow) >= n_packets:
            subflow = flow.iloc[:n_packets]
            feat = {
                'avg_pkt_len': subflow['PacketLength'].mean(),
                'min_pkt_len': subflow['PacketLength'].min(),
                'max_pkt_len': subflow['PacketLength'].max(),
                'pkt_count': len(subflow),
                'duration': subflow['Timestamp'].max() - subflow['Timestamp'].min(),
                'src_port': subflow['SourcePort'].iloc[0],
                'dst_port': subflow['DestinationPort'].iloc[0]
            }
            features.append(feat)
            labels.append(flow['Label'].iloc[0])  # assumed to be constant per flow
    return pd.DataFrame(features), labels

# === Train RF for a specific packet count ===
def train_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=16, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    return clf if score >= MIN_F1_SCORE else None, score

# === Simulate context-dependent training ===
def build_context_rf_models(df):
    models = {}
    for n in PACKET_LIMITS:
        print(f"Training model for first {n} packets...")
        X, y = extract_features(df, n)
        model, score = train_rf(X, y)
        if model:
            models[n] = (model, score)
            print(f"  ✅ Model accepted: F1 = {score:.3f}")
        else:
            print(f"  ❌ Model rejected: F1 < τs")
    return models

# === Main driver ===
def main():
    # Simulated or preprocessed flow-level dataset
    df = pd.read_csv("cicids_sample.csv")  # Needs 'FlowID', 'PacketLength', 'Timestamp', 'SourcePort', 'DestinationPort', 'Label'
    models = build_context_rf_models(df)

    # Save models or simulate usage
    for n, (model, score) in models.items():
        print(f"RF_{n} ready with F1 = {score:.3f}")
        # Optionally: joblib.dump(model, f"rf_{n}.pkl")

if __name__ == "__main__":
    main()
