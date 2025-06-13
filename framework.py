from model import load_and_prepare_data, train_rf, simulate_asap_inference
from plain_pforest import export_forest, export_decision_tree
from optimize_pforest import export_decision_tree_optimized, export_random_forest_optimized
import os
import joblib
import pandas as pd

MIN_F1_SCORE = 0.85
CERTAINTY_THRESHOLD = 0.9
CONTEXT_LENGTHS = [3, 5, 10]  # Packet counts
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

DATA_DIR = "./data"
MODEL_DIR = "./models"
TCAM_DIR = "./tcam_rules"
RMT_DIR = "./rmt_input"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TCAM_DIR, exist_ok=True)
os.makedirs(RMT_DIR, exist_ok=True)

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

def main():
    df = load_and_prepare_data(data_dir=DATA_DIR, feature_columns=FEATURE_COLUMNS)
    print(f"Loaded {len(df)} flow records.")

    context_models = {}

    for context_len in CONTEXT_LENGTHS:
        print(f"Training for context length: {context_len}")

        subset_size = int(len(df) * (context_len * 0.1)) # simulate context length by sampling the dataset and train different models
        df_sub = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
        X = df_sub[FEATURE_COLUMNS]
        y = df_sub['Label']

        model, score = train_rf(X, y)
        if model:
            print(f"Model trained for context {context_len} with F1 score: {score:.3f}")
            model_path = os.path.join(MODEL_DIR, f"rf_ctx{context_len}.pkl")
            joblib.dump(model, model_path)
            context_models[context_len] = model
            export_forest(model, context_len, FEATURE_COLUMNS, TCAM_DIR, RMT_DIR)
            export_decision_tree(model, context_len, FEATURE_COLUMNS, TCAM_DIR, RMT_DIR)
            export_decision_tree_optimized(model, context_len, FEATURE_COLUMNS, TCAM_DIR, RMT_DIR)
            export_random_forest_optimized(model, context_len, FEATURE_COLUMNS, TCAM_DIR, RMT_DIR)
        else:
            print(f"Model rejected for context {context_len} (F1 < {MIN_F1_SCORE})")

    print("\nRunning simulated ASAP inference on sample flows:")
    sample_flows = df.sample(n=5)
    for idx, row in sample_flows.iterrows():
        features_dict = {n: pd.DataFrame([row[FEATURE_COLUMNS]]) for n in CONTEXT_LENGTHS}
        label, certainty, ctx = simulate_asap_inference(features_dict, context_models)
        label_str = 'MALICIOUS' if label == 1 else 'BENIGN'
        print(f"Flow {idx}: Predicted={label_str}, Certainty={certainty:.2f}, Context={ctx}, Real={row['Label']}")

if __name__ == "__main__":
    main()