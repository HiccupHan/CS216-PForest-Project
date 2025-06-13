import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import math
TCAM_DIR = "./tcam_rules"
VIS_DIR = "./tree_images"

os.makedirs(VIS_DIR, exist_ok=True)

def load_tcam_rules(file_name):
    path = os.path.join(TCAM_DIR, file_name)
    with open(path, 'r') as f:
        return json.load(f)

def visualize_decision_tree(tcam_rules, title = f"Decision Tree Visualization"):
    G = nx.DiGraph()
    fig_path = os.path.join(VIS_DIR, title.replace(" ", "_") + ".png")
    for rule_id, rule in enumerate(tcam_rules):
        prev = "root"
        for cond in rule["path"]:
            label = f"{cond[0]} {cond[1]} {round(cond[2], 2)}"
            node = f"{label}_{rule_id}"
            G.add_edge(prev, node)
            prev = node
        G.add_edge(prev, f"label: {rule['label']} [{rule_id}]")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', 
            font_size=9, arrows=True, edge_color='gray')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def generate_forest_cram_resource_table(tcam_rules):
    TCAM_WIDTH = 44
    TCAM_HEIGHT = 512
    LABEL_BITS = 16
    
    total_tcam_blocks = 0
    max_step_used = 0
    active_steps = set()
    unique_features = set()

    for rule in tcam_rules:
        step = rule.get("step", len(rule["path"]) - 1)
        key_size = rule.get("key_size", len(rule["path"]) * TCAM_WIDTH)
        entries = rule.get("entries", 1)

        blocks = math.ceil(key_size / TCAM_WIDTH) * math.ceil(entries / TCAM_HEIGHT)
        total_tcam_blocks += blocks

        max_step_used = max(max_step_used, step)
        active_steps.add(step)

        for cond in rule["path"]:
            unique_features.add(cond[0])

    return {
        "# Rules": len(tcam_rules),
        "# Unique Features Used": len(unique_features),
        "# Pipeline Steps (CRAM)": len(active_steps),
        "Max Step Value Used": max_step_used,
        "Total TCAM Blocks Used": total_tcam_blocks,
        "Total SRAM Bits for Labels": len(tcam_rules) * LABEL_BITS,
    }


def generate_tree_cram_resource_table(tcam_rules):
    max_depth = max(len(rule["path"]) for rule in tcam_rules)
    avg_depth = sum(len(rule["path"]) for rule in tcam_rules) / len(tcam_rules)
    unique_features = set()
    for rule in tcam_rules:
        for cond in rule["path"]:
            unique_features.add(cond[0])

    return {
        "# Rules": len(tcam_rules),
        "Max Tree Depth": max_depth,
        "Avg Tree Depth": round(avg_depth, 2),
        "# Unique Features Used": len(unique_features),
    }

def print_resource_table(table, title="CRAM Resource Summary"):
    print(f"\n{title}:")
    for k, v in table.items():
        print(f"{k:30}: {v}")


if __name__ == "__main__":
    context_lens = [3, 5, 10] # or 5 or 10
    for context_len in context_lens:
        tcam_name = f"tcam_tree_ctx{context_len}.json"
        opt_tcam_name = f"tcam_tree_opt_ctx{context_len}.json"
        forest_name = f"tcam_forest_ctx{context_len}.json"
        opt_forest_name = f"tcam_forest_opt_ctx{context_len}.json"
        tcam_rules = load_tcam_rules(tcam_name)
        forest_rules = load_tcam_rules(forest_name)
        opt_tcam_rules = load_tcam_rules(opt_tcam_name)
        opt_forest_rules = load_tcam_rules(opt_forest_name)
        visualize_decision_tree(tcam_rules, f"Decision Tree Visualization (Context={context_len})")
        visualize_decision_tree(opt_tcam_rules, f"Optimized Decision Tree Visualization (Context={context_len})")
        table = generate_tree_cram_resource_table(tcam_rules)
        print_resource_table(table, f"CRAM Resource Summary For Plain pForest Decision Tree (Context={context_len})")
        table = generate_forest_cram_resource_table(forest_rules)
        print_resource_table(table, f"CRAM Resource Summary For Plain pForest Random Forest (Context={context_len})")
        table = generate_tree_cram_resource_table(opt_tcam_rules)
        print_resource_table(table, f"CRAM Resource Summary For Optimized pForest Decision Tree (Context={context_len})")
        table = generate_forest_cram_resource_table(opt_forest_rules)
        print_resource_table(table, f"CRAM Resource Summary For Optimized pForest Random Forest (Context={context_len})")
