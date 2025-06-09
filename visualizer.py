import os
import json
import matplotlib.pyplot as plt
import networkx as nx

TCAM_DIR = "./tcam_rules"


def load_tcam_rules(context_len):
    path = os.path.join(TCAM_DIR, f"tcam_ctx{context_len}.json")
    with open(path, 'r') as f:
        return json.load(f)


def visualize_decision_tree(tcam_rules, context_len):
    G = nx.DiGraph()
    node_id = 0

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
    plt.title(f"Decision Tree Visualization (Context={context_len})")
    plt.tight_layout()
    plt.show()


def generate_cram_resource_table(tcam_rules):
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
        "# Pipeline Steps (CRAM)": max_depth,
        "Approx TCAM Entries": len(tcam_rules) * max_depth,
        "SRAM Bits (Label Storage)": len(tcam_rules) * 16
    }


def print_resource_table(res):
    print("\nCRAM Resource Summary:")
    for k, v in res.items():
        print(f"{k:30}: {v}")


if __name__ == "__main__":
    context_len = 3  # or 5 or 10
    tcam_rules = load_tcam_rules(context_len)
    visualize_decision_tree(tcam_rules, context_len)
    res = generate_cram_resource_table(tcam_rules)
    print_resource_table(res)
