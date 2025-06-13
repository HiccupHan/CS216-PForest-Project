import os
import json
import numpy as np

# Recursively check if a subtree rooted at `node` has the same label.
def check_same_subtree(tree, node):
    if tree.children_left[node] == tree.children_right[node]:  # leaf
        return np.argmax(tree.value[node])
    left = check_same_subtree(tree, tree.children_left[node])
    right = check_same_subtree(tree, tree.children_right[node])
    return left if left == right else None

def compress_paths_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    children_left = tree_.children_left
    children_right = tree_.children_right

    rules = []

    def recurse(node, path):
        same_label = check_same_subtree(tree_, node)
        if same_label is not None:
            class_val = tree.classes_[same_label]
            rules.append({"path": list(path), "label": str(class_val)})
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

def optimize_and_export_rules(rules, context_len, tag, tcam_dir, rmt_dir):
    os.makedirs(tcam_dir, exist_ok=True)
    os.makedirs(rmt_dir, exist_ok=True)

    seen = set()
    optimized_rules = []
    for idx, rule in enumerate(rules):
        rule_id = f"ctx{context_len}_{tag}_rule{idx}"
        condition_str = str(rule["path"])
        if condition_str not in seen:
            seen.add(condition_str)
            rule["id"] = rule_id
            optimized_rules.append(rule)

    # Save TCAM-style rules
    tcam_path = os.path.join(tcam_dir, f"tcam_{tag}_ctx{context_len}.json")
    with open(tcam_path, 'w') as f:
        json.dump(optimized_rules, f, indent=2)

    # Save RMT simulator input
    used_steps = sorted(set(len(rule["path"]) - 1 for rule in optimized_rules))
    step_map = {s: i for i, s in enumerate(used_steps)}

    rmt_entries = []
    for rule in optimized_rules:
        original_step = len(rule["path"]) - 1
        rmt_entries.append({
            "id": rule["id"],
            "step": step_map[original_step],
            "match": "ternary",
            "key_size": 44,
            "entries": 1
        })

    rmt_path = os.path.join(rmt_dir, f"rmt_{tag}_ctx{context_len}.json")
    with open(rmt_path, 'w') as f:
        json.dump(rmt_entries, f, indent=2)

    print(f"Exported {len(optimized_rules)} early-stopped rules to {tcam_path} and {rmt_path}")
    return optimized_rules

def export_decision_tree_optimized(model, context_len, feature_names, tcam_dir, rmt_dir):
    tree = model.estimators_[0]
    rules = compress_paths_to_rules(tree, feature_names)
    return optimize_and_export_rules(rules, context_len, f"tree_opt", tcam_dir, rmt_dir)

def export_random_forest_optimized(model, context_len, feature_names, tcam_dir, rmt_dir):
    all_rules = []
    for i, tree in enumerate(model.estimators_):
        tree_rules = compress_paths_to_rules(tree, feature_names)
        for rule in tree_rules:
            rule["tree"] = i
        all_rules.extend(tree_rules)

    return optimize_and_export_rules(all_rules, context_len, f"forest_opt", tcam_dir, rmt_dir)
