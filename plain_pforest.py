import os
import json

def parse_tree_to_rules(tree, feature_names):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    children_left = tree_.children_left
    children_right = tree_.children_right

    rules = []

    def recurse(node, path):
        if children_left[node] == children_right[node]:
            class_val = tree.classes_[tree_.value[node].argmax()]
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

def export_forest(model, context_len, feature_names, tcam_dir, rmt_dir):
    forest_rules = []
    for i, tree in enumerate(model.estimators_):
        rules = parse_tree_to_rules(tree, feature_names)
        for rule_idx, rule in enumerate(rules):
            rule["id"] = f"ctx{context_len}_tree{i}_rule{rule_idx}"
            rule["tree_id"] = i
        forest_rules.extend(rules)

    # Save TCAM-style rule paths
    tcam_path = os.path.join(tcam_dir, f"tcam_forest_ctx{context_len}.json")
    with open(tcam_path, "w") as f:
        json.dump(forest_rules, f, indent=2)

    # Save RMT-style match-action input per rule as independent TCAM rules
    rmt_entries = []
    used_steps = sorted(set(len(rule["path"]) - 1 for rule in forest_rules))
    step_map = {s: i for i, s in enumerate(used_steps)}
    for rule in forest_rules:
        original_step = len(rule["path"]) - 1
        rmt_entries.append({
            "id": rule["id"],
            "step": step_map[original_step], 
            "match": "ternary",
            "key_size": 44,
            "entries": 1
        })

    rmt_path = os.path.join(rmt_dir, f"rmt_forest_ctx{context_len}.json")
    with open(rmt_path, "w") as f:
        json.dump(rmt_entries, f, indent=2)

    return tcam_path, rmt_path

def export_decision_tree(model, context_len, feature_names, tcam_dir, rmt_dir):
    tree = model.estimators_[0]
    tcam_rules = parse_tree_to_rules(tree, feature_names)
    json_path = os.path.join(tcam_dir, f"tcam_tree_ctx{context_len}.json")
    with open(json_path, 'w') as f:
        json.dump(tcam_rules, f, indent=2)
    print(f"Exported TCAM rules for context {context_len} to {json_path}")
    # Generate RMT input format
    rmt_entries = []
    used_steps = sorted(set(len(rule["path"]) - 1 for rule in tcam_rules))
    step_map = {s: i for i, s in enumerate(used_steps)}
    for idx, rule in enumerate(tcam_rules):
        original_step = len(rule["path"]) - 1
        rmt_entries.append({
            "id": f"ctx{context_len}_rule_{idx}",
            "step": step_map[original_step],
            "match": "ternary",
            "key_size": 44,
            "entries": 1
        })
    rmt_path = os.path.join(rmt_dir, f"rmt_tree_ctx{context_len}.json")
    with open(rmt_path, 'w') as f:
        json.dump(rmt_entries, f, indent=2)
    print(f"Exported RMT input for context {context_len} to {rmt_path}")
