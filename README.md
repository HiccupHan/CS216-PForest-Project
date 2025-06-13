# CS216 PForest Project
Based on the paper "p-forest: In network inference with random forests"

Download the [Network Intrusion dataset(CIC-IDS- 2017)]{https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset} and place it in a data folder before running the code.

framework.py is the main entry point. Run python framework.py to generate the models, rmt simulator inputs, and the tcam rules in a readable json format. model.py contains code that trains the random forests. optimize_pforest.py contains code that compresses pforest implementation, and plain_pforest.py contains the vanilla implementation of the pforest. visualize.py generates images for a sample decision tree from both methods, runs sim.py (rmt simulator) provided by the TA for the decision trees, and the simulated CRAM resource table for random forests (this is due to the rf being too big to actually fit in the simulator setup)

## RMT Simulator: Sample simulator code for an ideal RMT chip

Run `python sim.py <filename>` to execute the script.

Every JSON entry must contain the fields: id, step, and match.

A match kind of "ternary" also requires entries and key_size fields. Note that we omit data_size for "ternary" because the amount of SRAM used to support the TCAM lookup is often negligible.

A match kind of "exact" also requires a method field. The method field must be either "index" or "hash." 

A "hash" method "exact" entry also requires the fields: entries, key_size, and data_size. The key_size and data_size are concatenated together and hashed. Assume a constant load factor of 0.8 (1.25 memory penalty).

An "index" method "exact" entry also requires the fields: key_size and data_size. Note that this lacks an entries field. An entries field is not needed because with a direct-indexed array, the key of length n is used to index into a 2^n array.
