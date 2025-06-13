import json
import sys
import math

LOAD_FACTOR = 0.8
HASH_MULTIPLIER = 1 / LOAD_FACTOR
STAGES = 20
TCAM_WIDTH = 44
TCAM_HEIGHT = 512
SRAM_WIDTH = 128
SRAM_HEIGHT = 1024
TCAM_BLOCK = TCAM_WIDTH * TCAM_HEIGHT
SRAM_PAGE = SRAM_WIDTH * SRAM_HEIGHT
BLOCKS_PER_STAGE = 24
PAGES_PER_STAGE = 80

def check_mandatory_fields(data):
    for entry in data:
        if "id" not in entry:
            raise ValueError("Missing id field")
        if "step" not in entry:
            raise ValueError("Missing step field")
        if "match" not in entry:
            raise ValueError("Missing match field")
        if "key_size" not in entry:
            raise ValueError("Missing key_size field")

        if not isinstance(entry["id"], str) or len(entry["id"]) <= 0:
            raise ValueError("id must be a non-empty string")
        if not isinstance(entry["step"], int) or entry["step"] < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(entry["match"], str) or ( entry["match"] != "ternary" and entry["match"] != "exact" ):
            raise ValueError("match must be either ternary or exact")
        if not isinstance(entry["key_size"], int) or entry["key_size"] < 0:
            raise ValueError("key_size must be a non-negative integer")
        
        if entry["match"] == "ternary":
            if "entries" not in entry:
                raise ValueError("Missing entries field for ternary type")
            if not isinstance(entry["entries"], int) or entry["entries"] < 1:
                raise ValueError("entries must be a positive integer")
            if "method" in entry:
                raise ValueError("Ternary type should not contain method field")
            if "data_size" in entry:
                raise ValueError("Ternary type should not contain data_size field")
        
        elif entry["match"] == "exact":
            if "method" not in entry:
                raise ValueError("Missing method field for exact type")
            if not isinstance(entry["method"], str) or ( entry["method"] != "index" and entry["method"] != "hash" ):
                raise ValueError("method must be either index or hash")
            
            if "data_size" not in entry:
                raise ValueError("Missing data_size field for exact type")
            if not isinstance(entry["data_size"], int) or entry["data_size"] < 0:
                raise ValueError("data_size must be a non-negative integer")
            
            if entry["method"] == "index":
                if "entries" in entry:
                    raise ValueError("Exact type with index should not contain entries field")
                
            elif entry["method"] == "hash":
                if "entries" not in entry:
                    raise ValueError("Missing entries field for exact type with hash")
                if not isinstance(entry["entries"], int) or entry["entries"] < 1:
                    raise ValueError("entries must be a positive integer")

def check_unique_ids(data):
    ids = [entry["id"] for entry in data]
    if len(ids) != len(set(ids)):
        raise ValueError("All ids must be unique")
    
def check_contiguous_steps(data):
    steps = [entry["step"] for entry in data]
    unique_steps = set(steps)
    contiguous_steps = set(range(0, max(unique_steps) + 1))

    if unique_steps != contiguous_steps:
        raise ValueError("Steps must form a contiguous sequence from 0")
    
def map_to_pipeline(data):
    curr_step = 0
    begin_curr_step = 0
    begin_next_step = begin_curr_step + 1
    tcam_mapping = [0] * STAGES
    sram_mapping = [0] * STAGES
    id_mapping = [[] for _ in range(STAGES)]

    for entry in data:
        if entry["step"] != curr_step:
            curr_step = entry["step"]
            begin_curr_step = begin_next_step
            begin_next_step = begin_curr_step + 1

        if entry["match"] == "ternary":
            curr_stage = begin_curr_step
            blocks = math.ceil(entry["key_size"] / TCAM_WIDTH) * math.ceil(entry["entries"] / TCAM_HEIGHT)

            while blocks > 0:
                if curr_stage >= STAGES:
                    raise ValueError("Could not complete mapping - out of available stages")

                available_blocks = BLOCKS_PER_STAGE - tcam_mapping[curr_stage]

                if min(available_blocks, blocks) > 0:
                    id_mapping[curr_stage].append(entry["id"])

                tcam_mapping[curr_stage] += min(available_blocks, blocks)
                blocks -= min(available_blocks, blocks)
                curr_stage += 1
            
            if curr_stage > begin_next_step:
                begin_next_step = curr_stage

        elif entry["match"] == "exact":
            curr_stage = begin_curr_step

            if entry["method"] == "index":
                pages = math.ceil((2**entry["key_size"] * entry["data_size"]) / SRAM_PAGE)
            elif entry["method"] == "hash":
                pages = math.ceil((HASH_MULTIPLIER * entry["entries"] * (entry["key_size"] + entry["data_size"])) / SRAM_PAGE)

            while pages > 0:
                if curr_stage >= STAGES:
                    raise ValueError("Could not complete mapping - out of available stages")

                available_pages = PAGES_PER_STAGE - sram_mapping[curr_stage]

                if min(available_pages, pages) > 0:
                    id_mapping[curr_stage].append(entry["id"])

                sram_mapping[curr_stage] += min(available_pages, pages)
                pages -= min(available_pages, pages)
                curr_stage += 1
            
            if curr_stage > begin_next_step:
                begin_next_step = curr_stage

    print("TCAM mapping: ")
    print(tcam_mapping)
    print("SRAM mapping: ")
    print(sram_mapping)
    print("id mapping: ")
    print(id_mapping)

def main():
    if len(sys.argv) != 2:
        print("Usage: python sim.py <filename>.json")
        sys.exit(1)

    filename = sys.argv[1]

    with open(filename, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Expected a JSON list")
        sys.exit(1)

    check_mandatory_fields(data)
    check_unique_ids(data)
    check_contiguous_steps(data)

    data = sorted(data, key=lambda x: x["step"])

    map_to_pipeline(data)

if __name__=="__main__":
    main()