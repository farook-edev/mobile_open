# This tool compares the output from the python implementation against the C++ implementation, and produces any mismatches between the 2 evaulation results.
#
# Any results that are identical are ignored.

import json

file1 = "results-cpp.jsonl"
suffix_a = "cpp"
file2 = "results-py.jsonl"
suffix_b = "py"
out_file = "discrepancies.jsonl"

id_field = "key"
compare_field = "follow_instruction_list"


def load_jsonl(path, id_field):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if id_field in obj:
                d[obj[id_field]] = obj
    return d

a = load_jsonl(file1, id_field)
b = load_jsonl(file2, id_field)

with open(out_file, "w", encoding="utf-8") as out:
    for k in sorted(set(a.keys()) & set(b.keys())):
        obj_a = a[k]
        obj_b = b[k]

        if obj_a.get(compare_field) == obj_b.get(compare_field):
            continue

        merged = {id_field: k}

        all_fields = set(obj_a.keys()) | set(obj_b.keys())
        all_fields.discard(id_field)

        for field in all_fields:
            in_a = field in obj_a
            in_b = field in obj_b

            if in_a and in_b:
                if obj_a[field] == obj_b[field]:
                    merged[field] = obj_a[field]
                else:
                    merged[f"{field}-{suffix_a}"] = obj_a[field]
                    merged[f"{field}-{suffix_b}"] = obj_b[field]
            elif in_a:
                merged[field] = obj_a[field]
            else:
                merged[field] = obj_b[field]

        out.write(json.dumps(merged, ensure_ascii=False) + "\n")
