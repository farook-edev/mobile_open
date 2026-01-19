import json

FILE1_PATH = "input_data.jsonl"
FILE2_PATH = "input_response_data_gpt4_20231107_145030.jsonl"
OUTPUT_PATH = "merged.jsonl"

MERGE_KEY = "prompt"


def load_jsonl_by_key(path, key):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num} invalid JSON: {e}")

            k = obj.get(key)
            if k is None:
                continue

            data[k] = obj

    return data


def merge_jsonl(file1, file2, output, key):
    data1 = load_jsonl_by_key(file1, key)
    data2 = load_jsonl_by_key(file2, key)

    common_keys = data1.keys() & data2.keys()

    with open(output, "w", encoding="utf-8") as out:
        for k in common_keys:
            merged = {
                **data1[k],
                **data2[k],  # file2 overrides file1 on conflicts
            }
            out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    print(f"Merged {len(common_keys)} records using key '{key}'")


if __name__ == "__main__":
    merge_jsonl(
        FILE1_PATH,
        FILE2_PATH,
        OUTPUT_PATH,
        MERGE_KEY,
    )
