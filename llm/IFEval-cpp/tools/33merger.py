# merge IFEval33 dataset result JSONL + JSONL containing "instruction_id_list", "kwargs" fields.

import json

FILE1 = "input.jsonl"
FILE2 = "merged.jsonl"
OUT   = "33-merged.jsonl"

KEY_FIELD = "key"                       # common unique field in both files
FIELDS_FROM_FILE2 = ["instruction_id_list", "kwargs"]    # fields to take from file2
COUNTER = "instruction_id_list"


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    # Load both JSONL files
    f1 = load_jsonl(FILE1)
    f2 = load_jsonl(FILE2)

    # Build lookup map from file2
    map2 = {obj[KEY_FIELD]: obj for obj in f2}
    count = 0

    with open(OUT, "w", encoding="utf-8") as out:
        for obj in f1:
            key = obj.get(KEY_FIELD)
            if key in map2:
                src = map2[key]
                # Copy selected fields
                count += len(src[COUNTER])
                print(count)
                for field in FIELDS_FROM_FILE2:
                    if field in src:
                        obj[field] = src[field]

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
