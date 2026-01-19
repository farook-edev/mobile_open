import json

input_path = "cpp-results.txt"
loose_output_path = "cpp-results-loose.jsonl"
strict_output_path = "cpp-results-strict.jsonl"

def iter_jsonl_after_details(path):
    """Yield parsed JSON objects after the 'Details:' line."""
    with open(path, "r", encoding="utf-8") as f:
        in_details = False
        for line in f:
            if not in_details:
                if line.strip().startswith("Details:"):
                    in_details = True
                continue

            line = line.strip()
            if not line:
                continue

            yield json.loads(line)

with open(loose_output_path, "w", encoding="utf-8") as loose_f, \
     open(strict_output_path, "w", encoding="utf-8") as strict_f:

    for obj in iter_jsonl_after_details(input_path):
        # ---- loose version ----
        loose_obj = {
            **{k: v for k, v in obj.items()
               if not k.endswith("_loose") and not k.endswith("_strict")},
            "follow_instruction_list": obj.get("follow_instruction_list_loose"),
            "follow_all_instructions": obj.get("follow_all_instructions_loose"),
        }

        loose_f.write(json.dumps(loose_obj, ensure_ascii=False) + "\n")

        # ---- strict version ----
        strict_obj = {
            **{k: v for k, v in obj.items()
               if not k.endswith("_loose") and not k.endswith("_strict")},
            "follow_instruction_list": obj.get("follow_instruction_list_strict"),
            "follow_all_instructions": obj.get("follow_all_instructions_strict"),
        }

        strict_f.write(json.dumps(strict_obj, ensure_ascii=False) + "\n")

print("Done.")
print(f"Loose results  -> {loose_output_path}")
print(f"Strict results -> {strict_output_path}")
