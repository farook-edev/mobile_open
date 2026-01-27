import json

inp = "merged.jsonl"
out = inp.split('.')[0]+'.json'

with open(inp, "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

with open(out, "w") as f:
    json.dump(data, f, indent=2)
