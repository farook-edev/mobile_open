import argparse
import json

from transformers import AutoTokenizer


def special_tokens(tokenizer):
    tokens = set(tokenizer.all_special_tokens)
    for token in tokenizer.added_tokens_decoder.values():
        if token.special:
            tokens.add(token.content)
    return sorted((t for t in tokens if t), key=len, reverse=True)


def remove_special_tokens(text, tokens):
    for token in tokens:
        text = text.replace(token, "")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    tokens = special_tokens(AutoTokenizer.from_pretrained(args.tokenizer))

    with open(args.input, "r", encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["response"] = remove_special_tokens(obj["response"], tokens)
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
