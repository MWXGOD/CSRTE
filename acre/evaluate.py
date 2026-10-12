#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from pypinyin import Style, lazy_pinyin


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def label(row):
    value = row.get("labels", row.get("solution"))
    if value is not None:
        return value
    for message in reversed(row.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def answer(text):
    text = str(text or "").replace("<|endoftext|>", "")
    start, end = text.rfind("<answer>"), text.rfind("</answer>")
    if start >= 0 and end > start:
        return text[start + 8:end].strip()
    return text.strip()


def triples(text):
    result = []
    for item in answer(text).split("$$"):
        fields = [part.strip() for part in item.split("##", 2)]
        if len(fields) == 3 and all(fields):
            result.append(tuple(fields))
    return result


def f1(pred, gold):
    pred, gold = set(pred), set(gold)
    hit = len(pred & gold)
    precision = hit / len(pred) if pred else 0.0
    recall = hit / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def strict(predictions, labels):
    triplet_hit = triplet_pred = triplet_gold = 0
    entity_hit = entity_pred = entity_gold = 0
    relation_hit = relation_pred = relation_gold = 0
    for pred, gold in zip(predictions, labels):
        p, g = triples(pred), triples(gold)
        ps, gs = set(p), set(g)
        pe = {(x[0]) for x in ps} | {(x[2]) for x in ps}
        ge = {(x[0]) for x in gs} | {(x[2]) for x in gs}
        pr = {x[1] for x in ps}
        gr = {x[1] for x in gs}
        triplet_hit += len(ps & gs); triplet_pred += len(ps); triplet_gold += len(gs)
        entity_hit += len(pe & ge); entity_pred += len(pe); entity_gold += len(ge)
        relation_hit += len(pr & gr); relation_pred += len(pr); relation_gold += len(gr)
    def metric(hit, pred, gold):
        p = hit / pred if pred else 0.0
        r = hit / gold if gold else 0.0
        return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return {"triplet": metric(triplet_hit, triplet_pred, triplet_gold),
            "entity": metric(entity_hit, entity_pred, entity_gold),
            "relation": metric(relation_hit, relation_pred, relation_gold)}


def pinyin_entity(text):
    return " ".join(lazy_pinyin(re.sub(r"\s+", "", str(text).lower()),
                               style=Style.TONE3, neutral_tone_with_five=True,
                               errors="default"))


def pae(predictions, labels):
    p, g = [], []
    for pred, gold in zip(predictions, labels):
        p.extend((pinyin_entity(h), r, pinyin_entity(t)) for h, r, t in triples(pred))
        g.extend((pinyin_entity(h), r, pinyin_entity(t)) for h, r, t in triples(gold))
    return {"triplet": f1(p, g)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    parser.add_argument("--label_path", required=True)
    args = parser.parse_args()
    result_rows, label_rows = rows(args.result_path), rows(args.label_path)
    if len(result_rows) != len(label_rows):
        raise ValueError(f"row count mismatch: {len(result_rows)} != {len(label_rows)}")
    predictions = [row.get("response", "") for row in result_rows]
    labels = [label(row) for row in label_rows]
    payload = {"num_samples": len(predictions), "strict": strict(predictions, labels), "pae": pae(predictions, labels)}
    output = Path(args.result_path).with_suffix(".metrics.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
