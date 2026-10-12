import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from swift.rewards import ORM, orms
except Exception:
    class ORM:  # type: ignore[override]

        def __init__(self, args: Any = None, **kwargs):
            self.args = args

        def __call__(self, **kwargs) -> List[float]:
            raise NotImplementedError

    orms = {}


PRINT_EVERY = 10
MAX_PRINT_GROUPS = 2
NONE_TOKEN = "none"
END_OF_TEXT = "<|endoftext|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

WEIGHT_FORMAT = 0.05
WEIGHT_TRIPLE_F1 = 0.80
WEIGHT_ENTITY_F1 = 0.05
WEIGHT_RELATION_F1 = 0.05
WEIGHT_PRONUNCIATION = 0.025
WEIGHT_PAUSE = 0.025

MAX_COMPLETION_CHARS = 4096
MAX_ANSWER_CHARS = 2048
MAX_TRIPLETS = 32
MIN_COT_SIMILARITY_FOR_FULL_REWARD = 0.15
MALFORMED_REWARD = -0.1
LOW_COT_REWARD_CAP = 0.35
EMPTY_ANSWER_WRONG_COT_CAP = 0.25
REPETITION_REWARD_CAP = 0.05
ZERO_TRIPLE_NONEMPTY_REWARD_CAP = 0.0
MAX_TRIPLET_FIELD_CHARS = 128


@dataclass(frozen=True)
class ParsedTriplets:
    raw_text: str
    cleaned_text: str
    is_none_output: bool
    separator_mode: str
    triplets: Tuple[Tuple[str, str, str], ...]
    unique_triplets: frozenset[Tuple[str, str, str]]
    entities: frozenset[str]
    relations: frozenset[str]
    segment_count: int
    valid_segment_count: int


@dataclass(frozen=True)
class ParsedCot:
    raw_text: str
    payload: Mapping[str, Any]
    has_think_tags: bool
    is_valid_json: bool
    has_duplicate_keys: bool
    pronunciation: str
    pause: str


def _normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _safe_f1(predicted: Iterable[Any], gold: Iterable[Any]) -> float:
    pred_set = set(predicted)
    gold_set = set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    hit = len(pred_set & gold_set)
    precision = hit / len(pred_set)
    recall = hit / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _multiset_f1(predicted: Iterable[Any], gold: Iterable[Any]) -> float:
    pred_counts = Counter(predicted)
    gold_counts = Counter(gold)
    pred_total = sum(pred_counts.values())
    gold_total = sum(gold_counts.values())
    if pred_total == 0 and gold_total == 0:
        return 1.0
    if pred_total == 0 or gold_total == 0:
        return 0.0
    hit = sum((pred_counts & gold_counts).values())
    precision = hit / pred_total
    recall = hit / gold_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _clean_text(text: Optional[str]) -> str:
    cleaned = "" if text is None else str(text)
    cleaned = cleaned.replace(END_OF_TEXT, "").strip()
    cleaned = re.sub(r"(?:\s*\$\$\s*)+$", "", cleaned).strip()
    return cleaned


def has_answer_tags(text: Optional[str]) -> bool:
    cleaned = _clean_text(text)
    start = cleaned.rfind(ANSWER_OPEN)
    end = cleaned.rfind(ANSWER_CLOSE)
    return start != -1 and end != -1 and end > start


def extract_answer_text(text: Optional[str]) -> str:
    cleaned = _clean_text(text)
    start = cleaned.rfind(ANSWER_OPEN)
    end = cleaned.rfind(ANSWER_CLOSE)
    if start != -1 and end != -1 and end > start:
        return cleaned[start + len(ANSWER_OPEN):end].strip()

    think_end = cleaned.rfind(THINK_CLOSE)
    if think_end != -1:
        return cleaned[think_end + len(THINK_CLOSE):].strip()

    return cleaned


def extract_think_text(text: Optional[str]) -> Tuple[str, bool]:
    cleaned = _clean_text(text)
    start = cleaned.find(THINK_OPEN)
    end = cleaned.find(THINK_CLOSE, start + len(THINK_OPEN)) if start != -1 else -1
    if start != -1 and end != -1 and end > start:
        return cleaned[start + len(THINK_OPEN):end].strip(), True
    end = cleaned.find(THINK_CLOSE)
    if end != -1:
        return cleaned[:end].strip(), True
    return "", False


def _json_loads_relaxed(text: str) -> Tuple[Mapping[str, Any], bool, bool]:
    if not text:
        return {}, False, False

    has_duplicate_keys = False

    def _hook(pairs):
        nonlocal has_duplicate_keys
        obj = {}
        seen = set()
        for key, value in pairs:
            if key in seen:
                has_duplicate_keys = True
            seen.add(key)
            obj[key] = value
        return obj

    try:
        obj = json.loads(text, object_pairs_hook=_hook)
        return ((obj, True, has_duplicate_keys) if isinstance(obj, Mapping) else ({}, False, has_duplicate_keys))
    except Exception:
        return {}, False, has_duplicate_keys


def parse_cot(text: Optional[str]) -> ParsedCot:
    think_text, has_tags = extract_think_text(text)
    payload, is_valid, has_duplicate_keys = _json_loads_relaxed(think_text)
    pronunciation = ""
    pause = ""
    if isinstance(payload, Mapping):
        pronunciation = payload.get("pronunciation", payload.get("pron", ""))
        pause = payload.get("pause", "")
    return ParsedCot(
        raw_text=think_text,
        payload=payload,
        has_think_tags=has_tags,
        is_valid_json=is_valid,
        has_duplicate_keys=has_duplicate_keys,
        pronunciation="" if pronunciation is None else str(pronunciation),
        pause="" if pause is None else str(pause),
    )


def has_required_cot_fields(cot: ParsedCot) -> bool:
    return bool(cot.pronunciation.strip() and cot.pause.strip())


def has_valid_cot_format(cot: ParsedCot) -> bool:
    return bool(
        cot.has_think_tags and cot.is_valid_json and not cot.has_duplicate_keys and has_required_cot_fields(cot))


def has_repetition_artifact(text: Optional[str]) -> bool:
    raw = "" if text is None else str(text)
    if re.search(r"(?:<ttt\|t|\|ttt\|t|tttt|(?:ent){8,}|<\|[^>]{0,20}){2,}", raw, flags=re.IGNORECASE):
        return True
    compact = re.sub(r"\s+", "", raw.lower())
    for size in range(3, 13):
        for i in range(0, max(len(compact) - size * 8 + 1, 0), size):
            piece = compact[i:i + size]
            if len(set(piece)) <= 2:
                continue
            if piece * 8 in compact:
                return True
    return False


def sequence_similarity(pred: str, gold: str) -> float:
    pred_norm = _normalize_text(pred)
    gold_norm = _normalize_text(gold)
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    return float(SequenceMatcher(None, pred_norm, gold_norm).ratio())


def _pause_signature(text: str) -> Tuple[str, Tuple[int, ...]]:
    tokens = re.split(r"(<pause>)", text, flags=re.IGNORECASE)
    text_parts: List[str] = []
    pause_positions: List[int] = []
    text_length = 0
    for token in tokens:
        if token.lower() == "<pause>":
            pause_positions.append(text_length)
            continue
        normalized = re.sub(r"\s+", "", _normalize_text(token))
        if not normalized:
            continue
        text_parts.append(normalized)
        text_length += len(normalized)
    return "".join(text_parts), tuple(pause_positions)


def pause_reward(pred_pause: str, gold_pause: str) -> float:
    pred_base, pred_positions = _pause_signature(pred_pause)
    gold_base, gold_positions = _pause_signature(gold_pause)
    text_score = sequence_similarity(pred_base, gold_base)
    position_score = _multiset_f1(pred_positions, gold_positions)
    return 0.5 * text_score + 0.5 * position_score


def cot_reward(pred: ParsedCot, gold: ParsedCot) -> Tuple[float, float, float]:
    pronunciation = sequence_similarity(pred.pronunciation, gold.pronunciation)
    pause = pause_reward(pred.pause, gold.pause)
    format_score = 1.0 if has_valid_cot_format(pred) else 0.0
    return float(pronunciation), float(pause), float(format_score)


def parse_triplets(text: Optional[str]) -> ParsedTriplets:
    cleaned_text = extract_answer_text(text)
    normalized_cleaned = _normalize_text(cleaned_text)

    if normalized_cleaned == NONE_TOKEN:
        return ParsedTriplets(
            raw_text="" if text is None else str(text),
            cleaned_text=cleaned_text,
            is_none_output=True,
            separator_mode="none",
            triplets=(),
            unique_triplets=frozenset(),
            entities=frozenset(),
            relations=frozenset(),
            segment_count=0,
            valid_segment_count=0,
        )

    if not cleaned_text:
        return ParsedTriplets(
            raw_text="" if text is None else str(text),
            cleaned_text=cleaned_text,
            is_none_output=False,
            separator_mode="empty",
            triplets=(),
            unique_triplets=frozenset(),
            entities=frozenset(),
            relations=frozenset(),
            segment_count=0,
            valid_segment_count=0,
        )

    if "$$" in cleaned_text:
        separator_mode = "standard"
        raw_segments = re.split(r"\s*\$\$\s*", cleaned_text)
    elif "\n" in cleaned_text:
        separator_mode = "newline"
        raw_segments = cleaned_text.splitlines()
    else:
        separator_mode = "single"
        raw_segments = [cleaned_text]

    triplets: List[Tuple[str, str, str]] = []
    segment_count = 0

    for raw_segment in raw_segments:
        segment = raw_segment.strip()
        if not segment:
            continue
        segment_count += 1
        parts = [part.strip() for part in segment.split("##", 2)]
        if len(parts) != 3 or any(not part for part in parts):
            continue
        head, relation, tail = parts
        triplets.append((_normalize_text(head), _normalize_text(relation), _normalize_text(tail)))

    unique_triplets = frozenset(triplets)
    entities = frozenset(entity for head, _, tail in unique_triplets for entity in (head, tail))
    relations = frozenset(relation for _, relation, _ in unique_triplets)

    return ParsedTriplets(
        raw_text="" if text is None else str(text),
        cleaned_text=cleaned_text,
        is_none_output=False,
        separator_mode=separator_mode,
        triplets=tuple(triplets),
        unique_triplets=unique_triplets,
        entities=entities,
        relations=relations,
        segment_count=segment_count,
        valid_segment_count=len(triplets),
    )


def format_reward(pred: ParsedTriplets, gold: ParsedTriplets) -> float:
    raw_lower = pred.raw_text.lower()
    cleaned_lower = pred.cleaned_text.lower()
    if gold.segment_count == 0:
        return 1.0 if pred.segment_count == 0 and pred.cleaned_text == "" else 0.0
    if pred.raw_text.count(ANSWER_OPEN) != 1 or pred.raw_text.count(ANSWER_CLOSE) != 1:
        return 0.0
    if pred.raw_text.count(THINK_CLOSE) > 1 or pred.raw_text.count(THINK_OPEN) > 1:
        return 0.0
    if any(token in cleaned_lower for token in ("<answer", "</answer", "<think", "</think", "<pause", "<|")):
        return 0.0
    if pred.is_none_output or pred.segment_count == 0:
        return 0.0
    if pred.valid_segment_count != pred.segment_count:
        return 0.0
    if pred.valid_segment_count > MAX_TRIPLETS:
        return 0.0
    if len(pred.cleaned_text) > MAX_ANSWER_CHARS:
        return 0.0
    if any(len(part) > MAX_TRIPLET_FIELD_CHARS for triplet in pred.triplets for part in triplet):
        return 0.0
    if any(any(token in part.lower() for token in ("<answer", "</answer", "<think", "</think", "<pause", "<|"))
           for triplet in pred.triplets for part in triplet):
        return 0.0

    if pred.separator_mode in {"standard", "single"}:
        return 1.0
    if pred.separator_mode == "newline":
        return 0.5
    return 0.0


def explain_simple_reward_case(completion: str, solution: str) -> Dict[str, Any]:
    pred = parse_triplets(completion)
    gold = parse_triplets(solution)
    pred_cot = parse_cot(completion)
    gold_cot = parse_cot(solution)

    triple_f1 = _safe_f1(pred.unique_triplets, gold.unique_triplets)
    entity_f1 = _safe_f1(pred.entities, gold.entities)
    relation_f1 = _safe_f1(pred.relations, gold.relations)
    cot_pronunciation, cot_pause, cot_format = cot_reward(pred_cot, gold_cot)
    answer_format = format_reward(pred, gold) if has_answer_tags(completion) else 0.0
    fmt = 0.5 * answer_format + 0.5 * cot_format
    repetition_artifact = has_repetition_artifact(completion)
    too_long = len("" if completion is None else str(completion)) > MAX_COMPLETION_CHARS
    low_cot_similarity = min(cot_pronunciation, cot_pause) < MIN_COT_SIMILARITY_FOR_FULL_REWARD

    if cot_format == 0.0 or answer_format == 0.0 or too_long:
        total_reward = MALFORMED_REWARD
    else:
        total_reward = (
            WEIGHT_FORMAT * fmt
            + WEIGHT_TRIPLE_F1 * triple_f1
            + WEIGHT_ENTITY_F1 * entity_f1
            + WEIGHT_RELATION_F1 * relation_f1
            + WEIGHT_PRONUNCIATION * cot_pronunciation
            + WEIGHT_PAUSE * cot_pause
        )
        if repetition_artifact:
            total_reward = min(total_reward, REPETITION_REWARD_CAP)
        elif gold.segment_count > 0 and triple_f1 == 0.0:
            total_reward = min(total_reward, ZERO_TRIPLE_NONEMPTY_REWARD_CAP)
        elif gold.segment_count == 0 and low_cot_similarity:
            total_reward = min(total_reward, EMPTY_ANSWER_WRONG_COT_CAP)
        elif low_cot_similarity:
            total_reward = min(total_reward, LOW_COT_REWARD_CAP)

    return {
        "prediction": pred,
        "gold": gold,
        "prediction_cot": pred_cot,
        "gold_cot": gold_cot,
        "format": float(fmt),
        "answer_format": float(answer_format),
        "cot_format": float(cot_format),
        "triple_f1": float(triple_f1),
        "entity_f1": float(entity_f1),
        "relation_f1": float(relation_f1),
        "cot_pronunciation": float(cot_pronunciation),
        "cot_pause": float(cot_pause),
        "repetition_artifact": bool(repetition_artifact),
        "too_long": bool(too_long),
        "low_cot_similarity": bool(low_cot_similarity),
        "reward": float(total_reward),
    }


class SpeechRESimpleReward(ORM):
    call_count = 0

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.num_generations = getattr(args, "num_generations", 4) or 4

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        SpeechRESimpleReward.call_count += 1
        rewards = []
        debug_rows = []

        for completion, gold in zip(completions, solution):
            breakdown = explain_simple_reward_case(completion, gold)
            rewards.append(breakdown["reward"])
            debug_rows.append(breakdown)

        if SpeechRESimpleReward.call_count % PRINT_EVERY == 0:
            print("\n" + "=" * 100, flush=True)
            print(f"[SpeechRESimpleReward] call={SpeechRESimpleReward.call_count}", flush=True)
            group_size = max(int(self.num_generations), 1)
            total = len(debug_rows)

            for group_idx in range(min(MAX_PRINT_GROUPS, max((total + group_size - 1) // group_size, 1))):
                start = group_idx * group_size
                end = min(start + group_size, total)
                group = debug_rows[start:end]
                if not group:
                    continue
                print("-" * 100, flush=True)
                print(f"[Group {group_idx}] GOLD: {sorted(group[0]['gold'].unique_triplets)}", flush=True)

                for idx, row in enumerate(group):
                    print(f"\n[Completion {idx}]", flush=True)
                    print(f"RAW: {row['prediction'].raw_text}", flush=True)
                    print(f"PRED: {sorted(row['prediction'].unique_triplets)}", flush=True)
                    print(
                        f"reward={row['reward']:.4f}, "
                        f"format={row['format']:.4f}, "
                        f"answer_format={row['answer_format']:.4f}, "
                        f"cot_format={row['cot_format']:.4f}, "
                        f"triple_f1={row['triple_f1']:.4f}, "
                        f"entity_f1={row['entity_f1']:.4f}, "
                        f"relation_f1={row['relation_f1']:.4f}, "
                        f"cot_pronunciation={row['cot_pronunciation']:.4f}, "
                        f"cot_pause={row['cot_pause']:.4f}",
                        flush=True,
                    )
            print("=" * 100 + "\n", flush=True)

        return rewards


orms["speechre_acre_reward"] = SpeechRESimpleReward
