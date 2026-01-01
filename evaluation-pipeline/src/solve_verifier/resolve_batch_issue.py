import json, re, unicodedata, yaml
from collections import defaultdict
from typing import Dict, List, Iterable, Union
from pathlib import Path


# ---------- Robust question extraction (works for all 6 prompts) ----------
SMART_TO_ASCII = str.maketrans({
    "’": "'", "‘": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
})

_TAIL_MARKERS_RE = re.compile(
    r"""
    ^\s*(
        answer\s*:                       # "Answer:"
      | step\s*-\s*by\s*-\s*step
      | step\s*by\s*step
      | explain\b
      | provide\b
      | your\s+response\s+should\b
      | the\s+last\s+number\b
      | walk\s+through\b
      | break\s+down\b
    )
    """,
    flags=re.IGNORECASE | re.MULTILINE | re.VERBOSE
)

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(SMART_TO_ASCII)
    s = re.sub(r"^\s*```.*?\n|\n```$", "", s, flags=re.DOTALL)  # strip code fences if any
    return s

def extract_core_question(text: str) -> str:
    """
    Return ONLY the actual question across all six prompt variants.
    - Find 'Question:' (case-insensitive, optional colon).
    - Take everything after it up to the first tail marker (Answer/Step-by-Step/etc.) or end-of-string.
    - Normalize whitespace & punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    s = _normalize(text)
    m_q = re.search(r'question\s*:?', s, flags=re.IGNORECASE)
    if not m_q:
        core = s
    else:
        start = m_q.end()
        tail = _TAIL_MARKERS_RE.search(s, pos=start)
        end = tail.start() if tail else len(s)
        core = s[start:end]
    core = re.sub(r'^\s*[:\-–—]*\s*', '', core)
    core = " ".join(core.strip().split())
    return core

# ---------- Aggregation over a JSONL stream (your base, reused) ----------
def aggregate_answers_by_question_lines(lines: Iterable[str],
                                        question_key: str = "question",
                                        answer_key: str = "answer"
                                       ) -> Dict[str, List[str]]:
    """
    Build {clean_question: [answers_as_strings,...]} from a JSONL stream.
    """
    buckets: Dict[str, List[str]] = defaultdict(list)
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        raw_q = obj.get(question_key, "")
        core_q = extract_core_question(raw_q)
        ans_val: Union[str, int, float, None] = obj.get(answer_key, None)
        if ans_val is None:
            continue
        buckets[core_q].append(str(ans_val))
    return buckets

def aggregate_answers_by_question_file(jsonl_path: str,
                                       question_key: str = "question",
                                       answer_key: str = "answer"
                                      ) -> Dict[str, List[str]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return aggregate_answers_by_question_lines(f, question_key, answer_key)

# ---------- NEW: build ground-truth index from data_paths ----------
def build_ground_truth_index(data_paths: List[str]) -> Dict[str, Dict[str, Union[int, str]]]:
    """
    Reads one or more ground-truth JSONL files where each line has:
      {"id": int, "question": str, "answer": (int/float/str), ...}
    Returns:
      { clean_question: {"id": id, "question": original_question, "answer": gt_answer_str} }
    If duplicate clean questions appear, first occurrence is kept.
    """
    gt: Dict[str, Dict[str, Union[int, str]]] = {}
    for p in data_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_q = obj.get("question", "")
                clean_q = extract_core_question(raw_q)
                if clean_q in gt:
                    # already seen; skip or assert consistent id/answer if you want
                    continue
                ans_val = obj.get("answer", None)
                if ans_val is None:
                    continue
                gt[clean_q] = {
                    "id": obj.get("id", None),
                    "question": raw_q,
                    "answer": str(ans_val),
                }
    return gt

# ---------- NEW: read model outputs from result_paths ----------
def aggregate_model_outputs(result_paths: List[str]) -> Dict[str, List[str]]:
    """
    Reads one or more result JSONL files where each line has:
      {"question": <prompt+Question...>, "answer": <model_answer_str>, ...}
    Returns:
      { clean_question: [list of model answers as strings in file order] }
    """
    merged: Dict[str, List[str]] = defaultdict(list)
    for p in result_paths:
        with open(p, "r", encoding="utf-8") as f:
            part = aggregate_answers_by_question_lines(f, question_key="question", answer_key="answer")
        for k, v in part.items():
            merged[k].extend(v)
    return merged

# ---------- NEW: merge and write ----------
def merge_and_write_jsonl(gt_index: Dict[str, Dict[str, Union[int, str]]],
                          outputs_map: Dict[str, List[str]],
                          output_path: str):
    """
    For each ground-truth question, attach outputs (if any) and write a JSONL with:
      {"id", "question", "answer", "outputs": [...]}
    Note: 'prompts' omitted per your instruction; add later if desired.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for clean_q, meta in gt_index.items():
            rec = {
                "id": meta.get("id"),
                "question": meta.get("question"),
                "answer": meta.get("answer"),
                "outputs": outputs_map.get(clean_q, []),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- Optional: load YAML config with paths ----------
def load_paths_from_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # In your YAML these are lists (even if single item). Keep that behavior.
    data_paths = cfg.get("data_paths", []) or cfg.get("data_path", [])
    result_paths = cfg.get("result_path", [])  # the key you provided is singular but value is a list
    output_paths = cfg.get("output_path", [])  # same
    if isinstance(data_paths, str): data_paths = [data_paths]
    if isinstance(result_paths, str): result_paths = [result_paths]
    if isinstance(output_paths, str): output_paths = [output_paths]
    return data_paths, result_paths, output_paths

# ---------- Example main ----------
if __name__ == "__main__":
    # If you want to drive it with YAML:
    # config_yaml = "/path/to/config.yaml"
    # data_paths, result_paths, output_paths = load_paths_from_yaml(config_yaml)

    # Or hardcode for a quick run:
    data_paths = [
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/amharic/symbolic/distribution_dataset_9.jsonl",

        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/chinese/symbolic/distribution_dataset_9.jsonl",


        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/english/symbolic/distribution_dataset_9.jsonl",

        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/french/symbolic/distribution_dataset_9.jsonl",

        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/igbo/symbolic/distribution_dataset_9.jsonl",


        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/japanese/symbolic/distribution_dataset_9.jsonl",


        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/swahili/symbolic/distribution_dataset_9.jsonl",

        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/twi/symbolic/distribution_dataset_9.jsonl",

        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/IC/distribution_dataset_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/dataset/distribution_dataset/yoruba/symbolic/distribution_dataset_9.jsonl",


    ]
    result_paths = [
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/amharic/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/chinese/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/english/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/french/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/igbo/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/japanese/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/swahili/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/twi/symbolic/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/IC/distribution_result_9/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_0/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_1/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_2/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_3/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_4/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_5/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_6/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_7/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_8/results/answers.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-sonnet-4-0/yoruba/symbolic/distribution_result_9/results/answers.jsonl",


    ]
    output_paths = [
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/amharic/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/chinese/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/english/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/french/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/igbo/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/japanese/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/swahili/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/twi/symbolic/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/IC/distribution_result_9.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_0.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_1.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_2.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_3.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_4.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_5.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_6.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_7.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_8.jsonl",
        "/home/mila/x/xut/github/evaluation-pipeline/result/distribution_dataset/claude-4-0/yoruba/symbolic/distribution_result_9.jsonl",
    ]

    assert len(data_paths) == len(result_paths) == len(output_paths), \
        f"data_paths {len(data_paths)}, result_paths{len(result_paths)}, output_paths{len(output_paths)} must be the same length and aligned."

    total_written = 0
    for gt_path, res_path, out_path in zip(data_paths, result_paths, output_paths):
        # Build per-split GT + outputs (keeps files independent)
        gt_index = build_ground_truth_index([gt_path])
        outputs_map = aggregate_model_outputs([res_path])

        merge_and_write_jsonl(gt_index, outputs_map, out_path)
        print(f"Wrote merged JSONL with {len(gt_index)} questions to: {out_path}")
        total_written += len(gt_index)

    print(f"Done. Total questions across splits: {total_written}")
