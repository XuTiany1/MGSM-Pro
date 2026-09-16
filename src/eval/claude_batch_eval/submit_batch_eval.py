"""
Submits an evaluation run to the Claude Batch API.

Usage (run from src/eval/):
    python -m claude_batch_eval.submit_batch_eval --config claude_batch_eval/config/claude-sonnet-4-0/mgsm.yaml

Requires the ANTHROPIC_API_KEY environment variable.
"""

import os
import json
import yaml
from argparse import ArgumentParser

from anthropic import Anthropic

from prompts.prompts import *
from prompts.exemplars import get_exemplars_prefix
from dataset_paths import resolve_dir, load_dataset

from claude_batch_eval.claude_batch_manager import ClaudeBatchApiManager


def construct_prompt(question: str, language: str, prompt_list=("english",), n_shots: int = 0):
    """prompt_list holds prompt names (e.g. "english"), matching a cot_prompt_<name> in prompts.py."""
    prefix = get_exemplars_prefix(language, n_shots)
    final_list_of_prompt = []

    for prompt_name in prompt_list:
        var_name = f"cot_prompt_{prompt_name}"
        if var_name in globals():
            final_list_of_prompt.append(prefix + globals()[var_name].format(question=question))
        else:
            print(f"Warning: Prompt template '{var_name}' not found in globals()")

    return final_list_of_prompt


def main():
    parser = ArgumentParser(description="Claude Batch API evaluation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_dataset_folder = config["base_dataset_folder"]
    dataset_type = config["dataset_type"]
    languages = config["language"]
    model_name = config["model_name"]  # e.g. "claude-sonnet-4-0"
    model_max_token = config["model_max_token"]
    model_temperature = config["model_temperature"]
    result_basedir = config["result_basedir"]

    prompt_ids = config.get("prompt_id_to_use") or ["english"]
    n_shots = int(config.get("n_shots", 0))
    print(f"Using prompts: {prompt_ids} with {n_shots} shots")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    # Gather (result_dir, language, dataset) for every language/variation combo.
    jobs = []
    for lang in languages:
        for i in range(config["total_num_variation"]):
            dataset = load_dataset(base_dataset_folder, dataset_type, lang, index=i)
            result_dir = resolve_dir(result_basedir, dataset_type, lang, model_name=model_name)
            jobs.append((result_dir, lang, dataset))

    for result_dir, lang, dataset in jobs:
        out_dir = os.path.join(result_dir, "results")
        answers_jsonl = os.path.join(out_dir, "answers.jsonl")

        if os.path.exists(answers_jsonl):
            print(f"[SKIPPING] - Path already exists {answers_jsonl}")
            continue

        manager = ClaudeBatchApiManager(client=client, base_dir=result_dir, deployment=model_name)

        prompts_dict = {}
        meta = {}
        for q_idx, data in enumerate(dataset):
            prompts = construct_prompt(data["question"], lang, prompt_list=prompt_ids, n_shots=n_shots)
            for p_idx, prompt in enumerate(prompts):
                custom_id = f"q{q_idx}_prompt{prompt_ids[p_idx]}"
                prompts_dict[custom_id] = prompt
                meta[custom_id] = data

        print(f"\n[PROCESSING] {result_dir}")
        batch_id, results = manager.compute_batch(
            custom_id_to_texts=prompts_dict,
            temperature=model_temperature,
            max_tokens=model_max_token,
            poll_interval=20,
            max_wait_seconds=7200,  # 2 hours
            verbose=True,
            write_history=True,
            save_results=True,
        )
        print(f"\n[BATCH COMPLETE] Batch ID: {batch_id}")

        os.makedirs(out_dir, exist_ok=True)
        with open(answers_jsonl, "w", encoding="utf-8") as f:
            for custom_id, result in results.items():
                answer_text = ""
                if result["type"] == "succeeded":
                    text_blocks = [b.get("text") for b in result["message"]["content"] if b.get("type") == "text"]
                    if text_blocks:
                        answer_text = text_blocks[0]
                else:
                    answer_text = f"ERROR: {result.get('error', 'Unknown error')}"

                data = meta[custom_id]
                rec = {
                    "custom_id": custom_id,
                    "model": model_name,
                    "id": data.get("id"),
                    "question": data.get("question"),
                    "gold_answer": data.get("answer"),
                    "prompt": prompts_dict[custom_id],
                    "output": answer_text,
                    "batch_id": batch_id,
                    "result_type": result["type"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[SAVED] Results written to {answers_jsonl}")


if __name__ == "__main__":
    main()
