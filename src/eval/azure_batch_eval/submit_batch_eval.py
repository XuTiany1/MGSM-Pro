"""
Submits an evaluation run to the Azure OpenAI Batch API.

Usage (run from src/eval/):
    python -m azure_batch_eval.submit_batch_eval --config azure_batch_eval/config/gpt-4.1/mgsm.yaml

Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.
"""

import os
import json
import yaml
from argparse import ArgumentParser

from openai import AzureOpenAI

from prompts.prompts import *
from prompts.exemplars import get_exemplars_prefix
from dataset_paths import resolve_dir, load_dataset

from azure_batch_eval.batch_manager import BatchMathManager


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
    parser = ArgumentParser(description="Azure OpenAI Batch API evaluation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_dataset_folder = config["base_dataset_folder"]
    dataset_type = config["dataset_type"]
    languages = config["language"]
    model_name = config["model_name"]  # Azure deployment name, e.g. "gpt-4.1"
    model_max_token = config["model_max_token"]
    model_temperature = config["model_temperature"]
    result_basedir = config["result_basedir"]

    prompt_ids = config.get("prompt_id_to_use") or ["english"]
    n_shots = int(config.get("n_shots", 0))
    print(f"Using prompts: {prompt_ids} with {n_shots} shots")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        raise RuntimeError("Missing required environment variable(s): AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")

    client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
        azure_endpoint=endpoint,
        api_key=api_key,
    )

    for lang in languages:
        for i in range(config["total_num_variation"]):
            dataset = load_dataset(base_dataset_folder, dataset_type, lang, index=i)
            result_dir = resolve_dir(result_basedir, dataset_type, lang, model_name=model_name)

            out_dir = os.path.join(result_dir, "results")
            answers_jsonl = os.path.join(out_dir, "answers.jsonl")
            if os.path.exists(answers_jsonl):
                print(f"[SKIPPING] - Path already exists {answers_jsonl}")
                continue

            manager = BatchMathManager(client=client, base_dir=result_dir, deployment=model_name)

            batch_texts = []
            meta = []  # parallel to batch_texts
            for data in dataset:
                prompts = construct_prompt(data["question"], lang, prompt_list=prompt_ids, n_shots=n_shots)
                for p_idx, prompt in enumerate(prompts):
                    batch_texts.append(prompt)
                    meta.append({
                        "id": data.get("id"),
                        "question": data.get("question"),
                        "answer": data.get("answer"),
                        "prompt_id": prompt_ids[p_idx],
                    })

            os.makedirs(out_dir, exist_ok=True)
            outputs, stats = manager.compute_math_batch(
                texts=batch_texts,
                temperature=model_temperature,
                max_tokens=model_max_token,
                check_interval=20,
                quiet=False,
            )
            print(f"[BATCH COMPLETE] {stats}")

            with open(answers_jsonl, "w", encoding="utf-8") as f:
                for idx, (m, prompt, output) in enumerate(zip(meta, batch_texts, outputs)):
                    rec = {
                        "idx": idx,
                        "id": m["id"],
                        "model": model_name,
                        "language": lang,
                        "n_shots": n_shots,
                        "prompt_id": m["prompt_id"],
                        "question": m["question"],
                        "prompt": prompt,
                        "gold_answer": m["answer"],
                        "output": output,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[SAVED] Results written to {answers_jsonl}")


if __name__ == "__main__":
    main()
