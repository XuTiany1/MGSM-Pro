import os
import json
import yaml
import tenacity
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from typing import List

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.api_core import retry

from vllm import LLM, SamplingParams

from openai import AzureOpenAI, OpenAI
from azure.ai.inference.models import UserMessage
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from prompts.prompts import *
from prompts.exemplars import get_exemplars_prefix
from dataset_paths import resolve_path, load_dataset, result_exists


# ---------------------------------------------------------------------------
# API clients
#
# Clients are created lazily (and cached) so that running with one
# model_platform never requires credentials for the others. All secrets and
# endpoints must be supplied via environment variables - see README for the
# expected variable names. Nothing here is hardcoded.
# ---------------------------------------------------------------------------

_clients = {}


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_gemini_client():
    if "gemini" not in _clients:
        _clients["gemini"] = genai.Client(api_key=_require_env("GOOGLE_API_KEY"))
    return _clients["gemini"]


def get_azure_clients():
    """Returns (chat_completions_client_for_gpt4.1, generic_inference_client)."""
    if "azure" not in _clients:
        api_key = _require_env("AZURE_API_KEY")
        azure_client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key,
        )
        azure_generic_client = ChatCompletionsClient(
            endpoint=_require_env("AZURE_INFERENCE_ENDPOINT"),
            credential=AzureKeyCredential(api_key),
            api_version=os.getenv("AZURE_INFERENCE_API_VERSION", "2024-05-01-preview"),
        )
        _clients["azure"] = (azure_client, azure_generic_client)
    return _clients["azure"]


def get_openai_client():
    if "openai" not in _clients:
        _clients["openai"] = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
    return _clients["openai"]


# ---------------------------------------------------------------------------
# Result logging (dataset/result path resolution lives in dataset_paths.py,
# shared with the other eval backends)
# ---------------------------------------------------------------------------

def log_result(result_basedir: str, dataset_type: str, language: str, result: List, model_name: str, index: int = 0):
    dataset_path = resolve_path(result_basedir, dataset_type, language, index, model_name=model_name)
    print(dataset_path)

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(f"[log_result] Writing {len(result)} entries to: {dataset_path}")

    with open(dataset_path, "w", encoding="utf-8") as f:
        for entry in result:
            line = {
                "id": entry.get("id"),
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "prompts": entry.get("prompts", []),
                "outputs": entry.get("outputs", []),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def construct_prompt(question: str, language: str, prompt_list=("english",), n_shots: int = 0):
    """prompt_list holds prompt names (e.g. "english"), matching a cot_prompt_<name> in prompts.py."""
    prefix = get_exemplars_prefix(language, n_shots)
    final_list_of_prompt = []

    for prompt_name in prompt_list:
        var_name = f"cot_prompt_{prompt_name}"
        if var_name in globals():
            curr_prompt = globals()[var_name].format(question=question)
            final_list_of_prompt.append(prefix + curr_prompt)
        else:
            print(f"Warning: Prompt template '{var_name}' not found in globals()")

    return final_list_of_prompt


def _build_prompt_batch(data_language: List[dict], language: str, prompt_ids, n_shots: int, prompt_transform=None):
    """Builds per-question result records and the flat list of prompts to send.

    `prompt_transform`, if given, is applied to each rendered prompt before it's
    sent to the model (and before it's logged, so results reflect what was sent).
    """
    result = []
    all_prompts = []

    for curr_data in data_language:
        prompts = construct_prompt(
            question=curr_data["question"],
            language=language,
            prompt_list=prompt_ids,
            n_shots=n_shots,
        )
        if prompt_transform is not None:
            prompts = [prompt_transform(p) for p in prompts]
        result.append({
            "id": curr_data["id"],
            "question": curr_data["question"],
            "answer": curr_data["answer"],
            "prompts": prompts,
            "outputs": [],
        })
        all_prompts.extend(prompts)

    return result, all_prompts


def _map_outputs_to_results(result: List[dict], outputs: List[str]):
    """Splits the flat `outputs` list back onto each question's `prompts`."""
    assert len(outputs) == sum(len(q["prompts"]) for q in result), (
        f"Got {len(outputs)} outputs but expected {sum(len(q['prompts']) for q in result)}."
    )

    cursor = 0
    for q_obj in result:
        n_variants = len(q_obj["prompts"])
        q_obj["outputs"] = outputs[cursor:cursor + n_variants]
        cursor += n_variants


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def load_vllm(model_name: str, model_max_token: int, model_temperature: float,
              tensor_parallel_size: int = 1, dtype: str = "bfloat16", enforce_eager: bool = False):
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=model_max_token,
        enforce_eager=enforce_eager,
    )
    sampling_params = SamplingParams(
        temperature=model_temperature,
        max_tokens=model_max_token,
        top_p=1.0,
        top_k=-1,
    )
    return llm, sampling_params


def _apply_reasoning_effort(tokenizer, prompt: str, reasoning_effort: str) -> str:
    """Renders `prompt` as a single user turn through the model's chat template,
    with a reasoning-effort directive (e.g. gpt-oss's Harmony format supports
    "low" / "medium" / "high", default "medium")."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        reasoning_effort=reasoning_effort,
        tokenize=False,
        add_generation_prompt=True,
    )


def solve_vllm(llm, sampling_param, data_language_list, result_basedir, dataset_type,
               language, model_name, prompt_ids, n_shots=0, index=0, reasoning_effort=None):

    all_language_result = []

    prompt_transform = None
    if reasoning_effort:
        tokenizer = llm.get_tokenizer()
        prompt_transform = partial(_apply_reasoning_effort, tokenizer, reasoning_effort=reasoning_effort)

    for data_language in data_language_list:
        result, all_prompts = _build_prompt_batch(data_language, language, prompt_ids, n_shots,
                                                   prompt_transform=prompt_transform)

        outputs = llm.generate(all_prompts, sampling_param)
        assert len(outputs) == len(all_prompts), (
            f"vLLM returned {len(outputs)} results but {len(all_prompts)} prompts were sent."
        )
        _map_outputs_to_results(result, [o.outputs[0].text for o in outputs])

        all_language_result.append(result)
        log_result(result_basedir=result_basedir, dataset_type=dataset_type,
                   language=language, result=result, model_name=model_name, index=index)

    return all_language_result


def is_retryable(e) -> bool:
    if retry.if_transient_error(e):
        return True
    if isinstance(e, genai_errors.ClientError) and e.code == 429:  # quota exhausted
        return True
    if isinstance(e, genai_errors.ServerError) and e.code in (500, 503):  # transient server errors
        return True
    return False


@retry.Retry(predicate=is_retryable, initial=1.0, multiplier=2.0, maximum=30.0, deadline=300.0)
def solve_single_question(question: str, client, model_name: str, max_tokens: int, thinking_budget: int):
    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        ),
    )
    return response.text


def solve_gemini(model_name: str, model_temperature: float, model_max_token: int, data_language_list, thinking_budget,
                 result_basedir, dataset_type, language, model_name_log, prompt_ids, n_shots=0, index=0):

    client = get_gemini_client()
    all_language_result = []

    for data_language in data_language_list:
        result, all_prompts = _build_prompt_batch(data_language, language, prompt_ids, n_shots)

        solve_func = partial(
            solve_single_question,
            client=client,
            model_name=model_name,
            max_tokens=model_max_token,
            thinking_budget=thinking_budget,
        )
        with ThreadPoolExecutor(max_workers=10) as executor:
            outputs = list(executor.map(solve_func, all_prompts))

        assert len(outputs) == len(all_prompts), (
            f"Gemini returned {len(outputs)} results but {len(all_prompts)} prompts were sent."
        )
        _map_outputs_to_results(result, outputs)

        all_language_result.append(result)
        log_result(result_basedir=result_basedir, dataset_type=dataset_type,
                   language=language, result=result, model_name=model_name_log, index=index)

    return all_language_result


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def solve_azure(model_name: str, model_temperature: float, model_max_token: int, data_language_list,
                result_basedir, dataset_type, language, model_name_log, prompt_ids, n_shots=0, index=0):

    azure_client, azure_generic_client = get_azure_clients()
    all_language_result = []

    for data_language in data_language_list:
        result, all_prompts = _build_prompt_batch(data_language, language, prompt_ids, n_shots)

        outputs = []
        desc = f"Generating ({model_name})"
        with tqdm(total=len(all_prompts), desc=desc, unit="prompt", leave=False, dynamic_ncols=True) as pbar:
            for prompt in all_prompts:
                if model_name == "gpt-4.1":
                    curr_result = azure_client.chat.completions.create(
                        messages=[UserMessage(content=prompt)],
                        max_completion_tokens=model_max_token,
                        temperature=model_temperature,
                        model=model_name,
                    )
                else:
                    curr_result = azure_generic_client.complete(
                        messages=[UserMessage(content=prompt)],
                        max_tokens=model_max_token,
                        temperature=model_temperature,
                        model=model_name,
                    )
                outputs.append(curr_result.choices[0].message.content)
                pbar.update(1)

        assert len(outputs) == len(all_prompts), (
            f"Azure returned {len(outputs)} results but {len(all_prompts)} prompts were sent."
        )
        _map_outputs_to_results(result, outputs)

        all_language_result.append(result)
        log_result(result_basedir=result_basedir, dataset_type=dataset_type,
                   language=language, result=result, model_name=model_name_log, index=index)

    return all_language_result


@tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_random_exponential(multiplier=1, min=1, max=60),
    reraise=True,
)
def solve_single_openai(client, model_name: str, model_max_token: int, prompt: str):
    curr_result = client.responses.create(
        model=model_name,
        reasoning={"effort": "minimal"},
        max_output_tokens=model_max_token,
        input=prompt,
    )
    return curr_result.output_text


def solve_openai(model_name: str, model_temperature: float, model_max_token: int, data_language_list,
                 result_basedir, dataset_type, language, model_name_log, prompt_ids, n_shots=0, index=0):

    client = get_openai_client()
    all_language_result = []

    for data_language in data_language_list:
        result, all_prompts = _build_prompt_batch(data_language, language, prompt_ids, n_shots)

        outputs = []
        desc = f"Generating ({model_name})"
        with tqdm(total=len(all_prompts), desc=desc, unit="prompt", leave=False, dynamic_ncols=True) as pbar:
            for prompt in all_prompts:
                output_text = solve_single_openai(
                    client=client,
                    model_name=model_name,
                    model_max_token=model_max_token,
                    prompt=prompt,
                )
                outputs.append(output_text)
                pbar.update(1)

        assert len(outputs) == len(all_prompts), (
            f"OpenAI returned {len(outputs)} results but {len(all_prompts)} prompts were sent."
        )
        _map_outputs_to_results(result, outputs)

        all_language_result.append(result)
        log_result(result_basedir=result_basedir, dataset_type=dataset_type,
                   language=language, result=result, model_name=model_name_log, index=index)

    return all_language_result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SOLVERS = {
    "gemini": solve_gemini,
    "AzureAPI": solve_azure,
    "openai": solve_openai,
}


def main():
    parser = ArgumentParser(description="Evaluation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_dataset_folder = config["base_dataset_folder"]
    dataset_type = config["dataset_type"]
    languages = config["language"]
    model_platform = config["model_platform"]
    model_name = config["model_name"]
    model_max_token = config["model_max_token"]
    model_temperature = config["model_temperature"]
    result_basedir = config["result_basedir"]
    thinking_budget = int(config.get("thinking_budget") or 0)
    reasoning_effort = config.get("reasoning_effort")

    prompt_ids = config.get("prompt_id_to_use") or ["english"]
    n_shots = int(config.get("n_shots", 0))
    print(f"Using prompts: {prompt_ids} with {n_shots} shots")

    all_language_data = []
    for lang in languages:
        curr_lang_data = []
        for i in range(config["total_num_variation"]):
            if result_exists(result_basedir, dataset_type, lang, model_name, index=i):
                continue
            curr_data = load_dataset(base_dataset_folder, dataset_type, lang, index=i)
            curr_lang_data.append((i, curr_data))

        if curr_lang_data:
            all_language_data.append((lang, curr_lang_data))

    if not all_language_data:
        print("[SKIP] All results already exist. Exiting.")
        return

    if model_platform == "vllm":
        tensor_parallel_size = int(config.get("tensor_parallel_size", 1))
        dtype = config.get("dtype", "bfloat16")
        enforce_eager = bool(config.get("enforce_eager", False))
        llm, sampling_param = load_vllm(
            model_name=model_name, model_max_token=model_max_token, model_temperature=model_temperature,
            tensor_parallel_size=tensor_parallel_size, dtype=dtype, enforce_eager=enforce_eager,
        )
        for lang, lang_data in all_language_data:
            for idx, data in lang_data:
                print(f"Processing {lang} - variation {idx}")
                solve_vllm(llm=llm, sampling_param=sampling_param, data_language_list=[data],
                          result_basedir=result_basedir, dataset_type=dataset_type,
                          language=lang, model_name=model_name, prompt_ids=prompt_ids,
                          n_shots=n_shots, index=idx, reasoning_effort=reasoning_effort)
        return

    solve_fn = _SOLVERS.get(model_platform)
    if solve_fn is None:
        raise ValueError(f"Unknown model_platform: {model_platform}")

    for lang, lang_data in all_language_data:
        for idx, data in lang_data:
            print(f"Processing {lang} - variation {idx}")
            solve_fn(model_name=model_name, model_temperature=model_temperature,
                    model_max_token=model_max_token, data_language_list=[data],
                    **({"thinking_budget": thinking_budget} if model_platform == "gemini" else {}),
                    result_basedir=result_basedir, dataset_type=dataset_type,
                    language=lang, model_name_log=model_name, prompt_ids=prompt_ids,
                    n_shots=n_shots, index=idx)


if __name__ == "__main__":
    main()
