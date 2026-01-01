import os
import time
import json
import yaml
import tenacity
import asyncio
from tqdm import tqdm
from google import genai
from functools import partial
from google.genai import types
from prompts.PROMPTS import *
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from google.genai import errors as genai_errors
from google.api_core import retry
from aiolimiter import AsyncLimiter
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from typing import List


# Some important hyperparameters
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Microsoft Azure Client setup (1st one is for gpt4.1, second one is for all other azure api models)
AZURE_API_KEY = "AZURE_API_KEY"
endpoint = "endpoint"
api_version = "api_version"
azure_client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=AZURE_API_KEY,
)

azure_generic_client = ChatCompletionsClient(
    endpoint="endpoint",
    credential=AzureKeyCredential(AZURE_API_KEY),
    api_version="api_version"
)


def load_dataset(basefolder: str, dataset_type: str, language: str, index=0):
    """ Loads dataset for specified dataset-type + langauge"""

    if dataset_type == "original-mgsm":
        dataset_path = os.path.join(basefolder, "original_dataset", "MGSM", language, "test.jsonl") 
    elif dataset_type == "original-afrimgsm":
        dataset_path = os.path.join(basefolder, "original_dataset", "AFRI_MGSM", language, "test.jsonl") 
    elif dataset_type == "ic-names":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "IC", f"{language}_ic_names_rank1.jsonl")
    elif dataset_type == "ic-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "IC", f"{language}_ic_numbers_rank1.jsonl")
    elif dataset_type == "ic-names-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "IC", f"{language}_ic_names_numbers_rank1.jsonl") 
    elif dataset_type == "symbolic-names":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "symbolic", f"{language}_symbolic_names_rank1.jsonl")
    elif dataset_type == "symbolic-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "symbolic", f"{language}_symbolic_numbers_rank1.jsonl")
    elif dataset_type == "symbolic-names-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "symbolic", f"{language}_symbolic_names_numbers_rank1.jsonl") 
    elif dataset_type == "distribution-ic":
        dataset_path = os.path.join(basefolder, "distribution_dataset", language, "IC", f"distribution_dataset_{index}.jsonl") 
    elif dataset_type == "distribution-symbolic":
        dataset_path = os.path.join(basefolder, "distribution_dataset", language, "symbolic", f"distribution_dataset_{index}.jsonl") 
    elif dataset_type == "debug":
        dataset_path = "/home/mila/x/xut/github/evaluation-pipeline/dataset/debug/test.jsonl"

    print(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data



def is_retryable(e) -> bool:
    # Check for general transient errors (e.g., networking issues)
    if retry.if_transient_error(e):
        return True
    # Catch 429 quota/resource exhausted errors
    elif (isinstance(e, genai_errors.ClientError) and e.code == 429):
        return True
    # Catch 503 model overloaded errors
    elif (isinstance(e, genai_errors.ServerError) and e.code == 503):
        return True
    # Catch 500 internal server errors (often transient)
    elif (isinstance(e, genai_errors.ServerError) and e.code == 500):
        return True
    else:
        return False

def construct_prompt(question: str, prompt_list = [0, 1, 2, 3, 4, 5]):
    """Construct prompts based on specified prompt IDs"""
    final_list_of_prompt = []

    for prompt_num in prompt_list:
        var_name = f"cot_prompt_{prompt_num}"
        if var_name in globals():
            prompt_template = globals()[var_name]
            curr_prompt = prompt_template.format(question=question)
            final_list_of_prompt.append(curr_prompt)
        else:
            print(f"Warning: Prompt template '{var_name}' not found in globals()")
    
    return final_list_of_prompt


def load_vllm(model_name: str, model_max_token: int, model_temperature: float):

    # setup VLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,  
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=model_max_token
    )
    # Sampling parameters for generation
    sampling_params = SamplingParams(
        temperature=model_temperature,
        max_tokens=model_max_token,
        top_p=1.0,
        top_k=-1,  
    )

    return llm, sampling_params

def solve_vllm(llm, sampling_param, data_langauge_list, result_basedir, dataset_type, language, model_name, prompt_ids, index=0):

    all_langauge_result = []

    for data_language in data_langauge_list:
        result = []
        all_prompts = []
        for curr_data in data_language:
            id = curr_data["id"]
            question = curr_data["question"]
            answer = curr_data["answer"]
            prompts = construct_prompt(question=question, prompt_list=prompt_ids)

            result.append({
                "id": id,
                "question": question,
                "answer": answer,
                "prompts": prompts,
                "outputs": []
            })
            all_prompts.extend(prompts)

        results = llm.generate(all_prompts, sampling_param)

        assert len(results) == len(
            all_prompts
        ), f"vLLM returned {len(results)} results but {len(all_prompts)} prompts were sent."

        # Map outputs back question‑by‑question
        cursor = 0
        for q_obj in result:
            n_variants = len(q_obj["prompts"])
            q_obj["outputs"] = [
                results[cursor + k].outputs[0].text for k in range(n_variants)
            ]
            cursor += n_variants
        # After the loop, cursor should equal len(results)
        assert cursor == len(results)

        all_langauge_result.append(result)
        
        # Log immediately after processing
        log_result(result_basedir=result_basedir, dataset_type=dataset_type, 
                   language=language, result=result, model_name=model_name, index=index)

    return all_langauge_result

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
def solve_azure(model_name: str, model_temperature: float, model_max_token: int, data_langauge_list, 
                result_basedir, dataset_type, language, model_name_log, prompt_ids, index=0):

    all_langauge_result = []

    for data_language in data_langauge_list:
        result = []
        all_prompts = []
        for curr_data in data_language:
            id = curr_data["id"]
            question = curr_data["question"]
            answer = curr_data["answer"]
            prompts = construct_prompt(question=question, prompt_list=prompt_ids)

            result.append({
                "id": id,
                "question": question,
                "answer": answer,
                "prompts": prompts,
                "outputs": []
            })
            all_prompts.extend(prompts)
        
        results = []
        desc = f"Generating ({model_name})"
        with tqdm(total=len(all_prompts), desc=desc, unit="prompt", leave=False, dynamic_ncols=True) as pbar:
            if model_name == "gpt-4.1":
                for prompt in all_prompts:
                    curr_result = azure_client.chat.completions.create(
                        messages=[UserMessage(content=prompt)],
                        max_completion_tokens=model_max_token,
                        temperature=model_temperature,
                        model=model_name
                    )
                    results.append(curr_result.choices[0].message.content)
                    pbar.update(1)
            else:
                for prompt in all_prompts:
                    curr_result = azure_generic_client.complete(
                        messages=[UserMessage(content=prompt)],
                        max_tokens=model_max_token,
                        temperature=model_temperature,
                        model=model_name
                    )
                    results.append(curr_result.choices[0].message.content)
                    pbar.update(1)


        assert len(results) == len(
            all_prompts
        ), f"Azure returned {len(results)} results but {len(all_prompts)} prompts were sent."

        # Map outputs back question‑by‑question
        cursor = 0
        for q_obj in result:
            n_variants = len(q_obj["prompts"])
            q_obj["outputs"] = [
                results[cursor + k] for k in range(n_variants)
            ]
            cursor += n_variants
        # After the loop, cursor should equal len(results)
        assert cursor == len(results)

        all_langauge_result.append(result)
        
        # Log immediately after processing
        log_result(result_basedir=result_basedir, dataset_type=dataset_type, 
                   language=language, result=result, model_name=model_name_log, index=index)

    return all_langauge_result

def solve_gemini(model_name: str, model_temperature: float, model_max_token: int, data_langauge_list, thinking_budget,
                 result_basedir, dataset_type, language, model_name_log, prompt_ids, index=0):

    all_langauge_result = []

    for data_language in data_langauge_list:
        result = []
        all_prompts = []
        for curr_data in data_language:
            id = curr_data["id"]
            question = curr_data["question"]
            answer = curr_data["answer"]
            prompts = construct_prompt(question=question, prompt_list=prompt_ids)

            result.append({
                "id": id,
                "question": question,
                "answer": answer,
                "prompts": prompts,
                "outputs": []
            })
            all_prompts.extend(prompts)

        with ThreadPoolExecutor(max_workers=10) as executor:
            solve_func = partial(
                solve_single_question,
                model_name=model_name,
                max_tokens=model_max_token,
                thinking_budget=thinking_budget
            )
            results = list(executor.map(solve_func, all_prompts))


        assert len(results) == len(
            all_prompts
        ), f"Gemini returned {len(results)} results but {len(all_prompts)} prompts were sent."

        # Map outputs back question‑by‑question
        cursor = 0
        for q_obj in result:
            n_variants = len(q_obj["prompts"])
            q_obj["outputs"] = [
                results[cursor + k] for k in range(n_variants)
            ]
            cursor += n_variants
        # After the loop, cursor should equal len(results)
        assert cursor == len(results)

        all_langauge_result.append(result)
        
        # Log immediately after processing
        log_result(result_basedir=result_basedir, dataset_type=dataset_type, 
                   language=language, result=result, model_name=model_name_log, index=index)

    return all_langauge_result





@retry.Retry(predicate=is_retryable,
            initial=1.0,
            multiplier=2.0,
            maximum=30.0,
            deadline=300.0 )
def solve_single_question(question: str, model_name: str, max_tokens: int, thinking_budget:int):

    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
        )
    )
    print(response.text)
    return response.text

def log_result(result_basedir: str, dataset_type: str, language: str, result: List, model_name: str, index=0):

    if dataset_type == "original-mgsm":
        dataset_path = os.path.join(result_basedir, "original_dataset", "MGSM", model_name, language, "raw_result.jsonl") 
    elif dataset_type == "original-afrimgsm":
        dataset_path = os.path.join(result_basedir, "original_dataset","AFRI_MGSM", model_name, language, "raw_result.jsonl") 
    elif dataset_type == "ic-names":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "names.jsonl")
    elif dataset_type == "ic-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "numbers.jsonl")
    elif dataset_type == "ic-names-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "names-numbers.jsonl") 
    elif dataset_type == "symbolic-names":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "names.jsonl")
    elif dataset_type == "symbolic-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "numbers.jsonl")
    elif dataset_type == "symbolic-names-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "names-numbers.jsonl") 
    elif dataset_type == "distribution-ic":
        dataset_path = os.path.join(result_basedir, "distribution_dataset", model_name, language, "IC", f"distribution_result_{index}.jsonl") 
    elif dataset_type == "distribution-symbolic":
        dataset_path = os.path.join(result_basedir, "distribution_dataset", model_name, language, "symbolic", f"distribution_result_{index}.jsonl") 

    elif dataset_type == "debug":
        dataset_path = "/home/mila/x/xut/github/evaluation-pipeline/result/debug/raw_result_gemini.jsonl"

    print(dataset_path)

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    print(f"[log_result] Writing {len(result)} entries to: {dataset_path}")

    # Write (overwrite) JSONL with just the requested fields
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


def result_exists(result_basedir, dataset_type, language, model_name, index=0):
    """Check if result file already exists for given parameters"""
    if dataset_type == "original-mgsm":
        path = os.path.join(result_basedir, "original_dataset", "MGSM", model_name, language, "raw_result.jsonl") 
    elif dataset_type == "original-afrimgsm":
        path = os.path.join(result_basedir, "original_dataset", "AFRI_MGSM", model_name, language, "raw_result.jsonl") 
    elif dataset_type == "ic-names":
        path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "names.jsonl")
    elif dataset_type == "ic-numbers":
        path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "numbers.jsonl")
    elif dataset_type == "ic-names-numbers":
        path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "names-numbers.jsonl") 
    elif dataset_type == "symbolic-names":
        path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "names.jsonl")
    elif dataset_type == "symbolic-numbers":
        path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "numbers.jsonl")
    elif dataset_type == "symbolic-names-numbers":
        path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "names-numbers.jsonl") 
    elif dataset_type == "distribution-ic":
        path = os.path.join(result_basedir, "distribution_dataset", model_name, language, "IC", f"distribution_result_{index}.jsonl") 
    elif dataset_type == "distribution-symbolic":
        path = os.path.join(result_basedir, "distribution_dataset", model_name, language, "symbolic", f"distribution_result_{index}.jsonl") 
    elif dataset_type == "debug":
        path = "/home/mila/x/xut/github/evaluation-pipeline/result/debug/raw_result_gemini.jsonl"
    else:
        return False
    
    exists = os.path.exists(path)
    if exists:
        print(f"[SKIP] Result already exists: {path}")
    return exists


def main():

    # Load YAML files
    parser = ArgumentParser(description="Evaluation runner")
    parser.add_argument( "--config",  type=str,  required=True,  help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_dataset_folder = config["base_dataset_folder"]
    dataset_type = config["dataset_type"]
    language = config["language"]
    model_platform = config["model_platform"]
    model_name = config["model_name"]
    model_max_token = config["model_max_token"]
    model_temperature = config["model_temperature"]
    result_basedir = config["result_basedir"]
    thinking_budget = int(config.get("thinking_budget") or 0)
    
    # Get prompt IDs to use, default to all 6 prompts if not specified
    prompt_ids = config.get("prompt_id_to_use", [0, 1, 2, 3, 4, 5])
    print(f"Using prompt IDs: {prompt_ids}")

    
    all_language_data = []
    for lang in language:

        if (config["dataset_type"] == "distribution-ic") or (config["dataset_type"] == "distribution-symbolic"):
            
            curr_lang_data = []

            for i in range(config["total_num_variation"]):
                # Skip if result already exists
                if result_exists(result_basedir, dataset_type, lang, model_name, index=i):
                    continue
                    
                curr_data = load_dataset(base_dataset_folder, dataset_type, lang, index=i)
                curr_lang_data.append((i, curr_data))  # Store index with data
            
            if curr_lang_data:  # Only add if there's data to process
                all_language_data.append((lang, curr_lang_data))

        else:
            # Skip if result already exists
            if result_exists(result_basedir, dataset_type, lang, model_name):
                continue
                
            curr_data = load_dataset(base_dataset_folder, dataset_type, lang)
            all_language_data.append((lang, curr_data))

    # Exit early if nothing to process
    if not all_language_data:
        print("[SKIP] All results already exist. Exiting.")
        return

    if (config["dataset_type"] == "distribution-ic") or (config["dataset_type"] == "distribution-symbolic"):

        if model_platform == "vllm":
            llm, sampling_param = load_vllm(model_name=model_name, model_max_token=model_max_token, model_temperature=model_temperature)

            for lang, lang_data in all_language_data:
                for idx, data in lang_data:
                    print(f"Processing {lang} - variation {idx}")
                    solve_vllm(llm=llm, sampling_param=sampling_param, data_langauge_list=[data],
                              result_basedir=result_basedir, dataset_type=dataset_type,
                              language=lang, model_name=model_name, prompt_ids=prompt_ids, index=idx)

        elif model_platform == "gemini":
            for lang, lang_data in all_language_data:
                for idx, data in lang_data:
                    print(f"Processing {lang} - variation {idx}")
                    solve_gemini(model_name=model_name, model_temperature=model_temperature, 
                                model_max_token=model_max_token, data_langauge_list=[data], 
                                thinking_budget=thinking_budget, result_basedir=result_basedir,
                                dataset_type=dataset_type, language=lang, model_name_log=model_name, 
                                prompt_ids=prompt_ids, index=idx)

        elif model_platform == "AzureAPI":
            for lang, lang_data in all_language_data:
                for idx, data in lang_data:
                    print(f"Processing {lang} - variation {idx}")
                    solve_azure(model_name=model_name, model_temperature=model_temperature, 
                               model_max_token=model_max_token, data_langauge_list=[data],
                               result_basedir=result_basedir, dataset_type=dataset_type,
                               language=lang, model_name_log=model_name, prompt_ids=prompt_ids, index=idx)

    else:
        if model_platform == "vllm":
            llm, sampling_param = load_vllm(model_name=model_name, model_max_token=model_max_token, model_temperature=model_temperature)
            
            for lang, data in all_language_data:
                print(f"Processing language: {lang}")
                solve_vllm(llm=llm, sampling_param=sampling_param, data_langauge_list=[data],
                          result_basedir=result_basedir, dataset_type=dataset_type,
                          language=lang, model_name=model_name, prompt_ids=prompt_ids, index=0)
                
        elif model_platform == "gemini":
            for lang, data in all_language_data:
                print(f"Processing language: {lang}")
                solve_gemini(model_name=model_name, model_temperature=model_temperature, 
                            model_max_token=model_max_token, data_langauge_list=[data], 
                            thinking_budget=thinking_budget, result_basedir=result_basedir,
                            dataset_type=dataset_type, language=lang, model_name_log=model_name, 
                            prompt_ids=prompt_ids, index=0)
                
        elif model_platform == "AzureAPI":
            for lang, data in all_language_data:
                print(f"Processing language: {lang}")
                solve_azure(model_name=model_name, model_temperature=model_temperature, 
                           model_max_token=model_max_token, data_langauge_list=[data],
                           result_basedir=result_basedir, dataset_type=dataset_type,
                           language=lang, model_name_log=model_name, prompt_ids=prompt_ids, index=0)


if __name__ == "__main__":
    main()