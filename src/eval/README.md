# Evaluation


- **`general_model_eval/`**: synchronous evaluation. Supports `vllm` (local GPU
  inference), `gemini`, `AzureAPI`, and `openai` via `model_platform` in the config.
  ```
  python -m general_model_eval.eval_model --config general_model_eval/config/<model>/<task>.yaml
  ```
- **`claude_batch_eval/`**: 
  ```
  python -m claude_batch_eval.submit_batch_eval --config claude_batch_eval/config/<model>/<task>.yaml
  ```
- **`azure_batch_eval/`**: 
  ```
  python -m azure_batch_eval.submit_batch_eval --config azure_batch_eval/config/<model>/<task>.yaml

  python -m azure_batch_eval.manage_batch_jobs list-pending --base-dir <result_dir>
  ```


## Config files

Each config is a YAML file with:

```yaml
base_dataset_folder: /path/to/dataset   # see dataset_paths.py for the expected layout
dataset_type: original-mgsm             # original-mgsm, original-afrimgsm, ic-names, ic-numbers,
                                         # ic-names-numbers, symbolic-names, symbolic-numbers,
                                         # symbolic-names-numbers, distribution-ic, distribution-symbolic
language: [english, french]
prompt_id_to_use: [english]             # names from prompts/prompts.py (cot_prompt_<name>)
n_shots: 8
total_num_variation: 1
model_name: gemini-2.5-flash
model_max_token: 4096
model_temperature: 0.1
result_basedir: /path/to/result
```

