# MGSM-Pro

[![arXiv](https://img.shields.io/badge/arXiv-2601.21225-b31b1b.svg)](https://arxiv.org/abs/2601.21225)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-mgsm--pro-yellow)](https://huggingface.co/datasets/McGill-NLP/mgsm-pro)
[![Conference](https://img.shields.io/badge/Accepted%20to-AACL%202026-blue)](https://arxiv.org/abs/2601.21225)

MGSM-Pro is an extension of MGSM dataset with GSM-Symbolic approach. The dataset evaluates multilingual mathematical robustness in LLMs. This repository contains the tools to build the dataset as well as the code to evaluate models used for the release. 

- Paper: [MGSM-Pro: A Simple Strategy for Robust Multilingual Mathematical Reasoning Evaluation](https://arxiv.org/abs/2601.21225)
- Dataset: [McGill-NLP mgsm-pro](https://huggingface.co/datasets/McGill-NLP/mgsm-pro)


## Environment setup

```bash
conda create -n venv python=3.10 -y
conda activate venv
pip install -r requirements.txt
```

After environment, setup API keys for evaluation. 

```bash
export GOOGLE_API_KEY=...        # gemini
export OPENAI_API_KEY=...        # openai
export ANTHROPIC_API_KEY=...     # claude_batch_eval
export AZURE_API_KEY=...         # AzureAPI (sync)
export AZURE_OPENAI_ENDPOINT=...
export AZURE_INFERENCE_ENDPOINT=...
```


## Dataset Construction

The tools to construct MGSM-Pro are located in [data_construction_tool/](data_construction_tool/):
- `template/<language>/<id>.json`:  problem templates with placeholders
- `native_names/<language>_words.json` — names to substitute in
- `numerical_combination/template/<id>.json`: the rules to generate the digits for each template
- `numerical_combination/combination/<id>.json`: the numerical values generated for each template



### Step 1: generate numerical combinations (optional)

The repository already contains 10 numerical combinations per template. But, you can always generate more with the following: 

```bash
cd src/digit_generation

python generate_combinations.py --problem_range 0 249 --num_combinations 10
```

Run `python generate_combinations.py --help` for all options 

### Step 2: generate the dataset

```bash
cd src/dataset_creation

python run_generator.py \
  --template-type ic \
  --mode N# \
  --num-combos 5 \
  --languages english,french,chinese \
  --output ./dataset/my_dataset
```
Some important arguments are the following: 
- `--template-type`: `symbolic` or `ic`
- `--mode`: `N` (names only), `#` (numbers only), or `N#` (both names and numbers)
- `--consistent-names`: reuse the same substituted names across all instances

Run `python run_generator.py --help` for the full option list.

## 3. Evaluating a model

See [src/eval/README.md](src/eval/README.md) for the full backend list,
required environment variables per backend, and config format. Quick start:

```bash
cd src/eval

python -m general_model_eval.eval_model --config general_model_eval/config/gemini-2.5-flash/mgsm.yaml
```
