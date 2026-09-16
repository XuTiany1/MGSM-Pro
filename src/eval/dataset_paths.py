import json
import os


# dataset_type -> (group_dir, subgroup_dir), laid out as
#   <base>/<group>/<subgroup>/[<model_name>/]<language>/<filename>
_ORIGINAL_DATASET_TYPES = {
    "original-mgsm": ("original_dataset", "MGSM"),
    "original-afrimgsm": ("original_dataset", "AFRI_MGSM"),
}

# dataset_type -> (group_dir, subgroup_dir), laid out as
#   <base>/<group>/[<model_name>/]<language>/<subgroup>/<filename>
_VARIANT_DATASET_TYPES = {
    "ic-names": ("dataset_N", "IC"),
    "ic-numbers": ("dataset_#", "IC"),
    "ic-names-numbers": ("dataset_N#", "IC"),
    "symbolic-names": ("dataset_N", "SYMBOLIC"),
    "symbolic-numbers": ("dataset_#", "SYMBOLIC"),
    "symbolic-names-numbers": ("dataset_N#", "SYMBOLIC"),
    "distribution-ic": ("distribution_dataset", "IC"),
    "distribution-symbolic": ("distribution_dataset", "symbolic"),
}


def resolve_dir(base_dir: str, dataset_type: str, language: str, model_name: str = None) -> str:
    """Resolves the directory holding the dataset (model_name=None) or result
    (model_name set) file for the given dataset_type + language.

    When model_name is given, it is inserted right before the language segment,
    matching how results are organized per-model alongside the source datasets.
    """
    if dataset_type == "debug":
        parts = [base_dir, "debug"]
        if model_name is not None:
            parts.append(model_name)
        return os.path.join(*parts)

    if dataset_type in _ORIGINAL_DATASET_TYPES:
        group, subgroup = _ORIGINAL_DATASET_TYPES[dataset_type]
        parts = [base_dir, group, subgroup]
        if model_name is not None:
            parts.append(model_name)
        parts.append(language)
        return os.path.join(*parts)

    if dataset_type in _VARIANT_DATASET_TYPES:
        group, subgroup = _VARIANT_DATASET_TYPES[dataset_type]
        parts = [base_dir, group]
        if model_name is not None:
            parts.append(model_name)
        parts.extend([language, subgroup])
        return os.path.join(*parts)

    raise ValueError(f"Unknown dataset_type: {dataset_type}")


def resolve_path(base_dir: str, dataset_type: str, language: str, index: int = 0, model_name: str = None) -> str:
    """Resolves the full dataset (model_name=None) or result (model_name set) file path."""
    directory = resolve_dir(base_dir, dataset_type, language, model_name)

    if dataset_type == "debug" or dataset_type in _ORIGINAL_DATASET_TYPES:
        filename = "test.jsonl" if model_name is None else "raw_result.jsonl"
    else:
        filename = f"distribution_dataset_{index}.jsonl" if model_name is None else f"distribution_result_{index}.jsonl"

    return os.path.join(directory, filename)


def load_dataset(basefolder: str, dataset_type: str, language: str, index: int = 0):
    """Loads dataset for the specified dataset_type + language."""
    dataset_path = resolve_path(basefolder, dataset_type, language, index)
    print(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def result_exists(result_basedir: str, dataset_type: str, language: str, model_name: str, index: int = 0) -> bool:
    """Checks if a result file already exists for the given parameters."""
    path = resolve_path(result_basedir, dataset_type, language, index, model_name=model_name)
    exists = os.path.exists(path)
    if exists:
        print(f"[SKIP] Result already exists: {path}")
    return exists
