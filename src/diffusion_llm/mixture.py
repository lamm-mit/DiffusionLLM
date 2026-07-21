"""Build reproducible, exact-size chat mixtures from Hugging Face datasets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    List,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

MIXTURE_FEATURES = Features(
    {
        "messages": List(
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ),
        "source": Value("string"),
        "license": Value("string"),
    }
)


@dataclass(frozen=True)
class SourceSpec:
    """One dataset source described by a mixture manifest."""

    dataset: str
    config: str | None = None
    train_split: str = "train"
    validation_split: str | None = "test"
    label: str | None = None
    license: str = "unknown"
    max_train_rows: int | None = None
    validation_rows: int = 0
    fill: bool = False
    preserve_original_source: bool = False
    revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("Every mixture source requires a non-empty 'dataset'.")
        if not isinstance(self.train_split, str) or not self.train_split.strip():
            raise ValueError(f"Source {self.dataset!r} requires a non-empty train_split.")
        if self.max_train_rows is not None and self.max_train_rows < 1:
            raise ValueError(f"max_train_rows for {self.dataset!r} must be positive.")
        if self.validation_rows < 0:
            raise ValueError(f"validation_rows for {self.dataset!r} cannot be negative.")

    @property
    def source_label(self) -> str:
        """Stable provenance label written into output rows."""
        if self.label:
            return self.label
        if self.config:
            return f"{self.dataset}:{self.config}"
        return self.dataset


@dataclass(frozen=True)
class MixtureBuildConfig:
    """Runtime controls for building and publishing a mixture."""

    manifest: str
    target_train_rows: int
    save_to_disk: str | None = None
    push_to_hub: bool = False
    hub_dataset_id: str | None = None
    hub_config_name: str = "default"
    hub_private: bool = False
    max_shard_size: str = "500MB"
    num_proc: int = 1
    upload_num_proc: int = 1
    seed: int = 42
    cache_dir: str | None = None
    validation_rows_per_source: int | None = None
    max_validation_rows: int | None = None
    allowed_roles: tuple[str, ...] = ("system", "user", "assistant")
    require_final_assistant: bool = True

    def __post_init__(self) -> None:
        if self.target_train_rows < 1:
            raise ValueError("--target-train-rows must be positive.")
        if self.num_proc < 1:
            raise ValueError("--num-proc must be positive.")
        if self.upload_num_proc < 1:
            raise ValueError("--upload-num-proc must be positive.")
        if self.validation_rows_per_source is not None:
            if self.validation_rows_per_source < 0:
                raise ValueError("--validation-rows-per-source cannot be negative.")
        if self.max_validation_rows is not None and self.max_validation_rows < 1:
            raise ValueError("--max-validation-rows must be positive.")
        if not self.allowed_roles:
            raise ValueError("At least one --allowed-role is required.")
        if not self.save_to_disk and not self.push_to_hub:
            raise ValueError("Choose --save-to-disk, --push-to-hub, or both.")
        if self.push_to_hub and not self.hub_dataset_id:
            raise ValueError("--push-to-hub requires --hub-dataset-id.")
        if not self.push_to_hub and (self.hub_dataset_id or self.hub_private):
            raise ValueError("Hub destination options require --push-to-hub.")


def load_manifest(path: str | Path) -> list[SourceSpec]:
    """Load and validate a JSON source manifest."""
    manifest_path = Path(path).expanduser()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Mixture manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in mixture manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError("A mixture manifest must be an object containing a 'sources' list.")
    if not payload["sources"]:
        raise ValueError("A mixture manifest must contain at least one source.")

    allowed_fields = {field.name for field in fields(SourceSpec)}
    specs = []
    for index, item in enumerate(payload["sources"]):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest source {index} must be a JSON object.")
        unknown = sorted(set(item) - allowed_fields)
        if unknown:
            raise ValueError(
                f"Manifest source {index} has unknown fields: {', '.join(unknown)}"
            )
        try:
            specs.append(SourceSpec(**item))
        except TypeError as exc:
            raise ValueError(f"Invalid manifest source {index}: {exc}") from exc

    fillers = [spec for spec in specs if spec.fill]
    if len(fillers) > 1:
        raise ValueError("A mixture manifest can mark at most one source with fill=true.")
    return specs


def _is_valid_chat_row(
    row: dict[str, Any],
    *,
    allowed_roles: tuple[str, ...],
    require_final_assistant: bool,
) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if require_final_assistant:
        if not isinstance(messages[-1], dict):
            return False
        if messages[-1].get("role") != "assistant":
            return False
    return all(
        isinstance(message, dict)
        and message.get("role") in allowed_roles
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def _normalize_chat_row(
    row: dict[str, Any],
    *,
    source_label: str,
    license_name: str,
    preserve_original_source: bool,
) -> dict[str, Any]:
    label = source_label
    if preserve_original_source:
        original_source = str(row.get("source") or "unknown")
        label = f"{source_label}:{original_source}"
    return {
        "messages": [
            {
                "role": str(message["role"]),
                "content": str(message["content"]).strip(),
            }
            for message in row["messages"]
        ],
        "source": label,
        "license": license_name,
    }


def _load_split(spec: SourceSpec, split: str, cache_dir: str | None) -> Dataset:
    kwargs: dict[str, Any] = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if spec.revision:
        kwargs["revision"] = spec.revision
    dataset = load_dataset(spec.dataset, spec.config, **kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected one Dataset for {spec.source_label}:{split}.")
    return dataset


def _clean_split(
    dataset: Dataset,
    spec: SourceSpec,
    config: MixtureBuildConfig,
    *,
    description: str,
) -> Dataset:
    workers = config.num_proc if config.num_proc > 1 else None
    original_columns = dataset.column_names
    dataset = dataset.filter(
        _is_valid_chat_row,
        fn_kwargs={
            "allowed_roles": config.allowed_roles,
            "require_final_assistant": config.require_final_assistant,
        },
        num_proc=workers,
        desc=f"Validating {description}",
    )
    return dataset.map(
        _normalize_chat_row,
        fn_kwargs={
            "source_label": spec.source_label,
            "license_name": spec.license,
            "preserve_original_source": spec.preserve_original_source,
        },
        remove_columns=original_columns,
        features=MIXTURE_FEATURES,
        num_proc=workers,
        desc=f"Normalizing {description}",
    )


def _validation_rows(spec: SourceSpec, config: MixtureBuildConfig) -> int:
    if config.validation_rows_per_source is not None:
        return config.validation_rows_per_source
    return spec.validation_rows


def _select_rows(dataset: Dataset, count: int) -> Dataset:
    return dataset.select(range(count))


def build_mixture(config: MixtureBuildConfig) -> DatasetDict:
    """Build an exact-size mixture and a disjoint validation split."""
    specs = load_manifest(config.manifest)
    fixed_parts: list[Dataset] = []
    validation_parts: list[Dataset] = []
    fill_pool: tuple[SourceSpec, Dataset, int] | None = None

    for index, spec in enumerate(specs):
        print(f"Loading training source: {spec.source_label}")
        train = _load_split(spec, spec.train_split, config.cache_dir)
        train = _clean_split(train, spec, config, description=spec.source_label)
        train = train.shuffle(seed=config.seed + index)

        requested_validation = _validation_rows(spec, config)
        reserved_from_train = requested_validation if spec.validation_split is None else 0
        if reserved_from_train > len(train):
            raise ValueError(
                f"Source {spec.source_label!r} has {len(train):,} valid rows but "
                f"requests {reserved_from_train:,} validation rows."
            )
        train_capacity = len(train) - reserved_from_train
        if spec.max_train_rows is not None:
            train_capacity = min(train_capacity, spec.max_train_rows)

        if reserved_from_train:
            start = len(train) - reserved_from_train
            validation_parts.append(train.select(range(start, len(train))))

        if spec.validation_split is not None and requested_validation:
            validation = _load_split(spec, spec.validation_split, config.cache_dir)
            validation = _clean_split(
                validation,
                spec,
                config,
                description=f"{spec.source_label}:{spec.validation_split}",
            ).shuffle(seed=config.seed + 10_000 + index)
            validation_parts.append(
                _select_rows(validation, min(requested_validation, len(validation)))
            )

        if spec.fill:
            fill_pool = (spec, train, train_capacity)
            continue
        if train_capacity:
            fixed_parts.append(_select_rows(train, train_capacity))

    fixed_count = sum(len(part) for part in fixed_parts)
    if fill_pool is not None:
        if fixed_count > config.target_train_rows:
            raise ValueError(
                f"Fixed sources contain {fixed_count:,} rows, exceeding target "
                f"{config.target_train_rows:,}. Add max_train_rows caps to the manifest."
            )
        fill_spec, fill_dataset, fill_capacity = fill_pool
        needed = config.target_train_rows - fixed_count
        if needed > fill_capacity:
            raise ValueError(
                f"Fill source {fill_spec.source_label!r} can provide {fill_capacity:,} "
                f"training rows, but {needed:,} are required."
            )
        if needed:
            fixed_parts.append(_select_rows(fill_dataset, needed))

    if not fixed_parts:
        raise ValueError("The manifest and target produced no training rows.")

    train = concatenate_datasets(fixed_parts).shuffle(seed=config.seed)
    if len(train) < config.target_train_rows:
        raise ValueError(
            f"Sources provide {len(train):,} rows, fewer than target "
            f"{config.target_train_rows:,}. Add a fill source or more data."
        )
    if len(train) > config.target_train_rows:
        train = _select_rows(train, config.target_train_rows)

    result = DatasetDict({"train": train})
    if validation_parts:
        validation = concatenate_datasets(validation_parts).shuffle(seed=config.seed)
        if config.max_validation_rows is not None:
            validation = _select_rows(
                validation,
                min(config.max_validation_rows, len(validation)),
            )
        result["validation"] = validation

    print(result)
    print(f"Training rows: {len(result['train']):,}")
    if "validation" in result:
        print(f"Validation rows: {len(result['validation']):,}")
    print("Licenses:", dict(Counter(result["train"]["license"])))
    return result


def write_mixture_outputs(mixture: DatasetDict, config: MixtureBuildConfig) -> None:
    """Save a recovery copy before optionally uploading the mixture."""
    output_workers = config.upload_num_proc if config.upload_num_proc > 1 else None
    if config.save_to_disk:
        output = Path(config.save_to_disk).expanduser().resolve()
        if output.exists():
            raise FileExistsError(
                f"Local dataset output already exists: {output}. Choose a new path."
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving recoverable local dataset: {output}")
        mixture.save_to_disk(str(output), num_proc=output_workers)

    if config.push_to_hub:
        print(
            f"Uploading {config.hub_dataset_id}:{config.hub_config_name} "
            f"with {config.upload_num_proc} process(es)"
        )
        mixture.push_to_hub(
            config.hub_dataset_id,
            config_name=config.hub_config_name,
            private=config.hub_private,
            max_shard_size=config.max_shard_size,
            num_proc=output_workers,
        )


def build_and_write_mixture(config: MixtureBuildConfig) -> DatasetDict:
    """Build a mixture, save it locally, and optionally publish it."""
    mixture = build_mixture(config)
    write_mixture_outputs(mixture, config)
    return mixture


def upload_saved_mixture(
    dataset_path: str,
    *,
    hub_dataset_id: str,
    hub_config_name: str = "default",
    hub_private: bool = False,
    max_shard_size: str = "500MB",
    num_proc: int = 1,
) -> DatasetDict:
    """Upload a previously saved mixture without rebuilding source data."""
    if num_proc < 1:
        raise ValueError("--num-proc must be positive.")
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Saved mixture not found: {path}")
    mixture = load_from_disk(str(path))
    if not isinstance(mixture, DatasetDict):
        raise ValueError(f"Saved mixture must be a DatasetDict: {path}")
    if "train" not in mixture:
        raise ValueError(f"Saved mixture has no train split: {path}")
    workers = num_proc if num_proc > 1 else None
    print(
        f"Uploading {hub_dataset_id}:{hub_config_name} with {num_proc} process(es) "
        f"from {path}"
    )
    mixture.push_to_hub(
        hub_dataset_id,
        config_name=hub_config_name,
        private=hub_private,
        max_shard_size=max_shard_size,
        num_proc=workers,
    )
    return mixture
