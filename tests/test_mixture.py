"""Tests for configurable chat-mixture construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from diffusion_llm import mixture as mixture_module
from diffusion_llm.mixture import (
    MixtureBuildConfig,
    build_and_write_mixture,
    load_manifest,
    write_mixture_outputs,
)


def _chat_dataset(prefix: str, count: int, *, original_source: bool = False) -> Dataset:
    rows = {
        "messages": [
            [
                {"role": "user", "content": f"{prefix} question {index}"},
                {"role": "assistant", "content": f"{prefix} answer {index}"},
            ]
            for index in range(count)
        ]
    }
    if original_source:
        rows["source"] = [f"partition-{index % 2}" for index in range(count)]
    return Dataset.from_dict(rows)


def test_build_mixture_hits_exact_target_and_reserves_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "dataset": "core",
                        "validation_split": "test",
                        "license": "MIT",
                        "max_train_rows": 3,
                        "validation_rows": 1,
                    },
                    {
                        "dataset": "filler",
                        "validation_split": None,
                        "license": "ODC-BY-1.0",
                        "validation_rows": 2,
                        "fill": True,
                        "preserve_original_source": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    datasets = {
        ("core", "train"): _chat_dataset("core", 5),
        ("core", "test"): _chat_dataset("core-test", 2),
        ("filler", "train"): _chat_dataset("fill", 10, original_source=True),
    }

    def fake_load_dataset(dataset, config, *, split, **kwargs):
        assert config is None
        return datasets[(dataset, split)]

    monkeypatch.setattr(mixture_module, "load_dataset", fake_load_dataset)
    local_output = tmp_path / "mixture"
    config = MixtureBuildConfig(
        manifest=str(manifest),
        target_train_rows=7,
        save_to_disk=str(local_output),
        num_proc=1,
    )

    result = build_and_write_mixture(config)
    restored = load_from_disk(str(local_output))

    assert isinstance(result, DatasetDict)
    assert len(result["train"]) == 7
    assert len(result["validation"]) == 3
    assert len(restored["train"]) == 7
    assert result["train"].features == mixture_module.MIXTURE_FEATURES
    assert set(result["train"]["license"]) == {"MIT", "ODC-BY-1.0"}
    assert any(
        source.startswith("filler:partition-") for source in result["train"]["source"]
    )

    train_answers = {
        row["messages"][-1]["content"]
        for row in result["train"]
        if row["source"].startswith("filler:")
    }
    validation_answers = {
        row["messages"][-1]["content"]
        for row in result["validation"]
        if row["source"].startswith("filler:")
    }
    assert train_answers.isdisjoint(validation_answers)


def test_upload_defaults_to_one_process(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"sources": [{"dataset": "unused"}]}', encoding="utf-8")
    config = MixtureBuildConfig(
        manifest=str(manifest),
        target_train_rows=1,
        push_to_hub=True,
        hub_dataset_id="lamm-mit/example",
    )
    calls = []

    class FakeMixture:
        def push_to_hub(self, *args, **kwargs):
            calls.append((args, kwargs))

    write_mixture_outputs(FakeMixture(), config)

    assert calls == [
        (
            ("lamm-mit/example",),
            {
                "config_name": "default",
                "private": False,
                "max_shard_size": "500MB",
                "num_proc": None,
            },
        )
    ]


def test_manifest_rejects_multiple_fill_sources(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {"dataset": "first", "fill": True},
                    {"dataset": "second", "fill": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at most one"):
        load_manifest(manifest)
