"""The manual policy describes UI limits and remains separate from generation."""
from types import SimpleNamespace

import pytest

from src.rq3_manual.common import adapter_hashes, bundle_path, make_policy
from src.rq3_manual.prompts import RUBRIC_NAMES


def test_policy_records_ui_limits_and_complete_adapter_identity():
    evaluator = SimpleNamespace(prompt_sha256="official-file-hash", modules={
        "common": {name: "Pinned rubric " + name for name in RUBRIC_NAMES}})
    policy = make_policy("  Selected UI model  ", "temporary chat; tools off", evaluator)
    assert policy["interface"] == "chatgpt_manual"
    assert policy["judge_label"] == "Selected UI model"
    assert policy["temperature"] is None
    assert policy["model_snapshot"] is None
    assert policy["aggregate"] == "mean_applicable_normalized"
    assert set(policy["rubric_sha256"]) == set(RUBRIC_NAMES)
    assert policy["adapter_files_sha256"] == adapter_hashes()
    assert "src/rq3_manual/analysis.py" in policy["adapter_files_sha256"]
    assert "src/rq3/analyze.py" in policy["adapter_files_sha256"]


@pytest.mark.parametrize("relative", ["../escape.png", "/absolute.png", ".", ""])
def test_bundle_references_cannot_escape_or_target_directory(tmp_path, relative):
    with pytest.raises(ValueError, match="relative|escapes"):
        bundle_path(tmp_path, relative)


def test_bundle_symlink_cannot_escape(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        bundle_path(bundle, "link/outside.png")
