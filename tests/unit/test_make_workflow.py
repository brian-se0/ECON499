from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_makefile_exposes_official_targets() -> None:
    makefile = (_repo_root() / "Makefile").read_text(encoding="utf-8")

    for target in (
        "help:",
        "sync:",
        "sync-dev:",
        "sync-tracking:",
        "clean:",
        "clean-ingest:",
        "clean-silver:",
        "clean-surfaces:",
        "clean-features:",
        "clean-hpo:",
        "clean-train:",
        "clean-stats:",
        "clean-hedging:",
        "clean-report:",
        "ingest:",
        "silver:",
        "surfaces:",
        "features:",
        "hpo:",
        "hpo-all:",
        "hpo-30:",
        "hpo-100:",
        "tune:",
        "tune-all:",
        "train:",
        "train-30:",
        "train-100:",
        "walkforward:",
        "stats:",
        "hedging:",
        "report:",
        "pipeline:",
        "pipeline-30:",
        "pipeline-100:",
        "check:",
        "check-runtime:",
        "official-smoke:",
    ):
        assert target in makefile


def test_profile_alias_targets_do_not_require_recursive_make() -> None:
    makefile = (_repo_root() / "Makefile").read_text(encoding="utf-8")

    assert (
        "hpo-all: hpo-ridge hpo-elasticnet hpo-har-factor "
        "hpo-lightgbm hpo-random-forest hpo-neural-surface"
    ) in makefile
    assert "hpo-30: HPO_PROFILE = hpo_30_trials" in makefile
    assert "hpo-30: TRAIN_PROFILE = train_30_epochs" in makefile
    assert "train-30: HPO_PROFILE = hpo_30_trials" in makefile
    assert "train-30: TRAIN_PROFILE = train_30_epochs" in makefile
    assert "pipeline-30: HPO_PROFILE = hpo_30_trials" in makefile
    assert "pipeline-30: TRAIN_PROFILE = train_30_epochs" in makefile


def test_readme_documents_make_as_official_interface() -> None:
    readme = (_repo_root() / "README.md").read_text(encoding="utf-8")

    assert "`make` is the official interface" in readme
    assert "make pipeline" in readme
    assert "make check-runtime" in readme
    assert "make official-smoke" in readme
    assert "make hpo-30" in readme
    assert "make train-100" in readme
    assert "make clean" in readme
    assert "make clean-train" in readme
    assert "never deletes or mutates the protected raw options directory" in readme
    assert "Direct script invocation is not the documented workflow" in readme
    assert "uv run python scripts/01_ingest_cboe.py" not in readme
