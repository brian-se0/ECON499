SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python
MODEL ?=
HPO_PROFILE ?= hpo_30_trials
TRAIN_PROFILE ?= train_30_epochs

HPO_PROFILE_PATH := configs/workflow/$(HPO_PROFILE).yaml
TRAIN_PROFILE_PATH := configs/workflow/$(TRAIN_PROFILE).yaml

ifeq ($(strip $(LIMIT)),)
LIMIT_ARG :=
else
LIMIT_ARG := --limit $(LIMIT)
endif

.PHONY: help sync sync-dev sync-tracking lint test typecheck check check-runtime clean clean-ingest clean-silver clean-surfaces clean-features clean-hpo clean-train clean-stats clean-hedging clean-report ingest silver surfaces features hpo hpo-all hpo-30 hpo-100 tune tune-all train train-30 train-100 walkforward stats hedging report pipeline pipeline-30 pipeline-100 all

help:
	@Write-Host "Official ECON499 workflow targets:"
	@Write-Host "  make sync             - install runtime dependencies with uv"
	@Write-Host "  make sync-dev         - install runtime + dev dependencies"
	@Write-Host "  make sync-tracking    - install runtime + dev + MLflow tracking extras"
	@Write-Host "  make hpo-30           - mandatory Optuna HPO stage with the 30-trial profile"
	@Write-Host "  make hpo-100          - mandatory Optuna HPO stage with the 100-trial profile"
	@Write-Host "  make train-30         - walk-forward training/forecasting with the 30-epoch profile"
	@Write-Host "  make train-100        - walk-forward training/forecasting with the 100-epoch profile"
	@Write-Host "  make pipeline-30      - full official pipeline with 30-trial HPO + 30-epoch training"
	@Write-Host "  make pipeline-100     - full official pipeline with 100-trial HPO + 100-epoch training"
	@Write-Host "  make clean            - remove all derived artifacts across every pipeline stage"
	@Write-Host "  make clean-<stage>    - clean one stage and every downstream stage"
	@Write-Host "  make check            - run ruff, pytest, and mypy"
	@Write-Host "  make check-runtime    - verify the official Windows/GPU/CUDA runtime contract"
	@Write-Host "Optional variables:"
	@Write-Host "  LIMIT=<n>             - limit files for stages 01-03 during smoke runs"
	@Write-Host "  MODEL=<name>          - select the model for make hpo / make tune"
	@Write-Host "  HPO_PROFILE=<name>    - one of hpo_30_trials or hpo_100_trials"
	@Write-Host "  TRAIN_PROFILE=<name>  - one of train_30_epochs or train_100_epochs"
	@Write-Host "Clean stages:"
	@Write-Host "  ingest silver surfaces features hpo train stats hedging report"

sync:
	$(UV) sync

sync-dev:
	$(UV) sync --extra dev

sync-tracking:
	$(UV) sync --extra dev --extra tracking

lint:
	$(UV) run $(PYTHON) -m ruff check .

test:
	$(UV) run $(PYTHON) -m pytest

typecheck:
	$(UV) run $(PYTHON) -m mypy src tests

check: lint test typecheck

check-runtime:
	$(UV) run $(PYTHON) scripts/check_runtime.py

clean:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py all --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-ingest:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py ingest --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-silver:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py silver --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-surfaces:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py surfaces --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-features:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py features --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-hpo:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py hpo --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-train:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py train --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-stats:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py stats --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-hedging:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py hedging --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

clean-report:
	$(UV) run $(PYTHON) scripts/clean_pipeline_artifacts.py report --hpo-profile-name $(HPO_PROFILE) --training-profile-name $(TRAIN_PROFILE)

ingest:
	$(UV) run $(PYTHON) scripts/01_ingest_cboe.py $(LIMIT_ARG)

silver:
	$(UV) run $(PYTHON) scripts/02_build_option_panel.py $(LIMIT_ARG)

surfaces:
	$(UV) run $(PYTHON) scripts/03_build_surfaces.py $(LIMIT_ARG)

features:
	$(UV) run $(PYTHON) scripts/04_build_features.py

hpo:
	@if ("$(MODEL)" -eq "") { throw "MODEL is required. Example: make hpo MODEL=ridge" }
	$(UV) run $(PYTHON) scripts/05_tune_models.py "$(MODEL)" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-all:
	$(MAKE) hpo MODEL=ridge HPO_PROFILE=$(HPO_PROFILE) TRAIN_PROFILE=$(TRAIN_PROFILE)
	$(MAKE) hpo MODEL=elasticnet HPO_PROFILE=$(HPO_PROFILE) TRAIN_PROFILE=$(TRAIN_PROFILE)
	$(MAKE) hpo MODEL=har_factor HPO_PROFILE=$(HPO_PROFILE) TRAIN_PROFILE=$(TRAIN_PROFILE)
	$(MAKE) hpo MODEL=lightgbm HPO_PROFILE=$(HPO_PROFILE) TRAIN_PROFILE=$(TRAIN_PROFILE)
	$(MAKE) hpo MODEL=random_forest HPO_PROFILE=$(HPO_PROFILE) TRAIN_PROFILE=$(TRAIN_PROFILE)
	$(MAKE) hpo MODEL=neural_surface HPO_PROFILE=$(HPO_PROFILE) TRAIN_PROFILE=$(TRAIN_PROFILE)

hpo-30:
	$(MAKE) hpo-all HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs

hpo-100:
	$(MAKE) hpo-all HPO_PROFILE=hpo_100_trials TRAIN_PROFILE=train_100_epochs

tune: hpo

tune-all: hpo-all

train:
	$(UV) run $(PYTHON) scripts/06_run_walkforward.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

train-30:
	$(MAKE) train HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs

train-100:
	$(MAKE) train HPO_PROFILE=hpo_100_trials TRAIN_PROFILE=train_100_epochs

walkforward: train

stats:
	$(UV) run $(PYTHON) scripts/07_run_stats.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hedging:
	$(UV) run $(PYTHON) scripts/08_run_hedging_eval.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

report:
	$(UV) run $(PYTHON) scripts/09_make_report_artifacts.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

pipeline: check check-runtime ingest silver surfaces features hpo-all train stats hedging report

pipeline-30:
	$(MAKE) pipeline HPO_PROFILE=hpo_30_trials TRAIN_PROFILE=train_30_epochs

pipeline-100:
	$(MAKE) pipeline HPO_PROFILE=hpo_100_trials TRAIN_PROFILE=train_100_epochs

all: pipeline
