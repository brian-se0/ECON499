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

.PHONY: help sync sync-dev sync-tracking lint test typecheck check check-runtime official-smoke clean clean-ingest clean-silver clean-surfaces clean-features clean-hpo clean-train clean-stats clean-hedging clean-report ingest silver surfaces features hpo hpo-all hpo-ridge hpo-elasticnet hpo-har-factor hpo-lightgbm hpo-random-forest hpo-neural-surface hpo-30 hpo-100 tune tune-all train train-30 train-100 walkforward stats hedging report pipeline pipeline-30 pipeline-100 all

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
	@Write-Host "  make official-smoke   - run the official stage01-stage09 Windows/GPU smoke bundle"
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

official-smoke:
	$(UV) run $(PYTHON) scripts/official_smoke.py

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

hpo-ridge:
	$(UV) run $(PYTHON) scripts/05_tune_models.py "ridge" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-elasticnet:
	$(UV) run $(PYTHON) scripts/05_tune_models.py "elasticnet" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-har-factor:
	$(UV) run $(PYTHON) scripts/05_tune_models.py "har_factor" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-lightgbm:
	$(UV) run $(PYTHON) scripts/05_tune_models.py "lightgbm" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-random-forest:
	$(UV) run $(PYTHON) scripts/05_tune_models.py "random_forest" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-neural-surface:
	$(UV) run $(PYTHON) scripts/05_tune_models.py "neural_surface" --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hpo-all: hpo-ridge hpo-elasticnet hpo-har-factor hpo-lightgbm hpo-random-forest hpo-neural-surface

hpo-30: HPO_PROFILE = hpo_30_trials
hpo-30: TRAIN_PROFILE = train_30_epochs
hpo-30: hpo-all

hpo-100: HPO_PROFILE = hpo_100_trials
hpo-100: TRAIN_PROFILE = train_100_epochs
hpo-100: hpo-all

tune: hpo

tune-all: hpo-all

train:
	$(UV) run $(PYTHON) scripts/06_run_walkforward.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

train-30: HPO_PROFILE = hpo_30_trials
train-30: TRAIN_PROFILE = train_30_epochs
train-30: train

train-100: HPO_PROFILE = hpo_100_trials
train-100: TRAIN_PROFILE = train_100_epochs
train-100: train

walkforward: train

stats:
	$(UV) run $(PYTHON) scripts/07_run_stats.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

hedging:
	$(UV) run $(PYTHON) scripts/08_run_hedging_eval.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

report:
	$(UV) run $(PYTHON) scripts/09_make_report_artifacts.py --hpo-profile-config-path $(HPO_PROFILE_PATH) --training-profile-config-path $(TRAIN_PROFILE_PATH)

pipeline: check check-runtime ingest silver surfaces features hpo-all train stats hedging report

pipeline-30: HPO_PROFILE = hpo_30_trials
pipeline-30: TRAIN_PROFILE = train_30_epochs
pipeline-30: pipeline

pipeline-100: HPO_PROFILE = hpo_100_trials
pipeline-100: TRAIN_PROFILE = train_100_epochs
pipeline-100: pipeline

all: pipeline
