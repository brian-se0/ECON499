# Windows equivalent of `make pipeline` followed by `make stats-sensitivity` and
# `make postmortem`, for machines without GNU Make. It runs the same stage scripts with
# the same arguments and order as the Makefile, using an existing project environment.
# Each stage runs as a separate process with stdout/stderr captured to UTF-8 logs; a
# stage fails only on a nonzero exit code (Python libraries log routinely to stderr).
param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$HpoProfile = "hpo_30_trials",
    [string]$TrainProfile = "train_30_epochs",
    [string]$LogDir = "data/run_logs",
    [string]$StartAt = "",
    [switch]$SkipChecks,
    [switch]$SkipPostHoc
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot
New-Item -ItemType Directory -Force $LogDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
# Repositories on exFAT drives fail git's ownership check; trust this checkout for the
# child processes only so run manifests still record the commit hash.
$env:GIT_CONFIG_COUNT = "1"
$env:GIT_CONFIG_KEY_0 = "safe.directory"
$env:GIT_CONFIG_VALUE_0 = ($RepoRoot -replace "\\", "/")

$hpoPath = "configs/workflow/$HpoProfile.yaml"
$trainPath = "configs/workflow/$TrainProfile.yaml"
$profileArgs = @("--hpo-profile-config-path", $hpoPath, "--training-profile-config-path", $trainPath)
$script:started = ($StartAt -eq "")

function Quote([string]$Value) {
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

function Invoke-Stage([string]$Name, [string[]]$Arguments) {
    if (-not $script:started) {
        if ($Name -ne $StartAt) {
            Write-Host ("SKIP  {0}" -f $Name)
            return
        }
        $script:started = $true
    }
    $out = Join-Path $LogDir "$Name.log"
    $err = Join-Path $LogDir "$Name.stderr.log"
    $began = Get-Date
    Write-Host ("[{0:yyyy-MM-dd HH:mm:ss}] START {1}" -f $began, $Name)
    $argumentLine = ($Arguments | ForEach-Object { Quote $_ }) -join " "
    $process = Start-Process -FilePath $Python -ArgumentList $argumentLine -NoNewWindow -Wait `
        -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $minutes = ((Get-Date) - $began).TotalMinutes
    if ($process.ExitCode -ne 0) {
        Write-Host ("[{0:yyyy-MM-dd HH:mm:ss}] FAIL  {1} (exit {2}) after {3:N1} min; see {4}" -f (Get-Date), $Name, $process.ExitCode, $minutes, $err)
        exit $process.ExitCode
    }
    Write-Host ("[{0:yyyy-MM-dd HH:mm:ss}] DONE  {1} in {2:N1} min" -f (Get-Date), $Name, $minutes)
}

if (-not $SkipChecks) {
    Invoke-Stage "check_ruff" @("-m", "ruff", "check", ".")
    Invoke-Stage "check_pytest" @("-m", "pytest")
    Invoke-Stage "check_mypy" @("-m", "mypy", "src", "tests", "scripts")
    Invoke-Stage "check_runtime" @("scripts/check_runtime.py")
}
Invoke-Stage "01_ingest" @("scripts/01_ingest_cboe.py")
Invoke-Stage "02_silver" @("scripts/02_build_option_panel.py")
Invoke-Stage "03_surfaces" @("scripts/03_build_surfaces.py")
Invoke-Stage "04_features" @("scripts/04_build_features.py")
foreach ($model in @("ridge", "elasticnet", "har_factor", "lightgbm", "random_forest", "neural_surface")) {
    Invoke-Stage "05_hpo_$model" (@("scripts/05_tune_models.py", $model) + $profileArgs)
}
Invoke-Stage "06_walkforward" (@("scripts/06_run_walkforward.py") + $profileArgs)
Invoke-Stage "07_stats" (@("scripts/07_run_stats.py") + $profileArgs)
Invoke-Stage "08_hedging" (@("scripts/08_run_hedging_eval.py") + $profileArgs)
Invoke-Stage "09_report" (@("scripts/09_make_report_artifacts.py") + $profileArgs)
if (-not $SkipPostHoc) {
    Invoke-Stage "10_stats_sensitivity" (@("scripts/10_run_stats_sensitivity.py") + $profileArgs)
    Invoke-Stage "11_postmortem" (@("scripts/11_neural_postmortem.py") + $profileArgs)
}
Write-Host ("[{0:yyyy-MM-dd HH:mm:ss}] PIPELINE COMPLETE" -f (Get-Date))
