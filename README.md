# Practicum Environment Guide

This workspace is WSL-first.

Do not run the active modeling pipelines with Windows Python or a Windows-native
virtual environment. The committed `.venv_modeling/pyvenv.cfg` points at
`/usr/bin/python3`, and the active modeling workflow assumes a Linux/WSL Python
runtime.

## Single Source of Truth

- Environment contract and PowerShell handoff: this file.
- Modeling workflow, commands, and artifact map:
  `project/modeling/README.md`.
- Modeling dependency manifest:
  `project/modeling/requirements-modeling.txt`.

## Prerequisites

- WSL2 with the `Ubuntu` distro available.
- Linux `python3` inside WSL.
- Repo opened from the Windows working tree at:
  `C:\Users\raede\Desktop\essay help master\Practicum`.

## One-Time WSL Setup

Run these commands inside WSL from the repo root:

```bash
cd "/mnt/c/Users/raede/Desktop/essay help master/Practicum"
python3 -m venv --system-site-packages .venv_modeling
source .venv_modeling/bin/activate
python -m pip install --upgrade pip
python -m pip install -r project/modeling/requirements-modeling.txt
```

If `.venv_modeling` already exists, reactivate it instead of recreating it:

```bash
cd "/mnt/c/Users/raede/Desktop/essay help master/Practicum"
source .venv_modeling/bin/activate
python -m pip install -r project/modeling/requirements-modeling.txt
```

## Run From PowerShell

Use the wrapper in the repo root. It enters WSL, switches to the repo, activates
`.venv_modeling`, and then runs the command you provide.

```powershell
.\run_wsl_modeling.ps1 -Command "python project/modeling/run_pipeline.py full-run"
.\run_wsl_modeling.ps1 -Command "python project/modeling/pipelines/01_in_sample_pipeline.py strict-v2"
.\run_wsl_modeling.ps1 -Command "python project/modeling/pipelines/02_cross_event_pipeline.py stabilize-v3"
.\run_wsl_modeling.ps1 -Command "python project/modeling/pipelines/03_exploration_pipeline.py bug2-pr-pilot-v1"
```

Optional: override the distro if your WSL distro is not named `Ubuntu`.

```powershell
.\run_wsl_modeling.ps1 -Distro Ubuntu -Command "python project/modeling/run_pipeline.py full-run"
```

## Run Directly Inside WSL

```bash
cd "/mnt/c/Users/raede/Desktop/essay help master/Practicum"
source .venv_modeling/bin/activate
python project/modeling/run_pipeline.py full-run
```

## Scope

The WSL requirement applies to the active modeling code under `project/modeling/`.
Large data downloads, report artifacts, and cached outputs still live in the
same repo tree; this change only makes the execution contract explicit.
