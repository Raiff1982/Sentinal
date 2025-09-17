# Development Guide

## Setup
- Clone the repo
- Create a virtual environment
- Install dependencies: `pip install -r requirements.txt`
- Install in editable mode: `pip install -e .`

## Adding Features
- Add new modules to `sentinal/`
- Write tests in `sentinal/tests/`
- Document changes in `docs/`

## CI/CD
- All pushes and PRs to `main` run tests and linting via GitHub Actions
- See `.github/workflows/ci.yml`

## Packaging
- Update `setup.py` for new dependencies or entry points
