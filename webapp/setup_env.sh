#!/usr/bin/env bash
# Clean, separate environment for the web app (Linux/macOS). Run from the repository root.
set -euo pipefail
rm -rf webapp/.venv
python3 -m venv webapp/.venv
. webapp/.venv/bin/activate
pip install --upgrade pip wheel "setuptools<70"
pip install --no-build-isolation "bibtexparser<2"
pip install -r webapp/backend/requirements.txt
