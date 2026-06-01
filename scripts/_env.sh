#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x ".venv/bin/python" ]]; then
  rm -rf .venv
  echo ".venv not found; creating it with python3."
  python3 -m venv .venv || python3 -m venv --without-pip .venv
fi

VENV_PY=".venv/bin/python"
"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo "pip is unavailable in .venv; bootstrapping pip with get-pip.py."
  "$VENV_PY" - <<'PY'
import urllib.request
urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", ".venv/get-pip.py")
PY
  "$VENV_PY" .venv/get-pip.py
fi

export PATH="$PWD/.venv/bin:$PATH"
hash -r

if ! "$VENV_PY" - <<'PY'
import importlib
missing = []
for name in ["torch", "numpy", "pandas", "matplotlib", "yaml", "datasets", "tiktoken"]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages after activation: " + ", ".join(missing))
PY
then
  echo "Installing missing Python packages from requirements.txt."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
fi

"$VENV_PY" - <<'PY'
import importlib
missing = []
for name in ["torch", "numpy", "pandas", "matplotlib", "yaml", "datasets", "tiktoken"]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages after installation: " + ", ".join(missing))
PY
