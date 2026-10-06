# Creates a clean, separate environment for the web app.  Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File webapp/setup_env.ps1
$ErrorActionPreference = "Stop"
$py = "py -3.11"
try { Invoke-Expression "$py --version" | Out-Null } catch { $py = "python" }
Invoke-Expression "$py --version"
if (Test-Path webapp/.venv) { Remove-Item -Recurse -Force webapp/.venv }
Invoke-Expression "$py -m venv webapp/.venv"
$pip = "webapp/.venv/Scripts/python.exe -m pip"
Invoke-Expression "$pip install --upgrade pip wheel `"setuptools<70`""
Invoke-Expression "$pip install --no-build-isolation `"bibtexparser<2`""
Invoke-Expression "$pip install -r webapp/backend/requirements.txt"
Write-Host "`nDone. Start the backend with:`n  webapp/.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir webapp/backend --port 8000"
