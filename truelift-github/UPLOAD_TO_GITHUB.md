# Upload this folder to GitHub

This is a **clean publish copy** of TrueLift (no `.venv`, `node_modules`, trained weights, or local uploads).

## Option A — GitHub website

1. Create a new repository on GitHub (empty, no README).
2. In this folder, open PowerShell:

```powershell
cd D:\MyProjects\truelift-github
git init
git add .
git commit -m "Initial commit: TrueLift causal targeting engine"
git branch -M main
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

## Option B — GitHub CLI

```powershell
cd D:\MyProjects\truelift-github
gh repo create YOUR_REPO --public --source=. --remote=origin --push
```

## After clone (for you or recruiters)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
truelift all --dataset synthetic
truelift serve
```

Open http://127.0.0.1:3000 (Next.js). API: http://127.0.0.1:8000.
