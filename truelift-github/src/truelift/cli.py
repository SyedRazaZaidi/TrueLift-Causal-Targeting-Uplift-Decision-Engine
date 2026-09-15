from __future__ import annotations

import argparse

from truelift.config import Settings
from truelift.pipeline import DEFAULT_MODELS, run_engine


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="truelift", description="TrueLift causal decisioning engine")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default=None, help="synthetic | hillstrom | ihdp | jobs | criteo")
    common.add_argument("--outcome", default=None)
    common.add_argument("--models", default=None, help="comma-separated model names")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", parents=[common], help="Download/build dataset and write prepared tables")
    sub.add_parser("train", parents=[common], help="Fit the full estimator zoo + policies + audit")
    sub.add_parser("evaluate", parents=[common], help="Alias of train (frozen eval is part of the run)")
    serve = sub.add_parser("serve", parents=[common], help="API + Next.js control plane")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--web-port", type=int, default=None, help="Next.js port (default 3000)")
    serve.add_argument("--api-only", action="store_true", help="FastAPI only; static fallback at /")
    all_p = sub.add_parser("all", parents=[common], help="prepare + train + (optionally) serve")
    all_p.add_argument("--serve", action="store_true")
    args = p.parse_args(argv)

    settings = Settings()
    if args.dataset:
        settings.dataset = args.dataset
    if args.outcome:
        settings.outcome = args.outcome
    models = DEFAULT_MODELS
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.cmd == "prepare":
        from truelift.data import load_dataset, save_prepared

        fr = load_dataset(settings)
        save_prepared(fr, settings.dataset)
        print(f"prepared {settings.dataset} n={fr.n} arms={fr.arm_names}")
        return
    if args.cmd in {"train", "evaluate"}:
        art = run_engine(settings, models)
        print(f"champion={art['champion']} auuc={next(r['auuc'] for r in art['leaderboard'] if r['model']==art['champion']):.4f} ship={art['card']['ship']}")
        return
    if args.cmd == "serve":
        _serve(
            settings,
            args.host,
            args.port,
            api_only=getattr(args, "api_only", False),
            web_port=getattr(args, "web_port", None),
        )
        return
    if args.cmd == "all":
        art = run_engine(settings, models)
        print(f"champion={art['champion']} auuc={next(r['auuc'] for r in art['leaderboard'] if r['model']==art['champion']):.4f} ship={art['card']['ship']}")
        if getattr(args, "serve", False):
            _serve(settings, None, None)


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent.parent


def _start_next(web_port: int) -> None:
    import os
    import shutil
    import subprocess
    import sys

    web = _repo_root() / "web"
    if not (web / "package.json").exists():
        print("web/ missing — skipping Next.js")
        return
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        print("npm not found — install Node.js or use --api-only")
        return
    if not (web / "node_modules").exists():
        print("Running npm install in web/ (first time)…")
        subprocess.check_call([npm, "install"], cwd=web, shell=sys.platform == "win32")
    env = os.environ.copy()
    env["PORT"] = str(web_port)
    subprocess.Popen(
        [npm, "run", "dev", "--", "-p", str(web_port)],
        cwd=web,
        env=env,
        shell=sys.platform == "win32",
    )
    print(f"Next.js UI: http://127.0.0.1:{web_port}")


def _serve(settings: Settings, host: str | None, port: int | None, *, api_only: bool = False, web_port: int | None = None) -> None:
    import uvicorn

    from truelift.serve.app import create_app

    api_port = port or settings.port
    wp = web_port or settings.web_port
    redirect = None
    if not api_only and settings.serve_web:
        _start_next(wp)
        redirect = settings.web_origin.replace("localhost", "127.0.0.1")
        if f":{wp}" not in redirect:
            redirect = f"http://127.0.0.1:{wp}"
    app = create_app(redirect_root=redirect)
    print(f"TrueLift API: http://{host or settings.host}:{api_port}")
    if redirect:
        print(f"Open the dashboard at {redirect}")
    uvicorn.run(app, host=host or settings.host, port=api_port, log_level="info")


if __name__ == "__main__":
    main()
