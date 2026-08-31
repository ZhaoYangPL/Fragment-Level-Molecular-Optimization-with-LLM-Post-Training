#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SELECTION_PATH = ROOT_DIR / "out" / "explicit_kl_pilot_selection.json"
ENTRY = ROOT_DIR / "train_grpo_delta_abs_explicit_kl.py"
ANCHOR = (
    "tinker://6adddb5a-0ab5-5a96-9178-b23dcbc0823d:train:0/weights/"
    "Layer-2D-11000-RL1e-6"
)
BETA_SUFFIXES = {
    0.0: "0",
    0.01: "001",
    0.03: "003",
    0.05: "005",
}


def main() -> int:
    if not SELECTION_PATH.exists():
        raise FileNotFoundError(
            f"Pilot selection is missing: {SELECTION_PATH}. Run "
            "select_explicit_kl_pilot.py first."
        )
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    beta = float(payload["selected"]["beta"])
    if beta not in BETA_SUFFIXES:
        raise ValueError(f"Unsupported selected beta: {beta}")
    suffix = BETA_SUFFIXES[beta]
    command = [
        sys.executable,
        str(ENTRY),
        "--checkpoint",
        ANCHOR,
        "--kl-reference-checkpoint",
        ANCHOR,
        "--num-iterations",
        "500",
        "--save-name",
        f"Layer-2D-11500-ExplicitKL-beta{suffix}",
        "--run-tag",
        f"selected11000_explicitkl_beta{suffix}_iters500",
        "--learning-rate",
        "1e-6",
        "--refresh-sampling-every",
        "4",
        "--rng-seed",
        "42",
        "--kl-penalty-coef",
        str(beta),
        "--no-token-prompt",
        "--no-proxy-probe",
    ]
    print("Launching selected 500-iteration run:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=ROOT_DIR.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
