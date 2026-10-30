#!/usr/bin/env python
"""Run every pipeline step in order, from Figure 1 to Figure 3, then export the figures.

Each step runs in its own process, exactly as if it were run by hand, and the run stops
at the first step that fails. On a server without a display, prefix the command with
``xvfb-run -a``: the brain renders need an X display.

Run      python pipeline/run_all.py                      # compute and plot every step, then export
         python pipeline/run_all.py --stage plot         # redraw every figure from source_data/
         python pipeline/run_all.py --from step3_connectivity
         python pipeline/run_all.py --only step1b_histology step1c_cortical_types
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent

# Run order. Step 1a must come first: every later step reads its gradient.
STEPS = (
    "step1a_mpc_gradient",      # Figure 1a: MPC gradient of the salience network
    "step1b_histology",         # Figure 1b: histological context of the gradient
    "step1c_cortical_types",    # Figure 1c: cortical types of the salience network
    "step2a_ieeg_mni",          # Figure 2a: MNI Open iEEG Atlas
    "step2b_ieeg_mica",         # Figure 2b: MICA iEEG spectra
    "step2c_ieeg_states",       # Figure 2c: wakefulness, N2 and N3
    "step3_connectivity",       # Figure 3: gradient-weighted connectivity projection
    "step3_schematic",          # Figure 3 (panel A): schematic of the projection
)
EXPORT = "step4_export_figures"  # composites -> PNG / PDF


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline steps in order (Figure 1 to Figure 3).")
    parser.add_argument("--stage", choices=("compute", "plot", "all"), default="all",
                        help="Stage of every step (default: all)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", choices=STEPS, help="First step to run; the later ones follow")
    group.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps (in pipeline order)")
    parser.add_argument("--no-export", action="store_true", help="Skip step 4 (Inkscape export)")
    args = parser.parse_args()

    steps = list(STEPS[STEPS.index(args.start):] if args.start else STEPS)
    if args.only:
        steps = [s for s in STEPS if s in args.only]
    commands = [[sys.executable, str(PIPELINE / f"{step}.py"), "--stage", args.stage] for step in steps]
    if not args.no_export and args.stage != "compute":
        commands.append([sys.executable, str(PIPELINE / f"{EXPORT}.py")])

    for i, cmd in enumerate(commands, 1):
        name = Path(cmd[1]).stem
        print(f"[{i}/{len(commands)}] {name} {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"{name} failed; the steps after it were not run.")
        print(f"[{i}/{len(commands)}] {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
