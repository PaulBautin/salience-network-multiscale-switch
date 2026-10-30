"""Shared by the pipeline steps: the command line, settings and logging.

Every step is run the same way:

    python pipeline/<step>.py [--stage compute|plot|all]

``compute`` reads its inputs, writes any private tables to ``<work_dir>/<step>/`` and
the de-identified figure tables, with a ``manifest.json`` recording how they were made, to
``source_data/<step>/``. ``plot`` draws the figure panels from ``source_data/`` only, so
it runs without the private data. ``all`` (the default) does both. Steps whose inputs
are all public (``private=False``) can also compute without ``config/paths.yaml``.

The log of each run is appended to ``<work_dir>/<step>/<step>.log`` (it can contain
subject IDs, so it stays private); without ``config/paths.yaml`` it goes to the console.
"""
import argparse
import logging
import os
import re

import matplotlib

from salience_switch import config, provenance

STAGES = ("compute", "plot", "all")
# The creation date that VTK's SVG exporter (GL2PS) writes into the header of brain renders.
GL2PS_DATE = re.compile(rb"\nCreationDate: [^\n]*")


def strip_render_dates() -> None:
    """Remove the GL2PS creation date from the brain renders in ``results/figures/``."""
    for svg in config.FIGURES_DIR.glob("*.svg"):
        with open(svg, "rb") as fh:
            head = fh.read(1024)
        if b"GL2PS" in head and GL2PS_DATE.search(head):
            svg.write_bytes(GL2PS_DATE.sub(b"", svg.read_bytes(), count=1))


def run(step: str, compute, plot, stage: str = "all", private: bool = True) -> None:
    """Run one step's stages with its settings from ``config/analysis.yaml``.

    Parameters
    ----------
    step : str
        Step name, e.g. ``'step1a_mpc_gradient'``.
    compute : callable
        ``compute(settings, paths)``: inputs -> private and public tables.
    plot : callable
        ``plot(settings)``: public tables -> figure panels.
    stage : {'compute', 'plot', 'all'}
    private : bool
        Whether the compute stage reads private data (and so needs ``config/paths.yaml``).
    """
    # Byte-identical SVG panels on every run: a fixed element-id salt and a fixed date
    # (matplotlib honours SOURCE_DATE_EPOCH; GL2PS dates are stripped after the stages), so
    # a redraw changes only panels whose data changed.
    matplotlib.rcParams["svg.hashsalt"] = "salience-switch"
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    settings = config.step_settings(step)
    try:
        paths = config.load_paths()
    except FileNotFoundError:
        if private and stage != "plot":
            raise
        paths = None
    if paths is not None:
        provenance.setup_step_logger(step, config.work_dir(step, paths), {"stage": stage, **settings})
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if stage in ("compute", "all"):
        compute(settings, paths)
    if stage in ("plot", "all"):
        plot(settings)
    strip_render_dates()


def main(step: str, compute, plot, description: str, private: bool = True) -> None:
    """Command line of a step: ``--stage compute|plot|all``.

    Parameters
    ----------
    step : str
        Step name.
    compute, plot : callable
        See `run`.
    description : str
        One-line description shown by ``--help``.
    private : bool
        See `run`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--stage", choices=STAGES, default="all",
                        help="compute: inputs -> tables; plot: tables -> figures; all: both (default)")
    run(step, compute, plot, parser.parse_args().stage, private=private)
