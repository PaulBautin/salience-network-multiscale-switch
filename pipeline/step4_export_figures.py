#!/usr/bin/env python
"""Step 4: export the composite figures (results/*.svg) to PNG, and optionally PDF, with Inkscape.

The composites link their panels (xlink:href to results/figures/*.svg) rather than
embedding them. Viewers that draw an SVG as an image (the VS Code preview, a browser
<img>) never load linked files, so they show the composites with empty frames; the
exports are self-contained. Inkscape draws each linked panel as a bitmap (at the image's
inkscape:svg-dpi of the panel's own size), so the export raises that resolution, in a
temporary copy, until every panel reaches --dpi at its placed size; a PDF export keeps
only the composite's own text and shapes as vectors. A composite is exported again only
when it, or a file it links, is newer than its export; with --watch the step keeps doing
so, so the exports follow the plot stages and Inkscape edits.

Run      python pipeline/step4_export_figures.py                      # PNG of every out-of-date composite
         python pipeline/step4_export_figures.py --formats png pdf --watch

Requires Inkscape >= 1.0 on the PATH (it resolves the linked panels).
"""
import argparse
import logging
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote

from salience_switch import config

logger = logging.getLogger("step4_export_figures")

# href of an <image>/<use> naming a file, not an in-document '#id' or an embedded 'data:' URI.
_HREF = re.compile(r'(?:xlink:)?href="(?!#|data:)([^"]+)"')
# Inkscape draws a linked SVG <image> as a bitmap rendered at the image's
# inkscape:svg-dpi (96 by default) of the linked file's own size.
_IMAGE = re.compile(r'<image\b[^>]*>')
_UNIT_IN = {"in": 1.0, "mm": 1 / 25.4, "cm": 1 / 2.54, "pt": 1 / 72, "pc": 1 / 6, "px": 1 / 96, "": 1 / 96}


def _length_in(value: str | None) -> float | None:
    """An SVG length ('210mm', '177.8pt', '400') in inches, or None if unparsable."""
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*([a-z]*)\s*", value or "")
    return float(m.group(1)) * _UNIT_IN[m.group(2)] if m and m.group(2) in _UNIT_IN else None


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf'(?<![\w:-]){re.escape(name)}="([^"]*)"', tag)
    return m.group(1) if m else None


def get_parser() -> argparse.ArgumentParser:
    """Command line of the export step."""
    parser = argparse.ArgumentParser(
        description="Step 4: export the composite figures in results/ to PNG (and PDF) with Inkscape.")
    parser.add_argument("--figures", nargs="+", type=Path, default=None,
                        help="Composite SVGs to export (default: every results/*.svg)")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png"],
                        help="Export formats, written next to each SVG (default: png)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PNG resolution, and the least resolution of the linked panels (default: 300)")
    parser.add_argument("--force", action="store_true", help="Export even if the exports are up to date")
    parser.add_argument("--watch", action="store_true",
                        help="Keep running and export again whenever a composite or a linked panel changes")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Polling interval (s) in --watch mode; inputs must be unchanged this long "
                             "before an export, so a plot stage finishes writing first (default: 2)")
    return parser


def linked_files(svg: Path) -> tuple[list[Path], list[str]]:
    """Files an SVG links directly (resolved against its folder), and the links that point nowhere."""
    found, missing = [], []
    for href in dict.fromkeys(_HREF.findall(svg.read_text(errors="ignore"))):
        href = unquote(href.removeprefix("file://"))
        if "://" in href:  # remote resource: nothing to track
            continue
        path = svg.parent / href
        if path.is_file():
            found.append(path)
        else:
            missing.append(href)
    return found, missing


def with_panel_dpi(text: str, folder: Path, dpi: int) -> str:
    """Composite SVG text whose linked SVG panels render at ``dpi`` or more as placed.

    Each ``<image>`` linking an SVG gets ``inkscape:svg-dpi`` = ``dpi`` x (placed width /
    the linked file's own width), never lower than it already has, so a panel drawn small
    and enlarged in the layout is not rendered coarser than the export. The placed width
    is the image's ``width`` in document units (ancestor group scaling is not followed).
    """
    root = re.search(r"<svg\b[^>]*>", text).group(0)
    view_box = (_attr(root, "viewBox") or "").split()
    page_in = _length_in(_attr(root, "width"))
    unit_in = page_in / float(view_box[2]) if page_in and len(view_box) == 4 else 1 / 96

    def _raise(m: re.Match) -> str:
        tag = m.group(0)
        href = unquote((_attr(tag, "xlink:href") or _attr(tag, "href") or "").removeprefix("file://"))
        width = _attr(tag, "width") or ""
        # A unitless width is in document units; one with a unit is absolute.
        placed = float(width) * unit_in if re.fullmatch(r"\s*[0-9.eE+-]+\s*", width) else _length_in(width)
        linked = folder / href
        if not href.endswith(".svg") or placed is None or not linked.is_file():
            return tag
        head = linked.read_text(errors="ignore")[:4096]
        own = _length_in(_attr(re.search(r"<svg\b[^>]*>", head).group(0), "width")) if "<svg" in head else None
        if not own:
            return tag
        current = float(_attr(tag, "inkscape:svg-dpi") or 96)
        target = max(current, math.ceil(dpi * placed / own))
        if "inkscape:svg-dpi=" in tag:
            return re.sub(r'inkscape:svg-dpi="[^"]*"', f'inkscape:svg-dpi="{target:g}"', tag)
        return tag.replace("<image", f'<image inkscape:svg-dpi="{target:g}"', 1)

    return _IMAGE.sub(_raise, text)


def export(inkscape: str, svg: Path, fmt: str, dpi: int) -> Path:
    """Render ``svg`` to ``svg`` with the ``fmt`` suffix (PNG on an opaque white background).

    Inkscape renders from a hidden temporary copy, next to the composite so its links
    resolve, whose linked panels render at ``dpi`` or more (``with_panel_dpi``).
    """
    out = svg.with_suffix(f".{fmt}")
    tmp = svg.with_name(f".{svg.stem}.export.svg")
    tmp.write_text(with_panel_dpi(svg.read_text(errors="ignore"), svg.parent, dpi))
    cmd = [inkscape, str(tmp), f"--export-type={fmt}", f"--export-filename={out}"]
    if fmt == "png":
        cmd += [f"--export-dpi={dpi}", "--export-background=white", "--export-background-opacity=1"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        tmp.unlink(missing_ok=True)
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError(f"Inkscape failed on {svg.name} -> {fmt}:\n{result.stderr.strip()[-2000:]}")
    return out


def main():
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise SystemExit("Inkscape was not found on the PATH; it is needed to render the linked panels.")
    logger.info(f"Inkscape       : {subprocess.run([inkscape, '--version'], capture_output=True, text=True).stdout.strip()}")
    logger.info(f"Formats        : {', '.join(args.formats)} (PNG at {args.dpi} dpi, white background)")

    links = {}  # composite -> (its mtime when parsed, linked files)
    failed = {}  # (composite, format) -> newest input mtime of the failed export, not retried until it changes
    force = args.force
    while True:
        figures = args.figures or sorted(p for p in (config.PROJECT_ROOT / "results").glob("*.svg")
                                         if not p.name.startswith("."))  # skip export temporaries
        for svg in figures:
            if not svg.is_file():
                logger.warning(f"{svg} does not exist; skipped.")
                continue
            mtime = svg.stat().st_mtime
            if links.get(svg, (None,))[0] != mtime:
                found, missing = linked_files(svg)
                for href in missing:
                    logger.warning(f"{svg.name}: linked file '{href}' not found; it renders as an empty frame.")
                links[svg] = (mtime, found)
            newest = max([mtime] + [p.stat().st_mtime for p in links[svg][1] if p.is_file()])
            if args.watch and time.time() - newest < args.interval:
                continue  # inputs still being written; export on a later poll
            for fmt in args.formats:
                out = svg.with_suffix(f".{fmt}")
                if failed.get((svg, fmt)) == newest:
                    continue
                if force or not out.is_file() or out.stat().st_mtime < newest:
                    t0 = time.time()
                    try:
                        export(inkscape, svg, fmt, args.dpi)
                    except RuntimeError as err:
                        logger.error(str(err))
                        failed[(svg, fmt)] = newest
                        continue
                    logger.info(f"{svg.name} -> {out.name} ({time.time() - t0:.1f} s)")
        force = False
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
