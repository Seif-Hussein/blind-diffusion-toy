"""Aggregate runner reports into one Markdown file."""

from __future__ import annotations

from pathlib import Path

from blind_splitting_bddm.src.utils import common_parser, prepare_config, write_markdown


def run(cfg: dict) -> Path:
    save_dir = Path(cfg.get("save_dir", "blind_splitting_bddm/results/smoke"))
    reports = sorted(save_dir.rglob("*report*.md")) + sorted(save_dir.rglob("*summary*.md"))
    lines = ["# Posterior-Guided BDDM Splitting Report", ""]
    lines.append("This report separates scale mismatch, optimization-induced bias, anisotropy, dual memory, and posterior correction error.")
    lines.append("")
    for report in reports:
        if report.name == "combined_report.md":
            continue
        lines.append(f"## {report.relative_to(save_dir)}")
        lines.append("")
        lines.append(report.read_text(encoding="utf-8").strip())
        lines.append("")
    out = save_dir / "combined_report.md"
    write_markdown(out, lines)
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = common_parser("Aggregate reports")
    args = parser.parse_args()
    run(prepare_config(args))


if __name__ == "__main__":
    main()
