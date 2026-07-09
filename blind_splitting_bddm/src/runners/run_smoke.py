"""Run the reduced Colab smoke suite."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blind_splitting_bddm.src.runners import (
    make_report,
    run_closed_loop,
    run_dual_law_tests,
    run_posterior_correction_tests,
    run_prior_scale_tests,
)
from blind_splitting_bddm.src.utils import common_parser, prepare_config, seed_all


def _child_cfg(cfg: dict, name: str) -> dict:
    child = deepcopy(cfg)
    root = Path(cfg.get("save_dir", "blind_splitting_bddm/results/smoke"))
    child["save_dir"] = str(root / name)
    return child


def run(cfg: dict) -> None:
    seed_all(int(cfg.get("seed", 123)))
    print("running prior scale smoke")
    run_prior_scale_tests.run(_child_cfg(cfg, "prior_scale"))
    print("running posterior correction smoke")
    pcfg = _child_cfg(cfg, "posterior_correction")
    pcfg.setdefault("experiments", {})
    pcfg["experiments"]["d_values"] = [int(cfg.get("prior", {}).get("ambient_dim", 50))]
    pcfg["experiments"]["split_methods"] = ["gradient", "pdhg"]
    run_posterior_correction_tests.run(pcfg)
    print("running PDHG dual-law smoke")
    run_dual_law_tests.run(_child_cfg(cfg, "dual_law"))
    print("running closed-loop smoke")
    ccfg = _child_cfg(cfg, "closed_loop")
    ccfg.setdefault("experiments", {})
    ccfg["experiments"]["d_values"] = [int(cfg.get("prior", {}).get("ambient_dim", 50))]
    ccfg["experiments"]["split_methods"] = ["gradient"]
    run_closed_loop.run(ccfg)
    make_report.run(cfg)


def main() -> None:
    parser = common_parser("Run complete reduced smoke suite")
    args = parser.parse_args()
    run(prepare_config(args))


if __name__ == "__main__":
    main()
