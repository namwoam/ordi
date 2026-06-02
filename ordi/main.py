"""
ORDI CLI entry point.

Usage:
  python -m ordi.main run [E1|E2|...|E8|COTS|all]
  python -m ordi.main plot [E1|E2|...|E8|all]
  python -m ordi.main run all && python -m ordi.main plot all
"""

import sys
from ordi.eval.experiments import ALL_EXPERIMENTS, run_all
from ordi.eval.plots import plot_all

PLOT_FNS = {
    "E1": "plot_E1", "E2": "plot_E2", "E3": "plot_E3", "E4": "plot_E4",
    "E5": "plot_E5", "E6": "plot_E6", "E7": "plot_E7", "E8": "plot_E8",
}


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    cmd, target = args[0], args[1].upper()

    if cmd == "run":
        if target == "ALL":
            run_all()
        elif target in ALL_EXPERIMENTS:
            ALL_EXPERIMENTS[target]()
        else:
            print(f"Unknown experiment: {target}. Choose from {list(ALL_EXPERIMENTS)}")
            sys.exit(1)

    elif cmd == "plot":
        import ordi.eval.plots as _plots
        if target == "ALL":
            plot_all()
        elif target in PLOT_FNS:
            getattr(_plots, PLOT_FNS[target])()
        else:
            print(f"Unknown plot: {target}.")
            sys.exit(1)

    else:
        print(f"Unknown command: {cmd}. Use 'run' or 'plot'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
