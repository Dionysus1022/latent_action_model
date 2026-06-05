from __future__ import annotations

import sys
from collections.abc import Sequence

from planners import build_single_peak_dataset


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        build_single_peak_dataset.main()
        return
    old_argv = sys.argv
    try:
        sys.argv = [str(old_argv[0]), *[str(arg) for arg in argv]]
        build_single_peak_dataset.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
