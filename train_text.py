from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "text"])
    main()
