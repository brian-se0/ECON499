"""Install the supported macOS LightGBM build for the Mac CPU profile."""

from __future__ import annotations

import platform
import subprocess
import sys


def main() -> None:
    if platform.system() != "Darwin":
        message = "The no-OpenMP LightGBM installer is only for the mac_cpu profile on macOS."
        raise RuntimeError(message)

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--force-reinstall",
        "--no-deps",
        "--no-binary",
        "lightgbm",
        "--config-settings=cmake.define.USE_OPENMP=OFF",
        "lightgbm==4.6.0",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
