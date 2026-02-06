from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from simai.backends.binary import run_binary

BINARY_NAME = "SimAI_simulator"


def _find_default_config() -> Path:
    """Find the bundled default SimAI.conf."""
    # Check in vendored location
    vendored = Path(__file__).resolve().parent.parent / "_vendor" / "SimAI.conf"
    if vendored.is_file():
        return vendored

    # Check SIMAI_PATH
    env_path = os.environ.get("SIMAI_PATH")
    if env_path:
        candidate = Path(env_path) / "astra-sim-alibabacloud" / "inputs" / "config" / "SimAI.conf"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Cannot find default SimAI.conf. Provide --config or set SIMAI_PATH."
    )


def run_ns3(
    *,
    workload: Path,
    topology: Path,
    config: Path | None = None,
    threads: int = 8,
    send_latency: int | None = None,
    nvls: bool = False,
    pxn: bool = False,
    output: Path | None = None,
) -> Path:
    """Run the SimAI NS-3 simulator backend.

    The binary writes output relative to cwd, so we run it from a temp
    directory and then move results to the user's chosen output path.

    Returns the output directory path.
    """
    workload = workload.resolve()
    topology = topology.resolve()

    if config is None:
        config = _find_default_config()
    config = config.resolve()

    # Build command-line arguments
    args: list[str] = [
        "-w", str(workload),
        "-n", str(topology),
        "-c", str(config),
        "-t", str(threads),
    ]

    # Build environment variables for the binary
    env: dict[str, str] = {}
    if send_latency is not None:
        env["AS_SEND_LAT"] = str(send_latency)
    if nvls:
        env["AS_NVLS_ENABLE"] = "1"
    if pxn:
        env["AS_PXN_ENABLE"] = "1"

    # Determine output directory
    output_dir = Path(output) if output else Path("results")
    output_dir = output_dir.resolve()

    # Run from a temp directory to capture output
    with tempfile.TemporaryDirectory(prefix="simai_ns3_") as tmpdir:
        run_binary(BINARY_NAME, args, cwd=tmpdir, env=env)

        # Move all generated files to output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in Path(tmpdir).iterdir():
            dest = output_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        print(f"Results saved to: {output_dir}")

    return output_dir
