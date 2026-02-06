from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def analytical(
    workload: Annotated[
        Path,
        typer.Option("--workload", "-w", help="Path to workload file (from workflow generate)."),
    ],
    num_gpus: Annotated[
        int,
        typer.Option("--num-gpus", "-g", help="Number of GPUs to simulate."),
    ],
    gpus_per_server: Annotated[
        int,
        typer.Option("--gpus-per-server", help="GPUs per server (NVLink domain size)."),
    ] = 8,
    nvlink_bandwidth: Annotated[
        Optional[float],
        typer.Option("--nvlink-bandwidth", help="NVLink bandwidth in GB/s."),
    ] = None,
    nic_bandwidth: Annotated[
        Optional[float],
        typer.Option("--nic-bandwidth", help="NIC bus bandwidth in GB/s."),
    ] = None,
    nics_per_server: Annotated[
        Optional[int],
        typer.Option("--nics-per-server", help="Number of NICs per server."),
    ] = None,
    busbw: Annotated[
        Optional[Path],
        typer.Option("--busbw", help="Bus bandwidth YAML file."),
    ] = None,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type: A100, H100, H800, etc."),
    ] = None,
    dp_overlap: Annotated[
        Optional[float],
        typer.Option("--dp-overlap", help="Data-parallel communication overlap ratio (0.0-1.0)."),
    ] = None,
    tp_overlap: Annotated[
        Optional[float],
        typer.Option("--tp-overlap", help="Tensor-parallel overlap ratio."),
    ] = None,
    ep_overlap: Annotated[
        Optional[float],
        typer.Option("--ep-overlap", help="Expert-parallel overlap ratio."),
    ] = None,
    pp_overlap: Annotated[
        Optional[float],
        typer.Option("--pp-overlap", help="Pipeline-parallel overlap ratio."),
    ] = None,
    result_prefix: Annotated[
        Optional[str],
        typer.Option("--result-prefix", help="Prefix for result file names."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for result CSV files (default: ./results/)."),
    ] = None,
):
    """Run the analytical (fast, approximate) network simulation."""
    from simai.backends.analytical import run_analytical

    run_analytical(
        workload=workload,
        num_gpus=num_gpus,
        gpus_per_server=gpus_per_server,
        nvlink_bandwidth=nvlink_bandwidth,
        nic_bandwidth=nic_bandwidth,
        nics_per_server=nics_per_server,
        busbw=busbw,
        gpu_type=gpu_type,
        dp_overlap=dp_overlap,
        tp_overlap=tp_overlap,
        ep_overlap=ep_overlap,
        pp_overlap=pp_overlap,
        result_prefix=result_prefix,
        output=output,
    )


@app.command()
def ns3(
    workload: Annotated[
        Path,
        typer.Option("--workload", "-w", help="Path to workload file."),
    ],
    topology: Annotated[
        Path,
        typer.Option("--topology", "-n", help="Network topology directory."),
    ],
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="SimAI config file path (default: bundled SimAI.conf)."),
    ] = None,
    threads: Annotated[
        int,
        typer.Option("--threads", "-t", help="Number of simulation threads."),
    ] = 8,
    send_latency: Annotated[
        Optional[int],
        typer.Option("--send-latency", help="Send latency in microseconds."),
    ] = None,
    nvls: Annotated[
        bool,
        typer.Option("--nvls/--no-nvls", help="Enable NVLink Switch."),
    ] = False,
    pxn: Annotated[
        bool,
        typer.Option("--pxn/--no-pxn", help="Enable PXN (PCIe cross-node)."),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for results."),
    ] = None,
):
    """Run the NS-3 (detailed, packet-level) network simulation."""
    from simai.backends.ns3 import run_ns3

    run_ns3(
        workload=workload,
        topology=topology,
        config=config,
        threads=threads,
        send_latency=send_latency,
        nvls=nvls,
        pxn=pxn,
        output=output,
    )
