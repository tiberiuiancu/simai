# SimAI

Python wrapper for the [SimAI](https://github.com/aliyun/SimAI) datacenter network simulator. Provides a CLI and Python API for generating training workloads and running network simulations, with pre-built binaries bundled in the wheel.

## Installation

Download the latest wheel from [GitHub Releases](https://github.com/tiberiuiancu/simai/releases):

```bash
pip install https://github.com/tiberiuiancu/simai/releases/download/v0.1.0/simai-0.1.0-py3-none-manylinux_2_35_x86_64.whl
```

For GPU compute profiling (optional, requires CUDA):

```bash
pip install "simai[profiling]"
```

## Usage

### 1. Generate a workload

```bash
simai workflow generate \
    --framework Megatron \
    --num-gpus 64 \
    --tensor-parallel 4 \
    --pipeline-parallel 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --sequence-length 2048 \
    -o workload.txt
```

### 2. Run a simulation

**Analytical** (fast, approximate):

```bash
simai simulate analytical \
    -w workload.txt \
    -g 64 \
    --gpu-type H100 \
    -o results/
```

**NS-3** (detailed, packet-level):

```bash
simai simulate ns3 \
    -w workload.txt \
    -n topology/ \
    -o results/
```

## Building from source

Requires the SimAI submodule and compiled binaries:

```bash
git clone --recurse-submodules https://github.com/tiberiuiancu/simai.git
cd simai

# Build binaries (see vendor/simai/scripts/build.sh)
# Place SimAI_analytical and SimAI_simulator in build/bin/

SIMAI_PLATFORM_TAG=manylinux_2_35_x86_64 pip install build
python -m build --wheel
```

Or just push to GitHub — the CI workflow compiles the binaries and produces the wheel automatically.
