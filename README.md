# GVM

GVM is a GPU sharing and scheduling research prototype for colocating a
latency-critical (LC) vLLM service with a best-effort (BE) diffusion or
LLaMA-Factory workload. This repository includes the modified NVIDIA kernel
module, CUDA interception layer, applications, baselines, experiment harness,
and plotting scripts.

The setup below assumes Ubuntu, an NVIDIA Ampere-or-newer GPU, Python 3.12,
and CUDA 12.x.

```bash
git clone https://github.com/easonycliu/gvm.git
cd gvm
git submodule update --init --recursive --progress
```

# Setup
For AE, please skip the whole setup section.

## Setup GVM

### GVM GPU kernel module

The modified module is based on NVIDIA open GPU kernel modules 575.57.08 and
must be paired with matching GSP firmware and user-space driver components.
Installing a kernel driver can make the machine unavailable if the versions do
not match; use a disposable research machine and read
[`gvm-gpu-kernel-modules/README.md`](gvm-gpu-kernel-modules/README.md) first.

```bash
cd gvm-gpu-kernel-modules/scripts
./download_pkgs.sh
sudo bash ./uninstall_nv_driver.sh
# Reboot here when replacing an existing NVIDIA kernel driver.
./install_cuda.sh
./install_nv_driver.sh
./compile_modules.sh
./deploy_modules.sh
```

After a reboot, `./deploy_modules.sh` may need to be run again. Verify that
`nvidia-smi` works and `/sys/kernel/debug/nvidia-uvm` exists.

### GVM CUDA driver interception layer

The interception layer requires GCC, NVCC, Make, Python, `patchelf`, and the
Python package `lief`.

```bash
cd ~/gvm/gvm-cuda-driver
python3 -m venv .build-venv
source .build-venv/bin/activate
python -m pip install lief
make
make install INSTALL=install
deactivate
```

If `libcuda.so` cannot be detected automatically, pass its real path to both
commands with `CUDA=/path/to/libcuda.so`. Applications use the interception
layer by prepending `gvm-cuda-driver/install` to `LD_LIBRARY_PATH`; the
experiment launchers do this automatically.

## Setup Apps

Create one virtual environment per application. Install a PyTorch build that
matches the machine before installing each package; these are the versions
exercised by this repository.

### Diffusers in `DiffusionVenv`

```bash
cd ~/gvm
python3.12 -m venv venv/DiffusionVenv
source venv/DiffusionVenv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install diffusers==0.34.0 transformers==4.55.4 \
  accelerate sentencepiece protobuf
deactivate
```

### LLaMA-Factory in `LFVenv`

```bash
cd ~/gvm
python3.12 -m venv venv/LFVenv
source venv/LFVenv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install llamafactory==0.9.4
deactivate
```

### vLLM in `VllmVenv`

```bash
cd ~/gvm
python3.12 -m venv venv/VllmVenv
source venv/VllmVenv/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers==4.53.2 vllm==0.10.1.1
deactivate
```

The standalone serving benchmark uses a separate environment:

```bash
python3.12 -m venv venv/BenchmarkVenv
source venv/BenchmarkVenv/bin/activate
python -m pip install --upgrade pip
python -m pip install aiohttp numpy tqdm transformers==4.55.4
deactivate
```

The standalone benchmark supports both text completions and video chat
completions. Video mode replays BurstGPT arrival timestamps with clips from the
local MMVU cache in `exp/data/mmvu_cache`; build or refresh that cache with
`exp/data/build_mmvu_cache.py`. The experiment launcher selects the mode and
endpoint automatically.

## Setup Baselines

### XSched

```bash
cd ~/gvm/3rdparty/xsched
make cuda
```

The harness loads `3rdparty/xsched/output/lib` and starts
`output/bin/xserver` automatically. Its vLLM process uses `VllmACVenv`, because
XSched relies on integration hooks in the adapted vLLM source and is not fully
transparent to an unmodified vLLM installation. Diffusion and LLaMA-Factory
continue to use their normal application environments. See
[`3rdparty/xsched/README.md`](3rdparty/xsched/README.md) for upstream details.

### TGS

TGS uses Docker in the experiment harness. Build its RPC stubs and high/low
priority interception libraries, then fetch the image named by the launchers.

```bash
cd ~/gvm/3rdparty/TGS
python3.12 -m venv ../../venv/TGSVenv
source ../../venv/TGSVenv/bin/activate
python -m pip install -r requirement.txt
make rpc
deactivate
cd hijack
./build.sh
docker pull easonliu12138/gvm_cuda_12_9
```

## Setup GVMAC

GVMAC uses the adapted source tree in `apps/vllm`. Its Python package is
editable and its CUDA extensions are compiled against the existing PyTorch
installation. The source requires Python `<3.13` and PyTorch 2.7.1.

```bash
cd ~/gvm
python3.12 -m venv venv/VllmACVenv
source venv/VllmACVenv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

cd apps/vllm
python use_existing_torch.py
python -m pip install -r requirements/build.txt
python -m pip install --force-reinstall nvidia-nvtx-cu12==12.8.55
# vLLM's runtime metadata only has a lower bound, but this revision is not
# compatible with the tokenizer API in Transformers 5.x. NumPy 2.5.x is also
# incompatible with the installed Numba and mistral-common versions.
python -m pip install numpy==2.2.6 transformers==4.53.2

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export NVTOOLSEXT_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nvtx"
export MAX_JOBS=6
python -m pip install --no-build-isolation --editable .
python -m pip check

# Install the notification extension into VllmACVenv. The adapted vLLM uses
# this module to receive memory.limit.high changes and resize its KV cache.
cd ~/gvm/gvm-notify
make install-python
python -c 'import gvm_notify; print("gvm-notify import: OK")'
deactivate
```

`NVTOOLSEXT_PATH` lets the adapted CMake configuration find NVTX3 from the pip
package; otherwise recent CUDA toolkits can fail with a missing
`CUDA::nvToolsExt` target. Reduce `MAX_JOBS` if compilation exhausts host RAM.

Verify the dependency versions in `VllmACVenv`:

```bash
source ~/gvm/venv/VllmACVenv/bin/activate
python -c 'import gvm_notify, numpy, tokenizers, transformers; print("gvm-notify: OK"); print("numpy:", numpy.__version__); print("transformers:", transformers.__version__); print("tokenizers:", tokenizers.__version__)'
```

The expected versions are NumPy 2.2.6, Transformers 4.53.2, and Tokenizers
0.21.4. Avoid force-reinstalling Transformers by itself: pip may upgrade NumPy
to 2.5.x, which conflicts with `numba==0.61.2` and
`mistral-common==1.11.7`. If that has already happened, repair the environment:

```bash
python -m pip install --force-reinstall numpy==2.2.6
python -m pip check
```

# Kick-The-Tire

```bash
cd ~/gvm/exp
./run_check.sh
```

The script checks the GVM kernel debugfs interface, imports all four application
environments, and allocates a 1 GiB CUDA tensor through the GVM interception
layer. It may request `sudo` access because debugfs is commonly hidden from
unprivileged users. A successful run ends approximately as follows:

```text
INFO  checking the GVM NVIDIA UVM debugfs interface with sudo
PASS  GVM NVIDIA UVM debugfs interface
PASS  Diffusers 0.34.0 (DiffusionVenv)
PASS  LLaMA-Factory 0.9.4 (LFVenv)
PASS  vLLM 0.10.1.1 (VllmVenv)
PASS  adapted vLLM ... from apps/vllm (VllmACVenv)
INFO  allocating a 1 GiB CUDA tensor through the GVM intercept layer
total cuda memory allocated: ...MB
allocated tensor: 1024 MiB
PASS  GVM CUDA allocation interception
All checks passed.
```

# Evaluation

Standalone baselines run only one side of the experiment. The LC-only run
requires a request dataset; the BE-only run does not:

```bash
cd exp/vllm+<diffusion|llama-factory>
../run_all.sh --method=exclusive-lc --duration=600 \
  --dataset=<path-to-dataset> --eval=overall
../run_all.sh --method=exclusive-be --duration=600 --eval=overall
```

The two runs produce independently timestamped files. Plotting scripts select
the latest LC-only and BE-only files separately and use them as the Exclusive
reference.

Run `run_all.sh` from one of the two experiment directories. Results are stored
under `data/<evaluation>/<experiment>/`; filenames include the method,
priorities where applicable, and a timestamp. `GVM` uses fixed limits and
priorities, `GVMFT` adds the transparent runtime scheduler, and `GVMAC` adds
application collaboration on top of GVMFT.

`<path-to-dataset>` is the BurstGPT CSV consumed by the standalone vLLM
benchmark.
For AE, the `<path-to-dataset>` will be under `~/BurstGPTDataset/burstgpt/BurstGPT_adjust.csv`

## Overall evaluation

```bash
cd ~/gvm/exp/vllm+<diffusion|llama-factory>
../run_all.sh --method=<GVM|GVMFT|GVMAC|TGS|xsched|GPreempt> \
  --duration=600 --dataset=<path-to-dataset> --mode=<text|video> --eval=overall
```

## No-memory-limit evaluation

```bash
cd ~/gvm/exp/vllm+<diffusion|llama-factory>
../run_all.sh --method=<GVM|GVMFT|GVMAC|TGS|xsched|GPreempt> \
  --duration=600 --dataset=<path-to-dataset> --eval=nomem
```

This evaluation requires a GPU with at least 80 GB of memory.

## Pareto evaluation

The reported Pareto experiments use vLLM plus diffusion.

```bash
cd ~/gvm/exp/vllm+diffusion

run_pareto() {
  ../run_all.sh --method=GVM --duration=600 \
    --dataset=<path-to-dataset> --eval=pareto "$@"
}

# Default LC memory limit (40 GB).
run_pareto --lcpriority=2 --bepriority=8
run_pareto --lcpriority=2 --bepriority=6
run_pareto --lcpriority=2 --bepriority=4
run_pareto --lcpriority=2 --bepriority=2
run_pareto --lcpriority=4 --bepriority=2
run_pareto --lcpriority=6 --bepriority=2

# LC memory limit: 28 GB.
run_pareto --lcpriority=2  --bepriority=10 --lcmemlimit=28000000000
run_pareto --lcpriority=2  --bepriority=8  --lcmemlimit=28000000000
run_pareto --lcpriority=2  --bepriority=6  --lcmemlimit=28000000000
run_pareto --lcpriority=2  --bepriority=4  --lcmemlimit=28000000000
run_pareto --lcpriority=2  --bepriority=2  --lcmemlimit=28000000000
run_pareto --lcpriority=4  --bepriority=2  --lcmemlimit=28000000000
run_pareto --lcpriority=6  --bepriority=2  --lcmemlimit=28000000000
run_pareto --lcpriority=8  --bepriority=2  --lcmemlimit=28000000000
run_pareto --lcpriority=10 --bepriority=2  --lcmemlimit=28000000000

# LC memory limit: 16 GB.
run_pareto --lcpriority=4  --bepriority=2 --lcmemlimit=16000000000
run_pareto --lcpriority=6  --bepriority=2 --lcmemlimit=16000000000
run_pareto --lcpriority=8  --bepriority=2 --lcmemlimit=16000000000
run_pareto --lcpriority=10 --bepriority=2 --lcmemlimit=16000000000
```
