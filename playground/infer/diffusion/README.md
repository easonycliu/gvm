## Prerequisite

* diffusers
* apply access to `stable-diffusion-3.5-large` on huggingface

```bash
pip install diffusers
```

## Run

```bash
python inference_sd3.5_large.py --help
# example
python inference_sd3.5_large.py --batch_size=10
```

### Run with LD_PRELOAD

```bash
pushd <path to gvm-dev>/csrc/custom_cuda_lib/
make -j
popd

LD_PRELOAD=<path to gvm-dev>/csrc/custom_cuda_lib/libcustom_cuda.so python inference_sd3.5_large.py
