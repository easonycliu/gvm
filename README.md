# Install
First install dependencies and specified cuda and linux.

```
./setup deps
./setup cuda
./setup cuda_custom
./setup linux
```

Once setup, use `grub-list.py` provided in `tools/grublist` to select next boot kenrel to the customized one and reboot.
After reboot, install nvidia kernel module.

```
./setup driver
```

Add nvcc path by putting this line `export PATH="/usr/local/cuda-12.9/bin:$PATH"` to the end of `~/.bashrc`
And relogin to your bash to make it takes effect.

Install the applications.

```
./setup llama.cpp
./setup llamafactory
./setup diffusion
./setup vllm
```

The `llamafactory`, `diffusion` and `vllm` will have its own `venv`, find it in `playground/<train|infer>/venv` and source it to activate.
