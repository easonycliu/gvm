import torch
import argparse
from pathlib import Path
from typing import List
from diffusers import StableDiffusion3Pipeline
from datetime import datetime
import os
import json
import time
import signal

siguser1_timestamp = []

def siguser1_handler(signo, frame):
    global siguser1_timestamp
    siguser1_timestamp.append(time.time())

class SDConfig:

    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-3.5-medium",
        batch_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_dir: str = "outputs",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.output_dir = Path(output_dir)
        self.device = device
        self.torch_dtype = torch_dtype

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


def create_pipe(config: SDConfig) -> StableDiffusion3Pipeline:
    """Initialize and return the Stable Diffusion pipeline."""
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            config.model_path, torch_dtype=config.torch_dtype)
        pipe = pipe.to(config.device)
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to create pipeline: {str(e)}")


def generate_images(
    pipe: StableDiffusion3Pipeline,
    prompts: List[str],
    config: SDConfig,
) -> (List[torch.Tensor], List[int]):
    """Generate images from a list of prompts."""
    try:
        if len(prompts) < config.batch_size:
            prompts = prompts * (config.batch_size // len(prompts)
                                 ) + prompts[:config.batch_size % len(prompts)]
        prompts = prompts[:config.batch_size]
        timestep = [time.time()]
        def f(_1, _2, _3, args):
            timestep.append(time.time())
            return args
        pipe.scheduler.set_timesteps(num_inference_steps=config.num_inference_steps, device=pipe.device)
        for i in range(30):
            images = pipe(
                prompt=prompts,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                callback_on_step_end=f
            ).images
        return images, timestep
    except Exception as e:
        raise RuntimeError(f"Failed to generate images: {str(e)}")


def save_images(images: List[torch.Tensor],
                config: SDConfig,
                prefix: str = "image"):
    """Save generated images to the output directory."""
    for idx, image in enumerate(images):
        output_path = config.output_dir / f"{prefix}_{idx}.png"
        image.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Batch Inference")
    parser.add_argument("--model_path",
                        type=str,
                        default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--enable_memory_profiling", action="store_true")
    args = parser.parse_args()

    # Initialize configuration
    config = SDConfig(
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir,
        torch_dtype=torch.float16,
    )

    prompts = [
        "A photo of an astronaut riding a horse on mars",
        "A beautiful sunset over a mountain landscape",
        "A futuristic city with flying cars",
    ]
    results = {
        "steps": args.num_inference_steps
    }

    start_infer_time = datetime.now()
    timestep = None
    pipe = create_pipe(config)
    if args.enable_memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=100000)

        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            images, timestep = generate_images(pipe, prompts, config)
        prof.export_chrome_trace("sd_mem_profile.json")

        torch.cuda.memory._dump_snapshot("sd_mem_profile.pkl")
        torch.cuda.memory._record_memory_history(enabled=None)
    else:
        images, timestep = generate_images(pipe, prompts, config)
    end_infer_time = datetime.now()
    results["duration"] = (end_infer_time - start_infer_time).seconds
    results["iter_per_second"] = results["steps"] / results["duration"]
    results["step_timestamp"] = timestep
    results["interference_timestamp"] = siguser1_timestamp

    with open(os.path.join(args.output_dir, "{}.json".format(datetime.now().strftime("%m%d%Y-%H%M%S"))), "w") as f:
        json.dump(results, f, indent=4)

    if args.save_images:
        save_images(images, config)


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, siguser1_handler)
    main()
