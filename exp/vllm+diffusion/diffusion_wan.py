import argparse
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download

LOG_LEVEL = "INFO"
DISABLE_PROGRESS_BAR = True


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        filename = record.filename
        lineno = record.lineno
        timestamp = self.formatTime(record, "%m-%d %H:%M:%S")
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_level = f"{self.COLORS[level_name]}{level_name}{self.RESET}"
        else:
            colored_level = level_name
        log_message = (
            f"{colored_level} {timestamp} [{filename}:{lineno}] {record.getMessage()}"
        )
        return log_message


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    log_level = os.getenv("GVM_DIFFUSION_LOG_LEVEL", LOG_LEVEL)
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger()

default_config = {
    # Base Wan2.1 pipeline (full diffusers layout). The Turbo checkpoint
    # below is loaded into this pipeline's transformer.
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    # TurboDiffusion distilled transformer weights. Set turbo_repo to None
    # to run the unmodified base Wan2.1 pipeline.
    "turbo_repo": "TurboDiffusion/TurboWan2.1-T2V-1.3B-480P",
    "turbo_ckpt": "TurboWan2.1-T2V-1.3B-480P.pth",
    "batch_size": 1,
    # Turbo/distilled models converge in very few steps with CFG=1.
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    # 480P video defaults; height/width follow the model name.
    "num_frames": 81,
    "height": 480,
    "width": 832,
    "fps": 16,
    "output_dir": "diffusion_outputs",
}


class WanConfig:
    def __init__(
        self,
        model_path: str = default_config["model_path"],
        turbo_repo: str = default_config["turbo_repo"],
        turbo_ckpt: str = default_config["turbo_ckpt"],
        batch_size: int = default_config["batch_size"],
        num_inference_steps: int = default_config["num_inference_steps"],
        guidance_scale: float = default_config["guidance_scale"],
        num_frames: int = default_config["num_frames"],
        height: int = default_config["height"],
        width: int = default_config["width"],
        fps: int = default_config["fps"],
        output_dir: str = default_config["output_dir"],
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.turbo_repo = turbo_repo
        self.turbo_ckpt = turbo_ckpt
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.device = device
        self.torch_dtype = torch_dtype

        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    arrival_time: float


@dataclass
class InferenceResult:
    request_id: str
    prompt: str
    arrival_time: float
    start_time: float
    end_time: float
    inference_duration: float
    queue_wait_time: float
    batch_size: int


class DiffusionInferenceServer:
    """Offline Wan2.1 video diffusion server that processes requests sequentially."""

    def __init__(self, config: WanConfig, save_videos: bool = False):
        self.config = config
        self.save_videos = save_videos
        self.pipeline = None
        self.results = []
        self.shutdown_requested = False
        self.processed_requests = 0
        self.stream = torch.cuda.Stream()

    def init_pipeline(self):
        """Initialize the Wan2.1 pipeline and (optionally) load Turbo weights."""
        logger.info(f"Loading base Wan pipeline from {self.config.model_path}...")
        self.pipeline: WanPipeline = WanPipeline.from_pretrained(
            self.config.model_path,
            torch_dtype=self.config.torch_dtype,
        )

        if self.config.turbo_repo:
            logger.info(
                f"Downloading Turbo checkpoint "
                f"{self.config.turbo_repo}/{self.config.turbo_ckpt}..."
            )
            ckpt_path = hf_hub_download(
                repo_id=self.config.turbo_repo,
                filename=self.config.turbo_ckpt,
            )
            logger.info("Loading Turbo weights into transformer...")
            state_dict = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Reconcile shape differences between the Turbo reference impl
            # and diffusers' WanTransformer3DModel.
            target_sd = self.pipeline.transformer.state_dict()
            adapted = {}
            for k, v in state_dict.items():
                if k in target_sd and v.shape != target_sd[k].shape:
                    if v.numel() == target_sd[k].numel():
                        logger.info(
                            f"Reshaping {k}: {tuple(v.shape)} -> "
                            f"{tuple(target_sd[k].shape)}"
                        )
                        v = v.reshape(target_sd[k].shape)
                    else:
                        logger.warning(
                            f"Shape mismatch on {k}: ckpt={tuple(v.shape)} "
                            f"model={tuple(target_sd[k].shape)} (numel differs, "
                            f"skipping)"
                        )
                        continue
                adapted[k] = v

            missing, unexpected = self.pipeline.transformer.load_state_dict(
                adapted, strict=False
            )
            logger.info(
                f"Turbo weights loaded: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
            if len(missing) > 0 and len(missing) > 0.5 * len(state_dict):
                logger.warning(
                    "More than half of expected keys are missing. The .pth "
                    "may use a different key prefix; check the TurboDiffusion "
                    "repo (github.com/thu-ml/TurboDiffusion) for the canonical "
                    "loader."
                )

        self.pipeline.set_progress_bar_config(disable=DISABLE_PROGRESS_BAR)
        self.pipeline = self.pipeline.to(self.config.device)
        logger.info("Pipeline initialized successfully")

    def load_requests_from_file(
        self, dataset_path: str, num_requests: int
    ) -> List[InferenceRequest]:
        """Load prompts from text file and create inference requests."""
        requests = []
        current_time = time.time()

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    prompt = line.strip()
                    if prompt:
                        decoded_prompt = prompt.replace("\\n", "\n")
                        request = InferenceRequest(
                            request_id=f"req{idx + 1}",
                            prompt=decoded_prompt,
                            arrival_time=current_time,
                        )
                        requests.append(request)
                        if num_requests and len(requests) >= num_requests:
                            break

            logger.info(f"Loaded {len(requests)} requests from {dataset_path}")
            return requests

        except FileNotFoundError:
            logger.error(f"Request file '{dataset_path}' not found.")
            return []
        except Exception as e:
            logger.error(f"Error loading requests: {str(e)}")
            return []

    def process_batch(self, batch: List[InferenceRequest]) -> List[InferenceResult]:
        """Process a batch of inference requests."""
        start_time = time.time()
        ids = [r.request_id for r in batch]
        prompts = [r.prompt for r in batch]

        logger.info(f"Processing batch {ids}: {len(batch)} request(s)")

        try:
            with torch.cuda.stream(self.stream):
                output = self.pipeline(
                    prompt=prompts,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    num_frames=self.config.num_frames,
                    height=self.config.height,
                    width=self.config.width,
                )
            self.stream.synchronize()

            end_time = time.time()
            inference_duration = end_time - start_time

            # Save videos if requested. WanPipeline returns .frames as a
            # list (one per sample) of frame lists.
            if self.save_videos and getattr(output, "frames", None):
                for request, frames in zip(batch, output.frames):
                    output_path = (
                        self.config.output_dir / f"{request.request_id}.mp4"
                    )
                    export_to_video(frames, str(output_path), fps=self.config.fps)

            actual_batch_size = len(batch)
            results = []
            for request in batch:
                results.append(InferenceResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    arrival_time=request.arrival_time,
                    start_time=start_time,
                    end_time=end_time,
                    inference_duration=inference_duration,
                    queue_wait_time=start_time - request.arrival_time,
                    batch_size=actual_batch_size,
                ))

            logger.info(f"Completed batch {ids} in {inference_duration:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Error processing batch {ids}: {str(e)}")
            end_time = time.time()
            actual_batch_size = len(batch)
            return [
                InferenceResult(
                    request_id=r.request_id,
                    prompt=r.prompt,
                    arrival_time=r.arrival_time,
                    start_time=start_time,
                    end_time=end_time,
                    inference_duration=end_time - start_time,
                    queue_wait_time=start_time - r.arrival_time,
                    batch_size=actual_batch_size,
                )
                for r in batch
            ]

    def save_log(self, output_file: str):
        """Save timing results to CSV file."""
        if not self.results:
            logger.warning("No results to save.")
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.config.output_dir / output_file
        with open(log_path, "w", encoding="utf-8") as f:
            # Write per-request latency (batch duration / batch size)
            for result in self.results:
                f.write(f"{result.inference_duration / result.batch_size:.3f}\n")

        logger.info(f"Timing log saved to {log_path}")

        total_requests = len(self.results)
        avg_inference_time = (
            sum(r.inference_duration for r in self.results) / total_requests
        )

        logger.info(f"{total_requests} requests processed")
        logger.info(f"Average inference time: {avg_inference_time:.2f}s")

    def signal_handler(self, signum, frame):
        logger.warning("\nReceived interrupt signal. Shutting down gracefully...")
        logger.info(f"Processed {self.processed_requests} requests so far.")
        self.shutdown_requested = True

    def run_server(self, dataset_path: str, num_requests: int, output_log: str):
        """Main server loop that processes all requests sequentially."""
        signal.signal(signal.SIGINT, self.signal_handler)

        self.init_pipeline()

        requests = self.load_requests_from_file(dataset_path, num_requests)
        if not requests:
            logger.error("No valid requests found. Exiting.")
            return

        logger.info(f"Starting to process {len(requests)} requests (batch_size={self.config.batch_size})...")

        # Process requests in batches (while-loop so dynamic batch_size takes effect)
        i = 0
        while i < len(requests):
            if self.shutdown_requested:
                logger.warning("Shutdown requested. Stopping processing.")
                break

            bs = self.config.batch_size
            batch = requests[i:i + bs]
            results = self.process_batch(batch)
            self.results.extend(results)
            self.processed_requests += len(results)
            i += len(batch)

        logger.info("\nProcessing completed. Saving log...")
        self.save_log(output_log)

        if self.shutdown_requested:
            logger.info("Server shutdown gracefully.")
        else:
            logger.info("All requests processed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Wan2.1 Video Diffusion Inference Server")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=None,
        help="Number of requests to process",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="stats.txt",
        help="Output file for timing results",
    )

    parser.add_argument(
        "--batch_size", type=int, default=default_config["batch_size"],
        help="Number of prompts to process per batch",
    )
    parser.add_argument("--model_path", type=str, default=default_config["model_path"])
    parser.add_argument(
        "--turbo_repo",
        type=str,
        default=default_config["turbo_repo"],
        help="HF repo of the Turbo .pth checkpoint. Pass empty string to disable.",
    )
    parser.add_argument(
        "--turbo_ckpt", type=str, default=default_config["turbo_ckpt"]
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=default_config["num_inference_steps"]
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=default_config["guidance_scale"]
    )
    parser.add_argument("--num_frames", type=int, default=default_config["num_frames"])
    parser.add_argument("--height", type=int, default=default_config["height"])
    parser.add_argument("--width", type=int, default=default_config["width"])
    parser.add_argument("--fps", type=int, default=default_config["fps"])
    parser.add_argument("--output_dir", type=str, default=default_config["output_dir"])
    parser.add_argument(
        "--save_videos", action="store_true", help="Save generated videos as .mp4"
    )

    args = parser.parse_args()

    setup_logger()

    config = WanConfig(
        model_path=args.model_path,
        turbo_repo=args.turbo_repo or None,
        turbo_ckpt=args.turbo_ckpt,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        output_dir=args.output_dir,
    )

    server = DiffusionInferenceServer(config, save_videos=args.save_videos)
    server.run_server(args.dataset_path, args.num_requests, args.log_file)


if __name__ == "__main__":
    main()
