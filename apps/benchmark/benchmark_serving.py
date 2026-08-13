#!/usr/bin/env python3
"""Standalone BurstGPT text/video benchmark for OpenAI-compatible servers."""

import argparse
import asyncio
import base64
import csv
import json
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class SampleRequest:
    prompt: str
    prompt_len: int
    output_len: int
    timestamp: float
    multi_modal_content: dict | None = None


@dataclass
class RequestOutput:
    prompt_len: int
    generated_text: str = ""
    output_tokens: int = 0
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    error: str = ""


def load_burstgpt(
    path: str, tokenizer, count: int, time_scale: float
) -> list[SampleRequest]:
    requests = []
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["Model"] != "GPT-4" or int(row["Response tokens"]) <= 0:
                continue
            prompt_len = int(row["Request tokens"])
            token_ids = [
                (len(requests) + j) % tokenizer.vocab_size
                for j in range(prompt_len)
            ]
            requests.append(
                SampleRequest(
                    prompt=tokenizer.decode(token_ids),
                    prompt_len=prompt_len,
                    output_len=int(row["Response tokens"]),
                    timestamp=float(row["Timestamp"]) / time_scale,
                )
            )
            if len(requests) == count:
                break
    if len(requests) < count:
        raise ValueError(
            f"Requested {count} prompts, but only found "
            f"{len(requests)} usable rows"
        )
    requests.sort(key=lambda request: request.timestamp)
    return requests


def encode_video(path: Path) -> dict:
    video = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{video}"},
    }


def load_burstgpt_video(
    path: str,
    video_dir: str,
    tokenizer,
    count: int,
    time_scale: float,
    output_len: int,
) -> list[SampleRequest]:
    cache_root = Path(video_dir)
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{video_dir} has no manifest.json; build it with "
            "exp/data/build_mmvu_cache.py"
        )
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"{manifest_path} is empty")

    timestamps = []
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if int(row["Response tokens"]) > 0:
                timestamps.append(float(row["Timestamp"]) / time_scale)
                if len(timestamps) == count:
                    break
    if len(timestamps) < count:
        raise ValueError(
            f"Requested {count} prompts, but only found "
            f"{len(timestamps)} usable trace rows"
        )

    encoded: dict[int, dict] = {}
    requests = []
    for index, timestamp in enumerate(timestamps):
        entry_index = index % len(entries)
        entry = entries[entry_index]
        if entry_index not in encoded:
            video_path = (cache_root / entry["video"]).resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"Video in manifest is missing: {video_path}")
            encoded[entry_index] = encode_video(video_path)
        prompt = entry.get("question") or "Describe the video."
        prompt_len = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        requests.append(SampleRequest(
            prompt=prompt,
            prompt_len=prompt_len,
            output_len=output_len,
            timestamp=timestamp,
            multi_modal_content=encoded[entry_index],
        ))
    requests.sort(key=lambda request: request.timestamp)
    return requests


async def send_request(session, url, model, request, ignore_eos, progress, semaphore):
    output = RequestOutput(prompt_len=request.prompt_len)
    if request.multi_modal_content:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    request.multi_modal_content,
                ],
            }],
            "temperature": 0.0,
            "max_completion_tokens": request.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        payload = {
            "model": model,
            "prompt": request.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    if ignore_eos:
        payload["ignore_eos"] = True

    async def execute():
        started = time.perf_counter()
        most_recent = started
        first_chunk = True
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    return
                async for raw_line in response.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith(b":"):
                        continue
                    chunk = line.decode("utf-8").removeprefix("data: ")
                    if chunk == "[DONE]":
                        continue
                    data = json.loads(chunk)
                    if choices := data.get("choices"):
                        choice = choices[0]
                        text = (choice.get("delta", {}).get("content")
                                if request.multi_modal_content
                                else choice.get("text"))
                        now = time.perf_counter()
                        if first_chunk:
                            output.ttft = now - started
                            first_chunk = False
                        else:
                            output.itl.append(now - most_recent)
                        most_recent = now
                        output.generated_text += text or ""
                    if usage := data.get("usage"):
                        output.output_tokens = usage.get("completion_tokens") or 0
                output.latency = most_recent - started
                output.success = not first_chunk
                if not output.success:
                    output.error = "Never received a token-bearing streaming chunk"
        except Exception:
            output.error = "".join(traceback.format_exception(*sys.exc_info()))

    try:
        if semaphore is None:
            await execute()
        else:
            async with semaphore:
                await execute()
    finally:
        progress.update(1)
    return output


async def run_benchmark(args, requests):
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    connector = aiohttp.TCPConnector(limit=0)
    semaphore = (
        asyncio.Semaphore(args.max_concurrency)
        if args.max_concurrency
        else None
    )
    progress = tqdm(total=len(requests))
    tasks = []
    started = time.perf_counter()
    interrupted = False

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, trust_env=True
    ) as session:
        for request in requests:
            delay = started + request.timestamp - time.perf_counter()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                    interrupted = True
                    break
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                interrupted = True
                break
            tasks.append(asyncio.create_task(send_request(
                session,
                args.api_url,
                args.model,
                request,
                args.ignore_eos,
                progress,
                semaphore,
            )))

        pending = set(tasks)
        outputs = []
        # Continue observing the signal event while responses drain. A plain
        # gather() here would not wake when SIGTERM arrives after submission.
        while pending and not interrupted:
            if stop.is_set():
                interrupted = True
                break
            done, pending = await asyncio.wait(
                pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if not task.cancelled() and task.exception() is None:
                    outputs.append(task.result())

        if interrupted:
            # Preserve every response that completed before shutdown, then
            # cancel requests still waiting on or streaming from the server.
            for task in list(pending):
                if task.done() and not task.cancelled() and task.exception() is None:
                    outputs.append(task.result())
                elif not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    progress.close()
    return time.perf_counter() - started, outputs


def metric_summary(values):
    array = np.asarray(values or [0.0], dtype=float) * 1000
    return (
        float(np.mean(array)),
        float(np.median(array)),
        float(np.std(array)),
        float(np.percentile(array, 99)),
    )


def build_result(args, requests, duration, outputs):
    output_lens = []
    for output in outputs:
        if output.success and not output.output_tokens:
            output.output_tokens = len(
                args.tokenizer_obj.encode(
                    output.generated_text, add_special_tokens=False
                )
            )
        output_lens.append(output.output_tokens if output.success else 0)

    successful = [output for output in outputs if output.success]
    total_input = sum(output.prompt_len for output in successful)
    total_output = sum(output_lens)
    ttfts = [output.ttft for output in successful]
    tpots = [
        (output.latency - output.ttft) / (output.output_tokens - 1)
        for output in successful if output.output_tokens > 1
    ]
    itls_flat = [interval for output in successful for interval in output.itl]

    result = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "vllm",
        "mode": args.mode,
        "model_id": args.model,
        "tokenizer_id": args.tokenizer,
        "num_prompts": len(requests),
        "request_rate": "inf",
        "burstiness": 1.0,
        "max_concurrency": args.max_concurrency,
        "duration": duration,
        "completed": len(successful),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(successful) / duration,
        "request_goodput": None,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    for name, values in (("ttft", ttfts), ("tpot", tpots), ("itl", itls_flat)):
        mean, median, std, p99 = metric_summary(values)
        result[f"mean_{name}_ms"] = mean
        result[f"median_{name}_ms"] = median
        result[f"std_{name}_ms"] = std
        result[f"p99_{name}_ms"] = p99
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--mode", choices=("text", "video"), default="text")
    parser.add_argument("--video-dir",
                        help="local MMVU cache containing manifest.json")
    parser.add_argument("--expected-output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--time-scale", type=float, default=200.0)
    parser.add_argument("--api-url")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--result-dir", default=".")
    parser.add_argument("--result-filename")
    args = parser.parse_args()
    args.tokenizer = args.tokenizer or args.model
    if args.mode == "video" and not args.video_dir:
        parser.error("--video-dir is required with --mode=video")
    if args.expected_output_len <= 0:
        parser.error("--expected-output-len must be positive")
    if args.api_url is None:
        endpoint = "chat/completions" if args.mode == "video" else "completions"
        args.api_url = f"http://127.0.0.1:8000/v1/{endpoint}"
    if args.time_scale <= 0:
        parser.error("--time-scale must be positive")
    return args


def main():
    args = parse_args()
    args.tokenizer_obj = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    if args.mode == "video":
        requests = load_burstgpt_video(
            args.dataset_path,
            args.video_dir,
            args.tokenizer_obj,
            args.num_prompts,
            args.time_scale,
            args.expected_output_len,
        )
    else:
        requests = load_burstgpt(
            args.dataset_path,
            args.tokenizer_obj,
            args.num_prompts,
            args.time_scale,
        )
    duration, outputs = asyncio.run(run_benchmark(args, requests))
    result = build_result(args, requests, duration, outputs)
    filename = args.result_filename or (
        f"vllm-infqps-{args.model.rsplit('/', 1)[-1]}-{result['date']}.json"
    )
    path = Path(args.result_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=4)
    print(f"Successful requests: {result['completed']}")
    print(f"Benchmark duration (s): {duration:.2f}")
    print(f"Request throughput (req/s): {result['request_throughput']:.2f}")
    print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
