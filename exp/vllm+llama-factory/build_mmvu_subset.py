"""Build a 64-sample MMVU subset in LLaMA-Factory sharegpt format.

Run once from this directory:

    python3 build_mmvu_subset.py

Produces:
    mmvu_64.json           # registered as `mmvu_64` in dataset_info.json
    mmvu_64/<idx>.mp4      # actual video clips

Resolves the MMVU `video` field across the shapes the `datasets` library
may return (path string, dict with bytes/path, decoded video object).
"""

import json
import urllib.request
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "data"
VIDEO_DIR = OUT_DIR / "mmvu_64"
VIDEO_DIR.mkdir(exist_ok=True)
N = 64


def video_bytes(v):
    if isinstance(v, dict):
        if v.get("bytes"):
            return v["bytes"]
        if v.get("path"):
            with open(v["path"], "rb") as f:
                return f.read()
        raise ValueError(f"Video dict has no bytes/path: keys={list(v.keys())}")
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, str):
        if v.startswith(("http://", "https://")):
            with urllib.request.urlopen(v, timeout=60) as resp:
                return resp.read()
        with open(v, "rb") as f:
            return f.read()
    # Some datasets versions return a decoded object exposing `.filename`.
    filename = getattr(v, "filename", None)
    if filename:
        with open(filename, "rb") as f:
            return f.read()
    raise ValueError(f"Unsupported video field type: {type(v)}")


def main():
    print(f"Streaming MMVU and pulling first {N} samples...")
    ds = load_dataset("lmms-lab/MMVU", split="validation", streaming=True)

    samples = []
    seen = 0
    for ex in ds:
        if len(samples) >= N:
            break
        seen += 1
        try:
            data = video_bytes(ex["video"])
        except Exception as e:
            print(f"  [skip raw#{seen}] could not resolve video: {e}")
            continue

        idx = len(samples)
        out_path = VIDEO_DIR / f"{idx}.mp4"
        out_path.write_bytes(data)

        question = ex.get("question") or "Describe what happens in the video."
        answer = ex.get("answer")
        if answer is None:
            choices = ex.get("choices") or ex.get("options")
            answer_idx = ex.get("answer_idx")
            if choices and answer_idx is not None:
                answer = choices[answer_idx]
            else:
                answer = "A short description of the video."

        samples.append({
            "messages": [
                {"role": "user", "content": f"<video>{question}"},
                {"role": "assistant", "content": str(answer)},
            ],
            "videos": [f"mmvu_64/{idx}.mp4"],
        })

    print(f"Scanned {seen} raw samples, kept {len(samples)}.")

    out_json = OUT_DIR / "mmvu_64.json"
    out_json.write_text(json.dumps(samples, indent=2))
    print(f"Wrote {len(samples)} samples to {out_json}")
    print(f"Videos saved under {VIDEO_DIR}")


if __name__ == "__main__":
    main()
