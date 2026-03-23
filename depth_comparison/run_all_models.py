"""Run all 6 depth models on the sampled images. Run one model at a time to fit in 8GB VRAM."""
import os, sys, time, json, gc
import numpy as np
import cv2
import torch
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(
    os.environ.get("DATASET_DIR", os.path.join(BASE_DIR, "..", "..", "test", "test")),
    "images"
)
SAMPLE_FILE = os.path.join(BASE_DIR, "sample_images.txt")

def load_sample_list():
    with open(SAMPLE_FILE) as f:
        return [line.strip() for line in f if line.strip()]

def ensure_dirs(model_name):
    for sub in ["depth", "vis"]:
        os.makedirs(os.path.join(BASE_DIR, model_name, sub), exist_ok=True)

def save_depth(depth_np, stem, model_name):
    """Save raw 16-bit depth (in mm) and colorized visualization."""
    # Normalize to 16-bit for raw storage
    d = depth_np.astype(np.float32)
    if d.max() > d.min():
        d_norm = (d - d.min()) / (d.max() - d.min())
    else:
        d_norm = np.zeros_like(d)
    raw = (d_norm * 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(BASE_DIR, model_name, "depth", f"{stem}.png"), raw)

    # Colorized vis
    vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(BASE_DIR, model_name, "vis", f"{stem}.png"), vis)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ─── Model runners ───────────────────────────────────────────────────────────

def run_depth_anything_v2(samples):
    """Depth Anything V2 Large via HuggingFace Transformers."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth_anything_v2"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").cuda().eval()

    times = []
    for img_name in samples:
        stem = img_name.split(".")[0]
        image = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        depth = post[0]["predicted_depth"].cpu().numpy()
        save_depth(depth, stem, name)
        print(f"  {img_name} ({times[-1]:.3f}s)")

    del model, processor
    clear_gpu()
    return {"model": name, "times": times, "avg_time": np.mean(times)}


def run_depth_pro(samples):
    """Depth Pro (Apple) via HuggingFace Transformers."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth_pro"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf", trust_remote_code=True)
    model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf", trust_remote_code=True).cuda().eval()

    times = []
    for img_name in samples:
        stem = img_name.split(".")[0]
        image = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        depth = post[0]["predicted_depth"].cpu().numpy()
        save_depth(depth, stem, name)
        print(f"  {img_name} ({times[-1]:.3f}s)")

    del model, processor
    clear_gpu()
    return {"model": name, "times": times, "avg_time": np.mean(times)}


def run_marigold(samples):
    """Marigold LCM via Diffusers."""
    import diffusers
    name = "marigold"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16
    ).to("cuda")

    times = []
    for img_name in samples:
        stem = img_name.split(".")[0]
        image = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")

        t0 = time.time()
        output = pipe(image, num_inference_steps=4, ensemble_size=1)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        depth = output.prediction[0, 0].cpu().numpy()
        save_depth(depth, stem, name)
        print(f"  {img_name} ({times[-1]:.3f}s)")

    del pipe
    clear_gpu()
    return {"model": name, "times": times, "avg_time": np.mean(times)}


def run_pixel_perfect(samples):
    """Pixel-Perfect Depth via cloned repo."""
    name = "pixel_perfect"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    ppd_dir = os.path.join(BASE_DIR, "..", "pixel-perfect-depth")
    if not os.path.isdir(ppd_dir):
        print(f"  ERROR: Clone pixel-perfect-depth repo to {ppd_dir}")
        print(f"  git clone https://github.com/gangweix/pixel-perfect-depth {ppd_dir}")
        return {"model": name, "times": [], "avg_time": 0, "error": "repo not found"}

    sys.path.insert(0, ppd_dir)
    try:
        # PPD uses its own inference pipeline - run via subprocess for isolation
        import subprocess
        input_dir = os.path.join(BASE_DIR, name, "_input")
        os.makedirs(input_dir, exist_ok=True)

        # Copy sample images to temp input dir
        for img_name in samples:
            src = os.path.join(DATASET_DIR, img_name)
            dst = os.path.join(input_dir, img_name)
            if not os.path.exists(dst):
                cv2.imwrite(dst, cv2.imread(src))

        out_dir = os.path.join(BASE_DIR, name, "_raw_output")
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "run.py", "--input", input_dir, "--output", out_dir],
            cwd=ppd_dir, capture_output=True, text=True, timeout=3600
        )
        total_time = time.time() - t0

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            return {"model": name, "times": [], "avg_time": 0, "error": result.stderr[:500]}

        # Move outputs to standard format
        for img_name in samples:
            stem = img_name.split(".")[0]
            # PPD outputs depth as .npy or .png - check what exists
            for ext in [".npy", ".png"]:
                out_file = os.path.join(out_dir, stem + ext)
                if os.path.exists(out_file):
                    if ext == ".npy":
                        depth = np.load(out_file)
                    else:
                        depth = cv2.imread(out_file, -1).astype(np.float32)
                    save_depth(depth, stem, name)
                    break

        avg_time = total_time / len(samples) if samples else 0
        print(f"  Total: {total_time:.1f}s, avg: {avg_time:.3f}s/image")
        return {"model": name, "times": [avg_time] * len(samples), "avg_time": avg_time}
    finally:
        sys.path.pop(0)
        clear_gpu()


def run_vggt(samples):
    """VGGT (Meta) via cloned repo."""
    name = "vggt"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    vggt_dir = os.path.join(BASE_DIR, "..", "vggt")
    if not os.path.isdir(vggt_dir):
        print(f"  ERROR: Clone vggt repo to {vggt_dir}")
        print(f"  git clone https://github.com/facebookresearch/vggt {vggt_dir}")
        return {"model": name, "times": [], "avg_time": 0, "error": "repo not found"}

    sys.path.insert(0, vggt_dir)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda")

        times = []
        for img_name in samples:
            stem = img_name.split(".")[0]
            img_path = os.path.join(DATASET_DIR, img_name)
            images = load_and_preprocess_images([img_path]).to("cuda")

            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

            # Extract depth from predictions
            if "depth" in predictions:
                depth = predictions["depth"][0].cpu().numpy()
                if depth.ndim == 3:
                    depth = depth[0]
                save_depth(depth, stem, name)
            print(f"  {img_name} ({times[-1]:.3f}s)")

        del model
        clear_gpu()
        return {"model": name, "times": times, "avg_time": np.mean(times)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": name, "times": [], "avg_time": 0, "error": str(e)}
    finally:
        sys.path.pop(0)
        clear_gpu()


def run_depth_anything_v3(samples):
    """Depth Anything V3 via cloned repo."""
    name = "depth_anything_v3"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    da3_dir = os.path.join(BASE_DIR, "..", "Depth-Anything-3")
    if not os.path.isdir(da3_dir):
        print(f"  ERROR: Clone DA3 repo to {da3_dir}")
        print(f"  git clone https://github.com/ByteDance-Seed/Depth-Anything-3 {da3_dir}")
        return {"model": name, "times": [], "avg_time": 0, "error": "repo not found"}

    sys.path.insert(0, da3_dir)
    try:
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").cuda()

        times = []
        for img_name in samples:
            stem = img_name.split(".")[0]
            img_path = os.path.join(DATASET_DIR, img_name)

            t0 = time.time()
            prediction = model.inference([img_path])
            torch.cuda.synchronize()
            times.append(time.time() - t0)

            depth = prediction.depth[0]
            save_depth(depth, stem, name)
            print(f"  {img_name} ({times[-1]:.3f}s)")

        del model
        clear_gpu()
        return {"model": name, "times": times, "avg_time": np.mean(times)}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": name, "times": [], "avg_time": 0, "error": str(e)}
    finally:
        sys.path.pop(0)
        clear_gpu()


# ─── Main ────────────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "depth_anything_v2": run_depth_anything_v2,
    "depth_pro": run_depth_pro,
    "marigold": run_marigold,
    "pixel_perfect": run_pixel_perfect,
    "vggt": run_vggt,
    "depth_anything_v3": run_depth_anything_v3,
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_RUNNERS.keys()),
                        help="Which models to run (default: all)")
    args = parser.parse_args()

    samples = load_sample_list()
    print(f"Loaded {len(samples)} sample images")

    results = {}
    for model_name in args.models:
        if model_name not in MODEL_RUNNERS:
            print(f"Unknown model: {model_name}")
            continue
        try:
            result = MODEL_RUNNERS[model_name](samples)
            results[model_name] = result
        except Exception as e:
            print(f"FAILED {model_name}: {e}")
            results[model_name] = {"model": model_name, "error": str(e)}
        clear_gpu()

    # Save timing results
    timing_file = os.path.join(BASE_DIR, "timing_results.json")
    # Convert times lists for JSON serialization
    for k, v in results.items():
        if "times" in v:
            v["times"] = [float(t) for t in v["times"]]
        if "avg_time" in v:
            v["avg_time"] = float(v["avg_time"])
    with open(timing_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTiming results saved to {timing_file}")

if __name__ == "__main__":
    main()
