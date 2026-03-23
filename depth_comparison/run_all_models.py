"""Run all 6 depth models on the sampled images. One model at a time to manage VRAM."""
import os, sys, time, json, gc, subprocess
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
    """Normalize to 16-bit PNG + colorized INFERNO visualization."""
    d = depth_np.astype(np.float32)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    cv2.imwrite(os.path.join(BASE_DIR, model_name, "depth", f"{stem}.png"),
                (d_norm * 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(BASE_DIR, model_name, "vis", f"{stem}.png"),
                cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def timed(fn):
    t0 = time.time()
    result = fn()
    torch.cuda.synchronize()
    return result, time.time() - t0

# ─── Model runners ────────────────────────────────────────────────────────────

def run_depth_pro(samples):
    """Depth Pro (Apple) — metric depth via HuggingFace Transformers."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth_pro"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf", trust_remote_code=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf", trust_remote_code=True, torch_dtype=torch.float16
    ).cuda().eval()

    times = []
    for img_name in samples:
        stem = img_name.split(".")[0]
        image = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        t = time.time() - t0
        times.append(t)
        post = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        depth = post[0]["predicted_depth"].cpu().float().numpy()
        save_depth(depth, stem, name)
        print(f"  {img_name} ({t:.3f}s)")

    del model, processor
    clear_gpu()
    return {"model": name, "times": times, "avg_time": float(np.mean(times))}


def run_moge2(samples):
    """MoGe-2 (Microsoft) — metric depth + normals + intrinsics."""
    name = "moge2"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    # Pin utils3d to the commit MoGe requires (newer versions break the API)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"
        "@9a4eb15e4021b67b12c460c7057d642626897ec1"])
    try:
        from moge.model.v2 import MoGeModel
    except ImportError:
        print("  Installing MoGe...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/microsoft/MoGe.git"])
        from moge.model.v2 import MoGeModel

    device = torch.device("cuda")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

    times = []
    for img_name in samples:
        stem = img_name.split(".")[0]
        image_bgr = cv2.imread(os.path.join(DATASET_DIR, img_name))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

        t0 = time.time()
        with torch.no_grad():
            output = model.infer(tensor)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        depth = output["depth"].cpu().numpy()
        save_depth(depth, stem, name)
        print(f"  {img_name} ({times[-1]:.3f}s)")

    del model
    clear_gpu()
    return {"model": name, "times": times, "avg_time": float(np.mean(times))}


def run_depthfm(samples):
    """DepthFM (CompVis) — relative depth via flow matching."""
    name = "depthfm"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    depthfm_dir = os.path.join(BASE_DIR, "..", "depth-fm")
    ckpt_path = os.path.join(depthfm_dir, "checkpoints", "depthfm-v1.ckpt")

    if not os.path.isdir(depthfm_dir):
        print("  Cloning DepthFM repo...")
        subprocess.check_call(["git", "clone", "https://github.com/CompVis/depth-fm.git", depthfm_dir])
    # Install deps — skip pinned torch version but keep torchdiffeq and others
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchdiffeq", "einops", "omegaconf"])

    if not os.path.exists(ckpt_path):
        print("  Downloading DepthFM checkpoint (~1.7GB)...")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        subprocess.check_call(["wget", "-q", "-O", ckpt_path,
                               "https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt"])

    sys.path.insert(0, depthfm_dir)
    try:
        from depthfm import DepthFM
        model = DepthFM(ckpt_path)
        model.cuda().eval()

        times = []
        for img_name in samples:
            stem = img_name.split(".")[0]
            image = Image.open(os.path.join(DATASET_DIR, img_name)).convert("RGB")
            # Resize to nearest multiple of 64
            w, h = image.size
            nw = (w // 64) * 64 or 64
            nh = (h // 64) * 64 or 64
            image = image.resize((nw, nh), Image.LANCZOS)
            img_np = np.array(image).astype(np.float32)
            img_t = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            img_t = img_t.cuda()

            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    depth_t = model.predict_depth(img_t, num_steps=2, ensemble_size=1)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

            depth = depth_t[0, 0].cpu().numpy()
            save_depth(depth, stem, name)
            print(f"  {img_name} ({times[-1]:.3f}s)")

        del model
        clear_gpu()
        return {"model": name, "times": times, "avg_time": float(np.mean(times))}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": name, "times": [], "avg_time": 0, "error": str(e)}
    finally:
        sys.path.pop(0)
        clear_gpu()


def run_pixel_perfect(samples):
    """Pixel-Perfect Depth (gangweix) — diffusion-based relative depth."""
    name = "pixel_perfect"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    ppd_dir = os.path.join(BASE_DIR, "..", "pixel-perfect-depth")
    if not os.path.isdir(ppd_dir):
        print("  Cloning pixel-perfect-depth repo...")
        subprocess.check_call(["git", "clone",
                               "https://github.com/gangweix/pixel-perfect-depth", ppd_dir])
    # timm and einops are needed for the DINOv2 backbone import
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "einops"])

    input_dir = os.path.join(BASE_DIR, name, "_input")
    out_dir   = os.path.join(BASE_DIR, name, "_raw_output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(out_dir,   exist_ok=True)

    for img_name in samples:
        src = os.path.join(DATASET_DIR, img_name)
        dst = os.path.join(input_dir, img_name)
        if not os.path.exists(dst):
            cv2.imwrite(dst, cv2.imread(src))

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "run.py", "--input", input_dir, "--output", out_dir],
        cwd=ppd_dir, capture_output=True, text=True, timeout=7200
    )
    total_time = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr[:600]}")
        return {"model": name, "times": [], "avg_time": 0, "error": result.stderr[:600]}

    for img_name in samples:
        stem = img_name.split(".")[0]
        for ext in [".npy", ".png"]:
            out_file = os.path.join(out_dir, stem + ext)
            if os.path.exists(out_file):
                depth = np.load(out_file) if ext == ".npy" else cv2.imread(out_file, -1).astype(np.float32)
                save_depth(depth, stem, name)
                break

    avg_time = total_time / max(len(samples), 1)
    print(f"  Done — total {total_time:.1f}s, avg {avg_time:.3f}s/image")
    clear_gpu()
    return {"model": name, "times": [avg_time] * len(samples), "avg_time": avg_time}


def run_vggt(samples):
    """VGGT (Meta) — relative depth via cloned repo."""
    name = "vggt"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    vggt_dir = os.path.join(BASE_DIR, "..", "vggt")
    if not os.path.isdir(vggt_dir):
        print("  Cloning VGGT repo...")
        subprocess.check_call(["git", "clone",
                               "https://github.com/facebookresearch/vggt", vggt_dir])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r",
                               os.path.join(vggt_dir, "requirements.txt")])

    sys.path.insert(0, vggt_dir)
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()

        times = []
        for img_name in samples:
            stem = img_name.split(".")[0]
            images = load_and_preprocess_images([os.path.join(DATASET_DIR, img_name)]).to("cuda")

            t0 = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

            if "depth" in predictions:
                depth = predictions["depth"][0].cpu().numpy()
                while depth.ndim > 2:
                    depth = depth[0]
                save_depth(depth, stem, name)
            print(f"  {img_name} ({times[-1]:.3f}s)")

        del model
        clear_gpu()
        return {"model": name, "times": times, "avg_time": float(np.mean(times))}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": name, "times": [], "avg_time": 0, "error": str(e)}
    finally:
        sys.path.pop(0)
        clear_gpu()


def run_depth_anything_v3(samples):
    """Depth Anything V3 (ByteDance) — metric depth via cloned repo."""
    name = "depth_anything_v3"
    ensure_dirs(name)
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")

    da3_dir = os.path.join(BASE_DIR, "..", "Depth-Anything-3")
    if not os.path.isdir(da3_dir):
        print("  Cloning Depth-Anything-3 repo...")
        subprocess.check_call(["git", "clone",
                               "https://github.com/ByteDance-Seed/Depth-Anything-3", da3_dir])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", da3_dir])

    # Same utils3d pin as MoGe2 — DA3 uses the same lib
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"
        "@9a4eb15e4021b67b12c460c7057d642626897ec1"])

    sys.path.insert(0, da3_dir)
    try:
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained("depth-anything/da3metric-large").cuda().eval()

        times = []
        for img_name in samples:
            stem = img_name.split(".")[0]
            img_path = os.path.join(DATASET_DIR, img_name)

            t0 = time.time()
            with torch.no_grad():
                prediction = model.inference([img_path])
            torch.cuda.synchronize()
            times.append(time.time() - t0)

            save_depth(prediction.depth[0], stem, name)
            print(f"  {img_name} ({times[-1]:.3f}s)")

        del model
        clear_gpu()
        return {"model": name, "times": times, "avg_time": float(np.mean(times))}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"model": name, "times": [], "avg_time": 0, "error": str(e)}
    finally:
        sys.path.pop(0)
        clear_gpu()


# ─── Main ─────────────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "depth_pro":         run_depth_pro,
    "moge2":             run_moge2,
    "depthfm":           run_depthfm,
    "pixel_perfect":     run_pixel_perfect,
    "vggt":              run_vggt,
    "depth_anything_v3": run_depth_anything_v3,
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_RUNNERS.keys()),
                        help="Models to run (default: all)")
    args = parser.parse_args()

    samples = load_sample_list()
    print(f"Loaded {len(samples)} sample images")

    results = {}
    for model_name in args.models:
        if model_name not in MODEL_RUNNERS:
            print(f"Unknown model: {model_name}. Available: {list(MODEL_RUNNERS.keys())}")
            continue
        try:
            results[model_name] = MODEL_RUNNERS[model_name](samples)
        except Exception as e:
            print(f"FAILED {model_name}: {e}")
            results[model_name] = {"model": model_name, "error": str(e)}
        clear_gpu()

    timing_file = os.path.join(BASE_DIR, "timing_results.json")
    with open(timing_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTiming results saved to {timing_file}")

if __name__ == "__main__":
    main()
