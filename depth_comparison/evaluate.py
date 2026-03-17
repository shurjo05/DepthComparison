"""Evaluate depth model outputs: proxy metrics + visual comparison grids."""
import os, json
import numpy as np
import cv2

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "test", "test")
SAMPLE_FILE = os.path.join(BASE_DIR, "sample_images.txt")
GRID_DIR = os.path.join(BASE_DIR, "comparison_grids")

MODELS = ["depth_anything_v2", "depth_pro", "pixel_perfect", "vggt", "depth_anything_v3", "marigold"]

def load_sample_list():
    with open(SAMPLE_FILE) as f:
        return [line.strip() for line in f if line.strip()]

def load_yolo_bboxes(stem):
    """Load YOLO bboxes from label file, return list of (x1, y1, x2, y2) in pixels."""
    label_file = os.path.join(DATASET_DIR, "labels", f"{stem}.txt")
    if not os.path.exists(label_file):
        return []
    bboxes = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * 512)
            y1 = int((cy - h / 2) * 512)
            x2 = int((cx + w / 2) * 512)
            y2 = int((cy + h / 2) * 512)
            bboxes.append((max(0, x1), max(0, y1), min(512, x2), min(512, y2)))
    return bboxes

def compute_metrics(depth_path, rgb_path, bboxes):
    """Compute proxy metrics for a single depth map."""
    depth = cv2.imread(depth_path, -1)
    if depth is None:
        return None
    depth = depth.astype(np.float32)
    if depth.max() == depth.min():
        return None
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        return None

    metrics = {}

    if not bboxes:
        return None

    # Aggregate over all seed bboxes in image
    edge_scores, contrasts, smoothnesses = [], [], []
    for x1, y1, x2, y2 in bboxes:
        if x2 <= x1 or y2 <= y1:
            continue

        # 1. Edge alignment: Canny on RGB vs Canny on depth inside bbox
        rgb_crop = rgb[y1:y2, x1:x2]
        depth_crop = (depth_norm[y1:y2, x1:x2] * 255).astype(np.uint8)

        rgb_edges = cv2.Canny(rgb_crop, 50, 150)
        depth_edges = cv2.Canny(depth_crop, 50, 150)

        if rgb_edges.sum() > 0:
            # Fraction of RGB edges that have a nearby depth edge (within 3px)
            kernel = np.ones((7, 7), np.uint8)
            depth_edges_dilated = cv2.dilate(depth_edges, kernel)
            overlap = (rgb_edges > 0) & (depth_edges_dilated > 0)
            edge_scores.append(overlap.sum() / max(1, (rgb_edges > 0).sum()))

        # 2. Seed vs background contrast
        seed_mask = np.zeros((512, 512), dtype=bool)
        seed_mask[y1:y2, x1:x2] = True
        bg_mask = ~seed_mask

        seed_mean = depth_norm[seed_mask].mean()
        bg_mean = depth_norm[bg_mask].mean()
        contrasts.append(abs(seed_mean - bg_mean))

        # 3. Intra-seed smoothness (lower variance = smoother)
        seed_var = depth_norm[y1:y2, x1:x2].var()
        smoothnesses.append(seed_var)

    if not edge_scores:
        return None

    metrics["edge_alignment"] = float(np.mean(edge_scores))
    metrics["seed_bg_contrast"] = float(np.mean(contrasts))
    metrics["intra_seed_variance"] = float(np.mean(smoothnesses))
    return metrics


def make_comparison_grids(samples, n_grids=20):
    """Generate side-by-side grids: RGB | model1 | model2 | ... for n images."""
    os.makedirs(GRID_DIR, exist_ok=True)
    grid_samples = samples[:n_grids]

    for img_name in grid_samples:
        stem = img_name.split(".")[0]
        rgb = cv2.imread(os.path.join(DATASET_DIR, "images", img_name))
        if rgb is None:
            continue
        rgb = cv2.resize(rgb, (256, 256))

        panels = [rgb]
        labels = ["RGB"]

        for model in MODELS:
            vis_path = os.path.join(BASE_DIR, model, "vis", f"{stem}.png")
            if os.path.exists(vis_path):
                vis = cv2.imread(vis_path)
                vis = cv2.resize(vis, (256, 256))
            else:
                vis = np.zeros((256, 256, 3), dtype=np.uint8)
                cv2.putText(vis, "N/A", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2)
            panels.append(vis)
            labels.append(model)

        # Add labels on top
        labeled = []
        for panel, label in zip(panels, labels):
            p = panel.copy()
            cv2.putText(p, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            labeled.append(p)

        grid = np.hstack(labeled)
        cv2.imwrite(os.path.join(GRID_DIR, f"{stem}.png"), grid)

    print(f"Saved {len(grid_samples)} comparison grids to {GRID_DIR}")


def main():
    samples = load_sample_list()
    print(f"Evaluating {len(samples)} images across {len(MODELS)} models\n")

    all_metrics = {m: [] for m in MODELS}

    for img_name in samples:
        stem = img_name.split(".")[0]
        rgb_path = os.path.join(DATASET_DIR, "images", img_name)
        bboxes = load_yolo_bboxes(stem)

        for model in MODELS:
            depth_path = os.path.join(BASE_DIR, model, "depth", f"{stem}.png")
            if not os.path.exists(depth_path):
                continue
            m = compute_metrics(depth_path, rgb_path, bboxes)
            if m:
                all_metrics[model].append(m)

    # Aggregate
    summary = {}
    for model in MODELS:
        entries = all_metrics[model]
        if not entries:
            summary[model] = {"n_images": 0, "error": "no outputs"}
            continue
        summary[model] = {
            "n_images": len(entries),
            "edge_alignment_mean": float(np.mean([e["edge_alignment"] for e in entries])),
            "seed_bg_contrast_mean": float(np.mean([e["seed_bg_contrast"] for e in entries])),
            "intra_seed_variance_mean": float(np.mean([e["intra_seed_variance"] for e in entries])),
        }

    # Merge timing data if available
    timing_file = os.path.join(BASE_DIR, "timing_results.json")
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            timing = json.load(f)
        for model in MODELS:
            if model in timing and "avg_time" in timing[model]:
                summary[model]["avg_time_sec"] = timing[model]["avg_time"]

    # Print results
    print(f"\n{'Model':<22} {'Images':>6} {'Edge Align':>11} {'Contrast':>10} {'Smoothness':>11} {'Speed (s)':>10}")
    print("-" * 75)
    for model in MODELS:
        s = summary[model]
        if "error" in s:
            print(f"{model:<22} {'N/A':>6}")
            continue
        spd = f"{s.get('avg_time_sec', 0):.3f}" if "avg_time_sec" in s else "N/A"
        print(f"{model:<22} {s['n_images']:>6} {s['edge_alignment_mean']:>11.4f} "
              f"{s['seed_bg_contrast_mean']:>10.4f} {s['intra_seed_variance_mean']:>11.6f} {spd:>10}")

    # Save
    out_file = os.path.join(BASE_DIR, "results_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Generate comparison grids
    make_comparison_grids(samples)

if __name__ == "__main__":
    main()
