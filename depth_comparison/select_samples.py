"""Select 200 random images from the dataset for depth model comparison."""
import os, random

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test", "test", "images")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sample_images.txt")
N_SAMPLES = 200
SEED = 42

def main():
    images = sorted(f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg"))
    random.seed(SEED)
    samples = random.sample(images, min(N_SAMPLES, len(images)))
    samples.sort(key=lambda x: int(x.split(".")[0]))

    with open(OUTPUT_FILE, "w") as f:
        for img in samples:
            f.write(img + "\n")

    print(f"Selected {len(samples)} images → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
