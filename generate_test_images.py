"""
Generate test images for pixelation detection accuracy testing.
Creates both pixelated and clean (non-pixelated) samples.
"""
import numpy as np
from PIL import Image
import os

OUTPUT_DIR = "/tmp/pixelation_test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── PIXELATED IMAGES ────────────────────────────────────────────────────────

def make_heavily_pixelated(path):
    """Simulate heavy JPEG/codec macroblocking by downscale→upscale."""
    img = Image.new("RGB", (640, 480))
    pixels = img.load()
    for y in range(480):
        for x in range(640):
            pixels[x, y] = (
                int(128 + 100 * np.sin(x / 20) * np.cos(y / 20)),
                int(100 + 80 * np.cos(x / 15)),
                int(150 + 70 * np.sin(y / 25)),
            )
    # Extreme downscale then upscale → hard block boundaries
    small = img.resize((40, 30), Image.NEAREST)
    pixelated = small.resize((640, 480), Image.NEAREST)
    pixelated.save(path, "JPEG", quality=5)
    print(f"  Created: {path}")

def make_moderate_pixelated(path):
    """Moderate pixelation — 16x16 macroblock style."""
    img = Image.new("RGB", (640, 480))
    pixels = img.load()
    for y in range(480):
        for x in range(640):
            pixels[x, y] = (
                int(80 + 120 * np.sin(x / 40 + y / 60)),
                int(60 + 100 * np.cos(x / 50)),
                int(100 + 90 * np.sin(y / 35)),
            )
    small = img.resize((80, 60), Image.NEAREST)
    pixelated = small.resize((640, 480), Image.NEAREST)
    pixelated.save(path, "JPEG", quality=15)
    print(f"  Created: {path}")

def make_blocky_compression(path):
    """Simulate JPEG compression artifacts with very low quality."""
    img = Image.new("RGB", (640, 480))
    pixels = img.load()
    for y in range(480):
        for x in range(640):
            # High-frequency content that compresses badly
            pixels[x, y] = (
                int(127 + 127 * np.sin(x * 0.3) * np.sin(y * 0.3)),
                int(127 + 127 * np.cos(x * 0.2) * np.sin(y * 0.4)),
                int(127 + 127 * np.sin(x * 0.4) * np.cos(y * 0.2)),
            )
    img.save(path, "JPEG", quality=2)
    print(f"  Created: {path}")

def make_grid_artifact(path):
    """Explicit 8x8 grid pattern simulating DCT block boundaries."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    block = 8
    for y in range(0, 480, block):
        for x in range(0, 640, block):
            color = (
                np.random.randint(50, 200),
                np.random.randint(50, 200),
                np.random.randint(50, 200),
            )
            img[y:y+block, x:x+block] = color
    Image.fromarray(img).save(path, "JPEG", quality=10)
    print(f"  Created: {path}")

def make_macroblock_pattern(path):
    """16x16 macroblock pattern like H.264 artifacts."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    block = 16
    for y in range(0, 480, block):
        for x in range(0, 640, block):
            base_r = int(128 + 100 * np.sin(x / 80 + y / 60))
            base_g = int(128 + 80 * np.cos(x / 70))
            base_b = int(128 + 90 * np.sin(y / 90))
            img[y:y+block, x:x+block] = (
                np.clip(base_r, 0, 255),
                np.clip(base_g, 0, 255),
                np.clip(base_b, 0, 255),
            )
    Image.fromarray(img).save(path, "JPEG", quality=8)
    print(f"  Created: {path}")

# ─── CLEAN IMAGES ────────────────────────────────────────────────────────────

def make_clean_gradient(path):
    """Smooth gradient — no artifacts."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        for x in range(640):
            img[y, x] = (
                int(x / 640 * 255),
                int(y / 480 * 255),
                int((x + y) / (640 + 480) * 255),
            )
    Image.fromarray(img).save(path, "PNG")
    print(f"  Created: {path}")

def make_clean_photo_like(path):
    """Smooth natural-looking image saved at high quality."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        for x in range(640):
            img[y, x] = (
                int(128 + 80 * np.sin(x / 100) * np.cos(y / 80)),
                int(100 + 60 * np.cos(x / 120 + y / 90)),
                int(150 + 50 * np.sin((x + y) / 150)),
            )
    Image.fromarray(img).save(path, "JPEG", quality=95)
    print(f"  Created: {path}")

def make_clean_solid_color(path):
    """Solid color — trivially clean."""
    img = Image.new("RGB", (640, 480), color=(120, 160, 200))
    img.save(path, "PNG")
    print(f"  Created: {path}")

def make_clean_smooth_noise(path):
    """Gaussian-blurred noise — smooth, no block boundaries."""
    from PIL import ImageFilter
    arr = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=8))
    img.save(path, "JPEG", quality=90)
    print(f"  Created: {path}")

def make_clean_high_quality_jpeg(path):
    """High-quality JPEG of a natural scene simulation."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        for x in range(640):
            # Sky-like gradient
            img[y, x] = (
                int(50 + (y / 480) * 150),
                int(100 + (y / 480) * 100),
                int(200 - (y / 480) * 80),
            )
    Image.fromarray(img).save(path, "JPEG", quality=92)
    print(f"  Created: {path}")

# ─── GENERATE ALL ────────────────────────────────────────────────────────────

print("\n=== Generating PIXELATED test images ===")
make_heavily_pixelated(f"{OUTPUT_DIR}/pixelated_heavy.jpg")
make_moderate_pixelated(f"{OUTPUT_DIR}/pixelated_moderate.jpg")
make_blocky_compression(f"{OUTPUT_DIR}/pixelated_blocky_compression.jpg")
make_grid_artifact(f"{OUTPUT_DIR}/pixelated_grid_8x8.jpg")
make_macroblock_pattern(f"{OUTPUT_DIR}/pixelated_macroblock_16x16.jpg")

print("\n=== Generating CLEAN test images ===")
make_clean_gradient(f"{OUTPUT_DIR}/clean_gradient.png")
make_clean_photo_like(f"{OUTPUT_DIR}/clean_photo_like.jpg")
make_clean_solid_color(f"{OUTPUT_DIR}/clean_solid_color.png")
make_clean_smooth_noise(f"{OUTPUT_DIR}/clean_smooth_noise.jpg")
make_clean_high_quality_jpeg(f"{OUTPUT_DIR}/clean_high_quality.jpg")

print(f"\nAll test images saved to: {OUTPUT_DIR}")
