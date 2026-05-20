"""
Extended test image generator — 40 images covering edge cases,
real-world scenarios, borderline cases, and stress tests.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import os

OUTPUT_DIR = "/tmp/pixelation_test_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)

def save_jpg(arr, path, quality=90):
    Image.fromarray(arr.astype(np.uint8)).save(path, "JPEG", quality=quality)

def save_png(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, "PNG")

def pixelate(arr, block_size):
    """Downscale then upscale to create hard block boundaries."""
    h, w = arr.shape[:2]
    img = Image.fromarray(arr.astype(np.uint8))
    small = img.resize((w // block_size, h // block_size), Image.NEAREST)
    return np.array(small.resize((w, h), Image.NEAREST))

W, H = 640, 480
yy, xx = np.mgrid[0:H, 0:W]

print("\n" + "="*60)
print("  GENERATING EXTENDED TEST IMAGES (40 total)")
print("="*60)

# ─────────────────────────────────────────────────────────────
# PIXELATED — 20 images
# ─────────────────────────────────────────────────────────────
print("\n[PIXELATED — 20 images]")

# 1. Extreme pixelation (4x4 blocks)
arr = (128 + 80*np.sin(xx/30)*np.cos(yy/25)).clip(0,255)
arr = np.stack([arr, arr*0.8, arr*0.6], axis=2)
save_jpg(pixelate(arr, 4), f"{OUTPUT_DIR}/pix_01_extreme_4x4.jpg", quality=5)
print("  pix_01_extreme_4x4.jpg")

# 2. Heavy 8x8 block (JPEG DCT block size)
arr = np.stack([
    (128 + 100*np.sin(xx/20 + yy/30)).clip(0,255),
    (100 + 80*np.cos(xx/25)).clip(0,255),
    (150 + 70*np.sin(yy/20)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 8), f"{OUTPUT_DIR}/pix_02_heavy_8x8.jpg", quality=5)
print("  pix_02_heavy_8x8.jpg")

# 3. 16x16 macroblock (H.264 style)
arr = np.stack([
    (128 + 100*np.sin(xx/40 + yy/50)).clip(0,255),
    (100 + 80*np.cos(xx/45)).clip(0,255),
    (150 + 70*np.sin(yy/35)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 16), f"{OUTPUT_DIR}/pix_03_macroblock_16x16.jpg", quality=8)
print("  pix_03_macroblock_16x16.jpg")

# 4. 32x32 large blocks
arr = np.stack([
    (128 + 100*np.sin(xx/60 + yy/70)).clip(0,255),
    (100 + 80*np.cos(xx/65)).clip(0,255),
    (150 + 70*np.sin(yy/55)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 32), f"{OUTPUT_DIR}/pix_04_large_32x32.jpg", quality=10)
print("  pix_04_large_32x32.jpg")

# 5. Very low quality JPEG (quality=1)
arr = np.stack([
    (128 + 127*np.sin(xx*0.05)*np.cos(yy*0.04)).clip(0,255),
    (128 + 127*np.cos(xx*0.03 + yy*0.05)).clip(0,255),
    (128 + 127*np.sin(xx*0.04 + yy*0.03)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/pix_05_quality1_jpeg.jpg", quality=1)
print("  pix_05_quality1_jpeg.jpg")

# 6. Quality=3 JPEG
arr = np.stack([
    (128 + 100*np.sin(xx*0.08)*np.sin(yy*0.06)).clip(0,255),
    (100 + 80*np.cos(xx*0.07)).clip(0,255),
    (150 + 70*np.sin(yy*0.09)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/pix_06_quality3_jpeg.jpg", quality=3)
print("  pix_06_quality3_jpeg.jpg")

# 7. Checkerboard (hard edges at every pixel)
checker = ((xx // 8 + yy // 8) % 2) * 255
arr = np.stack([checker, checker*0.5, checker*0.3], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/pix_07_checkerboard.jpg", quality=5)
print("  pix_07_checkerboard.jpg")

# 8. Pixelated face-like (skin tones, pixelated)
arr = np.zeros((H, W, 3))
arr[:,:,0] = 200; arr[:,:,1] = 160; arr[:,:,2] = 120  # skin base
arr[100:300, 150:490, 0] = 220; arr[100:300, 150:490, 1] = 180
save_jpg(pixelate(arr, 12), f"{OUTPUT_DIR}/pix_08_skin_tone_pixelated.jpg", quality=8)
print("  pix_08_skin_tone_pixelated.jpg")

# 9. Dark scene pixelated (low light + compression)
arr = np.stack([
    (30 + 40*np.sin(xx/30 + yy/25)).clip(0,255),
    (20 + 30*np.cos(xx/35)).clip(0,255),
    (40 + 35*np.sin(yy/28)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 8), f"{OUTPUT_DIR}/pix_09_dark_scene.jpg", quality=5)
print("  pix_09_dark_scene.jpg")

# 10. Bright overexposed pixelated
arr = np.stack([
    (220 + 30*np.sin(xx/20)).clip(0,255),
    (210 + 25*np.cos(yy/25)).clip(0,255),
    (200 + 20*np.sin(xx/30 + yy/20)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 10), f"{OUTPUT_DIR}/pix_10_bright_overexposed.jpg", quality=6)
print("  pix_10_bright_overexposed.jpg")

# 11. Pixelated text-like (high contrast edges)
arr = np.zeros((H, W, 3))
for i in range(0, H, 20):
    arr[i:i+10, :, :] = 255
for j in range(0, W, 30):
    arr[:, j:j+15, :] = 200
save_jpg(pixelate(arr, 8), f"{OUTPUT_DIR}/pix_11_text_like.jpg", quality=5)
print("  pix_11_text_like.jpg")

# 12. Sports/motion blur + pixelation
arr = np.stack([
    (128 + 100*np.sin(xx*0.1 + yy*0.05)).clip(0,255),
    (80 + 120*np.cos(xx*0.08)).clip(0,255),
    (60 + 90*np.sin(yy*0.12)).clip(0,255),
], axis=2)
arr_img = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
arr = np.array(arr_img)
save_jpg(pixelate(arr, 16), f"{OUTPUT_DIR}/pix_12_motion_blur_pixelated.jpg", quality=8)
print("  pix_12_motion_blur_pixelated.jpg")

# 13. Banding artifact (color banding)
arr = np.zeros((H, W, 3))
for y in range(H):
    band = (y // 16) * 16
    arr[y, :, 0] = min(band, 255)
    arr[y, :, 1] = min(band * 0.8, 255)
    arr[y, :, 2] = min(band * 0.6, 255)
save_jpg(arr, f"{OUTPUT_DIR}/pix_13_color_banding.jpg", quality=5)
print("  pix_13_color_banding.jpg")

# 14. Mixed: half clean, half pixelated
arr_clean = np.stack([
    (128 + 80*np.sin(xx/80)*np.cos(yy/60)).clip(0,255),
    (100 + 60*np.cos(xx/90)).clip(0,255),
    (150 + 50*np.sin(yy/70)).clip(0,255),
], axis=2)
arr_pix = pixelate(arr_clean, 16)
arr_mixed = arr_clean.copy()
arr_mixed[:, W//2:, :] = arr_pix[:, W//2:, :]
save_jpg(arr_mixed, f"{OUTPUT_DIR}/pix_14_half_pixelated.jpg", quality=10)
print("  pix_14_half_pixelated.jpg")

# 15. Pixelated sky gradient
arr = np.zeros((H, W, 3))
for y in range(H):
    r = int(50 + (y/H)*150)
    g = int(100 + (y/H)*100)
    b = int(200 - (y/H)*80)
    arr[y, :] = [r, g, b]
save_jpg(pixelate(arr, 20), f"{OUTPUT_DIR}/pix_15_sky_pixelated.jpg", quality=5)
print("  pix_15_sky_pixelated.jpg")

# 16. Repeated save (re-compressed JPEG — generation loss)
arr = np.stack([
    (128 + 80*np.sin(xx/50)*np.cos(yy/40)).clip(0,255),
    (100 + 60*np.cos(xx/55)).clip(0,255),
    (150 + 50*np.sin(yy/45)).clip(0,255),
], axis=2)
tmp_path = "/tmp/tmp_recompress.jpg"
save_jpg(arr, tmp_path, quality=20)
for _ in range(5):  # re-compress 5 times
    img = Image.open(tmp_path)
    img.save(tmp_path, "JPEG", quality=20)
import shutil
shutil.copy(tmp_path, f"{OUTPUT_DIR}/pix_16_recompressed_5x.jpg")
print("  pix_16_recompressed_5x.jpg")

# 17. Pixelated with noise overlay
arr = np.stack([
    (128 + 100*np.sin(xx/30 + yy/25)).clip(0,255),
    (100 + 80*np.cos(xx/35)).clip(0,255),
    (150 + 70*np.sin(yy/28)).clip(0,255),
], axis=2)
arr_pix = pixelate(arr, 8).astype(float)
noise = rng.normal(0, 15, arr_pix.shape)
arr_noisy = (arr_pix + noise).clip(0, 255)
save_jpg(arr_noisy, f"{OUTPUT_DIR}/pix_17_pixelated_with_noise.jpg", quality=8)
print("  pix_17_pixelated_with_noise.jpg")

# 18. Pixelated portrait-like (face region)
arr = np.zeros((H, W, 3))
arr[:, :] = [70, 130, 180]   # blue background
arr[80:400, 160:480] = [220, 175, 130]  # face
arr[200:260, 200:260] = [50, 30, 20]    # eyes
arr[200:260, 380:440] = [50, 30, 20]
arr[320:360, 250:390] = [180, 80, 80]   # mouth
save_jpg(pixelate(arr, 12), f"{OUTPUT_DIR}/pix_18_portrait_pixelated.jpg", quality=8)
print("  pix_18_portrait_pixelated.jpg")

# 19. Pixelated sports field (green tones)
arr = np.stack([
    (60 + 30*np.sin(xx/20 + yy/15)).clip(0,255),
    (140 + 60*np.cos(xx/25)).clip(0,255),
    (40 + 20*np.sin(yy/18)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 16), f"{OUTPUT_DIR}/pix_19_sports_field.jpg", quality=6)
print("  pix_19_sports_field.jpg")

# 20. Pixelated night scene (very dark + blocks)
arr = np.stack([
    (15 + 20*np.sin(xx/15 + yy/12)).clip(0,255),
    (10 + 15*np.cos(xx/18)).clip(0,255),
    (25 + 30*np.sin(yy/14)).clip(0,255),
], axis=2)
save_jpg(pixelate(arr, 8), f"{OUTPUT_DIR}/pix_20_night_scene.jpg", quality=3)
print("  pix_20_night_scene.jpg")

# ─────────────────────────────────────────────────────────────
# CLEAN — 20 images
# ─────────────────────────────────────────────────────────────
print("\n[CLEAN — 20 images]")

# 1. Smooth sine wave gradient
arr = np.stack([
    (128 + 80*np.sin(xx/100)*np.cos(yy/80)).clip(0,255),
    (100 + 60*np.cos(xx/120 + yy/90)).clip(0,255),
    (150 + 50*np.sin((xx+yy)/150)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/clean_01_smooth_sine.jpg", quality=95)
print("  clean_01_smooth_sine.jpg")

# 2. Linear gradient horizontal
arr = np.zeros((H, W, 3))
arr[:, :, 0] = (xx / W * 255)
arr[:, :, 1] = (yy / H * 255)
arr[:, :, 2] = 128
save_png(arr, f"{OUTPUT_DIR}/clean_02_linear_gradient.png")
print("  clean_02_linear_gradient.png")

# 3. Radial gradient
cx, cy = W//2, H//2
dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
dist_norm = (dist / dist.max() * 255)
arr = np.stack([255 - dist_norm, dist_norm*0.5, dist_norm], axis=2)
save_png(arr, f"{OUTPUT_DIR}/clean_03_radial_gradient.png")
print("  clean_03_radial_gradient.png")

# 4. High quality photo-like (natural scene)
arr = np.stack([
    (128 + 80*np.sin(xx/150)*np.cos(yy/120)).clip(0,255),
    (100 + 60*np.cos(xx/180 + yy/140)).clip(0,255),
    (150 + 50*np.sin((xx+yy)/200)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/clean_04_high_quality_photo.jpg", quality=98)
print("  clean_04_high_quality_photo.jpg")

# 5. Solid blue (broadcast test card color)
arr = np.full((H, W, 3), [0, 100, 200], dtype=np.uint8)
save_png(arr, f"{OUTPUT_DIR}/clean_05_solid_blue.png")
print("  clean_05_solid_blue.png")

# 6. Solid green
arr = np.full((H, W, 3), [34, 139, 34], dtype=np.uint8)
save_png(arr, f"{OUTPUT_DIR}/clean_06_solid_green.png")
print("  clean_06_solid_green.png")

# 7. Sky gradient (clean)
arr = np.zeros((H, W, 3))
for y in range(H):
    t = y / H
    arr[y, :] = [int(135 + t*50), int(206 - t*80), int(235 - t*100)]
save_jpg(arr, f"{OUTPUT_DIR}/clean_07_sky_gradient.jpg", quality=95)
print("  clean_07_sky_gradient.jpg")

# 8. Gaussian blurred photo
arr = np.stack([
    (128 + 80*np.sin(xx/80)*np.cos(yy/60)).clip(0,255),
    (100 + 60*np.cos(xx/90)).clip(0,255),
    (150 + 50*np.sin(yy/70)).clip(0,255),
], axis=2)
blurred = np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)))
save_jpg(blurred, f"{OUTPUT_DIR}/clean_08_gaussian_blurred.jpg", quality=92)
print("  clean_08_gaussian_blurred.jpg")

# 9. Portrait-like clean (smooth skin tones)
arr = np.zeros((H, W, 3))
arr[:, :] = [70, 130, 180]
arr[80:400, 160:480] = [220, 175, 130]
arr[200:260, 200:260] = [50, 30, 20]
arr[200:260, 380:440] = [50, 30, 20]
arr[320:360, 250:390] = [180, 80, 80]
blurred = np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)))
save_jpg(blurred, f"{OUTPUT_DIR}/clean_09_portrait_clean.jpg", quality=95)
print("  clean_09_portrait_clean.jpg")

# 10. Sports field clean (green, smooth)
arr = np.stack([
    (60 + 30*np.sin(xx/80 + yy/60)).clip(0,255),
    (140 + 60*np.cos(xx/100)).clip(0,255),
    (40 + 20*np.sin(yy/70)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/clean_10_sports_field.jpg", quality=95)
print("  clean_10_sports_field.jpg")

# 11. Night scene clean (dark but smooth)
arr = np.stack([
    (15 + 20*np.sin(xx/60 + yy/50)).clip(0,255),
    (10 + 15*np.cos(xx/70)).clip(0,255),
    (25 + 30*np.sin(yy/55)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/clean_11_night_scene.jpg", quality=90)
print("  clean_11_night_scene.jpg")

# 12. Smooth noise (very blurred)
arr = rng.integers(80, 180, (H, W, 3), dtype=np.uint8)
blurred = np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(15)))
save_jpg(blurred, f"{OUTPUT_DIR}/clean_12_heavy_blur_noise.jpg", quality=90)
print("  clean_12_heavy_blur_noise.jpg")

# 13. Watercolor-like smooth
arr = np.stack([
    (180 + 40*np.sin(xx/200)*np.cos(yy/150)).clip(0,255),
    (140 + 50*np.cos(xx/180 + yy/160)).clip(0,255),
    (200 + 30*np.sin((xx+yy)/250)).clip(0,255),
], axis=2)
blurred = np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(8)))
save_jpg(blurred, f"{OUTPUT_DIR}/clean_13_watercolor_like.jpg", quality=92)
print("  clean_13_watercolor_like.jpg")

# 14. Broadcast test card (clean color bars)
arr = np.zeros((H, W, 3), dtype=np.uint8)
colors = [(255,255,255),(255,255,0),(0,255,255),(0,255,0),(255,0,255),(255,0,0),(0,0,255),(0,0,0)]
bar_w = W // len(colors)
for i, c in enumerate(colors):
    arr[:, i*bar_w:(i+1)*bar_w] = c
save_png(arr, f"{OUTPUT_DIR}/clean_14_color_bars.png")
print("  clean_14_color_bars.png")

# 15. Smooth vignette
arr = np.stack([
    (128 + 80*np.sin(xx/120)*np.cos(yy/100)).clip(0,255),
    (100 + 60*np.cos(xx/130)).clip(0,255),
    (150 + 50*np.sin(yy/110)).clip(0,255),
], axis=2)
vignette = 1 - (dist / dist.max()) * 0.6
arr = (arr * vignette[:,:,None]).clip(0, 255)
save_jpg(arr, f"{OUTPUT_DIR}/clean_15_vignette.jpg", quality=92)
print("  clean_15_vignette.jpg")

# 16. Overexposed clean (bright but smooth)
arr = np.stack([
    (220 + 30*np.sin(xx/150)).clip(0,255),
    (210 + 25*np.cos(yy/160)).clip(0,255),
    (200 + 20*np.sin(xx/140 + yy/130)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/clean_16_overexposed.jpg", quality=95)
print("  clean_16_overexposed.jpg")

# 17. Dark underexposed clean
arr = np.stack([
    (20 + 15*np.sin(xx/100)).clip(0,255),
    (15 + 12*np.cos(yy/110)).clip(0,255),
    (30 + 20*np.sin(xx/90 + yy/80)).clip(0,255),
], axis=2)
save_jpg(arr, f"{OUTPUT_DIR}/clean_17_underexposed.jpg", quality=90)
print("  clean_17_underexposed.jpg")

# 18. Smooth diagonal stripes (clean, no block artifacts)
stripe = ((xx + yy) % 40) / 40.0
arr = np.stack([
    (100 + 100*stripe).clip(0,255),
    (80 + 80*stripe).clip(0,255),
    (120 + 60*stripe).clip(0,255),
], axis=2)
blurred = np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)))
save_jpg(blurred, f"{OUTPUT_DIR}/clean_18_smooth_stripes.jpg", quality=92)
print("  clean_18_smooth_stripes.jpg")

# 19. Bokeh-like (blurred circles)
arr = np.zeros((H, W, 3))
for _ in range(20):
    cx2 = rng.integers(0, W)
    cy2 = rng.integers(0, H)
    r2 = rng.integers(30, 100)
    color = rng.integers(100, 255, 3)
    d2 = np.sqrt((xx - cx2)**2 + (yy - cy2)**2)
    mask = d2 < r2
    for c in range(3):
        arr[:,:,c] = np.where(mask, color[c], arr[:,:,c])
blurred = np.array(Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(20)))
save_jpg(blurred, f"{OUTPUT_DIR}/clean_19_bokeh_like.jpg", quality=92)
print("  clean_19_bokeh_like.jpg")

# 20. High quality re-saved (quality=95 twice — minimal generation loss)
arr = np.stack([
    (128 + 80*np.sin(xx/100)*np.cos(yy/80)).clip(0,255),
    (100 + 60*np.cos(xx/110 + yy/90)).clip(0,255),
    (150 + 50*np.sin((xx+yy)/140)).clip(0,255),
], axis=2)
tmp = "/tmp/tmp_hq.jpg"
save_jpg(arr, tmp, quality=95)
img = Image.open(tmp)
img.save(f"{OUTPUT_DIR}/clean_20_high_quality_resaved.jpg", "JPEG", quality=95)
print("  clean_20_high_quality_resaved.jpg")

print(f"\n✓ All 40 images saved to {OUTPUT_DIR}")
