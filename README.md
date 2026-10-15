# Image Pixel Manipulator

Create, crop, flip, recolor and filter images using **only NumPy**, with no OpenCV or PIL and no image files. Every picture is built from a 3D array of shape `(height, width, 3)`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N4UZIGOLGQ7oWdu0Qy_f9dpg-RlBCzz8?usp=sharing)

## Problem

Images are just numbers. This project treats an image as a NumPy array to show how slicing, dtypes and boolean masks work on image data, which is the same representation used in computer vision and ML pipelines.

## What I built

| Step | What it does | NumPy concepts |
|---|---|---|
| 1 | Blank 200x200 RGB image, sky and grass filled in | `np.zeros`, `np.full`, `dtype=np.uint8` |
| 2 | Synthetic scene (house, roof, door, window, fence, path, sun) | 3D slicing, stepped slices, boolean circle mask |
| 3 | Crop, flip left-right and up-down, extract R/G/B channels, swap RGB to BGR | negative-step slicing, fancy indexing |
| 4 | Brighten and darken safely, find and recolor bright pixels | `astype(float32)`, `np.clip`, boolean masking |
| 5 | Memory comparison of `uint8` vs `float32` vs `float64` | `nbytes`, `itemsize` |
| Bonus | Grayscale, normalization, standardization, data augmentation | `@` (matrix multiply), `axis` reductions, `np.stack` |

## Key learnings

- **`uint8` overflows silently.** `200 + 100` becomes `44`, not `255`. The fix is to cast to `float32`, do the math, `np.clip` to 0-255, then cast back to `uint8`.
- **dtype affects memory.** The same 200x200 RGB image takes 117 KB as `uint8` and 937 KB as `float64`, which is 8x more.
- **Slicing creates views, not copies.** I used `.copy()` when I needed to edit without changing the original.
- **ML link:** models work with normalized `float32` arrays, so the bonus section covers grayscale, 0-1 normalization, per-channel standardization and random augmentation (flip, brightness, crop) into a batch of shape `(N, H, W, 3)`.

## Output

**Synthetic scene built with slicing**

<img width="607" height="634" alt="scene" src="https://github.com/user-attachments/assets/1d6bb229-12c1-4c73-ae26-85ebc8455ae4" />

**Crop and flips**

<img width="1807" height="507" alt="flips" src="https://github.com/user-attachments/assets/3c592f29-a433-4ab8-875f-2118a74a8146" />

**Brightness adjustment**

<img width="1668" height="504" alt="brightness" src="https://github.com/user-attachments/assets/54f93f7e-3a0b-402b-b3c4-8fe89e127622" />

**Augmented batch**

<img width="1807" height="483" alt="augmentation" src="https://github.com/user-attachments/assets/faa86286-e772-452d-8ea0-1e7bbc53aa13" />

## How to run

1. Click the **Open in Colab** badge above (or upload `image_pixel_manipulator.ipynb` to [Google Colab](https://colab.research.google.com)).
2. Run all cells (Runtime > Run all).

Requirements: Python 3, NumPy, Matplotlib (both pre-installed in Colab).

## Ideas for extension

- Blur filter using a sliding window
- Edge detection with a Sobel kernel
- Run the same functions on a real image loaded with `plt.imread`

## Author

Jaydev Singh | [GitHub](https://github.com/jayy0101)
