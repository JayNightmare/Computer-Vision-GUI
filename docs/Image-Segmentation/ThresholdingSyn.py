# Thresholding Techniques Tutorial using OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Simple Thresholding
# ---------------------------------------------
# Create a synthetic grayscale gradient image
simple_img = np.tile(np.arange(0, 256, dtype=np.uint8), (100, 1))

# Apply different simple thresholding techniques
_, binary = cv2.threshold(simple_img, 127, 255, cv2.THRESH_BINARY)
_, binary_inv = cv2.threshold(simple_img, 127, 255, cv2.THRESH_BINARY_INV)
_, trunc = cv2.threshold(simple_img, 127, 255, cv2.THRESH_TRUNC)
_, tozero = cv2.threshold(simple_img, 127, 255, cv2.THRESH_TOZERO)
_, tozero_inv = cv2.threshold(simple_img, 127, 255, cv2.THRESH_TOZERO_INV)

# Plot results for Simple Thresholding
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
simple_titles = ['Original', 'Binary', 'Binary INV', 'Trunc', 'ToZero', 'ToZero INV']
simple_images = [simple_img, binary, binary_inv, trunc, tozero, tozero_inv]

for i, ax in enumerate(axes.ravel()):
    ax.imshow(simple_images[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(simple_titles[i])
    ax.axis('off')
plt.suptitle("1. Simple Thresholding", fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 2. Adaptive Thresholding
# ---------------------------------------------
# Create an image with uneven illumination
x = np.linspace(0, 255, 256).astype(np.uint8)
adaptive_img = np.tile(x, (256, 1))
cv2.circle(adaptive_img, (128, 128), 60, (0,), -1)  # Dark circle in center

# Apply adaptive thresholding
adaptive_mean = cv2.adaptiveThreshold(adaptive_img, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 5)

adaptive_gaussian = cv2.adaptiveThreshold(adaptive_img, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 5)

# Plot results for Adaptive Thresholding
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
adaptive_titles = ['Original (Uneven Lighting)', 'Adaptive Mean', 'Adaptive Gaussian']
adaptive_images = [adaptive_img, adaptive_mean, adaptive_gaussian]

for i, ax in enumerate(axes):
    ax.imshow(adaptive_images[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(adaptive_titles[i])
    ax.axis('off')
plt.suptitle("2. Adaptive Thresholding", fontsize=16)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 3. Otsu's Thresholding
# ---------------------------------------------
# Create bimodal image (two distinct intensity levels)
otsu_img = np.zeros((100, 256), dtype=np.uint8)
otsu_img[:, :128] = 60
otsu_img[:, 128:] = 200

# Add random noise to simulate real-world case
noise = np.random.normal(0, 15, otsu_img.shape).astype(np.uint8)
otsu_img = cv2.add(otsu_img, noise)

# Apply Otsu's thresholding
_, otsu_thresh = cv2.threshold(otsu_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply Gaussian Blur before Otsu
blurred = cv2.GaussianBlur(otsu_img, (5, 5), 0)
_, otsu_blur_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot results for Otsu's Method
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
otsu_titles = ['Original Bimodal + Noise', "Otsu's Threshold", "Otsu's with Gaussian Blur"]
otsu_images = [otsu_img, otsu_thresh, otsu_blur_thresh]

for i, ax in enumerate(axes):
    ax.imshow(otsu_images[i], cmap='gray', vmin=0, vmax=255)
    ax.set_title(otsu_titles[i])
    ax.axis('off')
plt.suptitle("3. Otsu's Thresholding", fontsize=16)
plt.tight_layout()
plt.show()
