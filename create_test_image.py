#!/usr/bin/env python3
"""
Create simple test image for nonlinear material solver test
"""
import numpy as np
import cv2

# Image size
width, height = 100, 100

# Create image (BGR format)
img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background (air)

# Draw nonlinear magnetic core (gray) - outer ring
cv2.rectangle(img, (20, 20), (80, 80), (128, 128, 128), -1)  # BGR: gray

# Draw coil (red) - center
cv2.rectangle(img, (40, 40), (60, 60), (0, 0, 255), -1)  # BGR: red

# Save image
cv2.imwrite('test_nonlinear_image.png', img)
print("Created test_nonlinear_image.png: 100x100 pixels")
print("  Air (white): background")
print("  Core (gray): outer frame (nonlinear material)")
print("  Coil (red): center (high current)")
