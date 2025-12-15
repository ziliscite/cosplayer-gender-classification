import numpy as np
from skimage.feature import hog, multiblock_lbp
from skimage.transform import integral_image
import cv2

def extract_hog_features(image):
  return hog(
    image,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    visualize=False,
    channel_axis=None
  )

def extract_mblbp_features(image, patch_size=16):
  h, w = image.shape[:2]
  region_size = h - 2 * patch_size
  step = patch_size // 2

  # int_img = integral_image(image)
  # codes = multiblock_lbp(int_img, 0, 0, 3, 3)

  codes = []
  for y in range(0, region_size, step):
    for x in range(0, region_size, step):
      patch = image[y:y + patch_size, x:x + patch_size]

      small = cv2.resize(patch, (3, 3), interpolation=cv2.INTER_AREA)

      # compute MB-LBP code
      center = small[1, 1]
      code = 0
      bit = 0
      for j in range(3):
        for i in range(3):
          if j == 1 and i == 1:
            continue
          code |= (1 << bit) if small[j, i] >= center else 0
          bit += 1

      codes.append(code)

  # Histogram 256 dim
  hist, _ = np.histogram(codes, bins=256, range=(0, 256))
  hist = hist / hist.sum()
  return hist