import numpy as np
from skimage.feature import hog, multiblock_lbp
from skimage.transform import integral_image
import cv2

def extract_hog_features(image):
  # HOG menghitung distribusi arah gradient untuk mendeteksi edge dan shape
  return hog(
    image,
    orientations=8,  # 8 arah gradient (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    pixels_per_cell=(16, 16),  # Setiap cell adalah 16×16 pixels, hitung histogram gradient per cell
    cells_per_block=(2, 2),  # Setiap block terdiri dari 2×2 cells (32×32 pixels), normalize kontras lokal
    visualize=False,
    channel_axis=None
  )

def extract_mblbp_features(image, patch_size=16):
  h, w = image.shape[:2]

  # Area efektif tempat patch (block) boleh digeser
  region_size = h - 2 * patch_size # Dikurangi agar tidak keluar batas
  step = patch_size // 2 # Step pergeseran patch (50% overlap)

  codes = []
  # Loop untuk tiap posisi patch
  for y in range(0, region_size, step):
    for x in range(0, region_size, step):
      # Ekstrak grid block 16×16 dari posisi (y, x)
      patch = image[y:y + patch_size, x:x + patch_size]

      # Downsample grid block 16×16 menjadi 3×3 matriks rata-rata intensitas
      small = cv2.resize(
        patch, (3, 3),
        interpolation=cv2.INTER_AREA
      )

      # Terapkan LBP pada matriks 3×3: bandingkan 8 tetangga dengan center
      center = small[1, 1]
      code = 0
      bit = 0

      # Iterasi semua sel 3x3 kecuali tengah
      for j in range(3):
        for i in range(3):
          if j == 1 and i == 1:
            continue

          # Bandingkan nilai tetangga dengan center
          # Jika >= center, maka set bit ke-n menjadi 1
          if small[j, i] >= center:
              code += (1 << bit)  # Increment code by 2^bit

          # Geser ke bit berikutnya
          bit += 1

      # Setiap grid block menghasilkan satu 8-bit LBP code (0-255)
      codes.append(code)

  # Buat histogram 256-dim (karena 8-bit, jadi 2^8 = 256 kemungkinan kode)
  hist, _ = np.histogram(codes, bins=256, range=(0, 256))

  # Normalisasi histogram
  hist = hist / hist.sum()
  return hist