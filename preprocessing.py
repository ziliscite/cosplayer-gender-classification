import cv2

# get the face detector model to segment the face from the rest of the body
face_detector = cv2.FaceDetectorYN.create(
  "detector/face_detection_yunet_2023mar.onnx",
  "",
  (320, 320),
  score_threshold=0.75,
  nms_threshold=0.3,
  top_k=5000
)

def resize_image(image, max_size=800):
  h, w = image.shape[:2]
  max_dim = max(h, w)

  # kalau udah kecil, gak usah di-resize
  if max_dim <= max_size:
    return image, 1.0

  scale = max_size / max_dim
  new_w = int(w * scale)
  new_h = int(h * scale)
  resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

  # return resized image and scale factor
  return resized, scale

def detect_face(image, detector):
  h, w = image.shape[:2]
  detector.setInputSize((w, h))

  # detect beberapa wajah dari satu gambar
  _, faces = detector.detect(image)
  if faces is None:
    return None

  # select 1 face with highest confidence
  return max(faces, key=lambda f: f[14])

# bounding box (x, y, w, h) from deteksi wajah
def crop_face_square(image, x, y, w, h, padding=0.3):
  # image dimensions
  img_h, img_w = image.shape[:2]

  # compute center of the bounding box
  center_x = x + w // 2
  center_y = y + h // 2

  # square size: the larger of width/height, scaled by padding
  size = max(w, h)
  size = int(size * (1 + padding))

  # compute half-size and clamp coordinates to image borders
  half_size = size // 2
  x1 = max(0, center_x - half_size)
  y1 = max(0, center_y - half_size)
  x2 = min(img_w, center_x + half_size)
  y2 = min(img_h, center_y + half_size)

  # slice and return the cropped region
  cropped_face = image[y1:y2, x1:x2]
  return cropped_face

def resize_to_128x128(image, interpolation=cv2.INTER_AREA):
  return cv2.resize(image, (128, 128), interpolation=interpolation)

def convert_to_grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_histogram_equalization(gray_image):
  # CLAHE preserves local contrast and is generally better for face images.
  return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_image)