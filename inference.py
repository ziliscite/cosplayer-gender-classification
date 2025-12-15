import preprocessing as prep
import knn
import feature_extraction as feat
import numpy as np
import cv2

data = np.load("features/gender_features_v2.npz")
X = data["X"]
y = data["y"]

face_detector = prep.face_detector

model = knn.KNN(k=17)
model.fit(X, y)

def to_uint8(img):
  if img.dtype != np.uint8:
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
  return img

def draw_bbox(image, x, y, w, h, score):
  img_draw = image.copy()

  # draw bounding box
  cv2.rectangle(
    img_draw,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
  )

  # draw confidence text
  cv2.putText(
    img_draw,
    f"{score:.2f}",
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
  )

  return img_draw

def classify_image(image):
  # preprocessings
  resized_img, scale = prep.resize_image(image, max_size=512)
  best_face = prep.detect_face(resized_img, face_detector)
  if best_face is None:
    return None

  # get bounding box in resized image coordinates
  x_r, y_r, w_r, h_r = best_face[:4].astype(int)

  # map bounding box back to original image coordinates
  if scale != 1.0:
    x = int(x_r / scale)
    y = int(y_r / scale)
    w = int(w_r / scale)
    h = int(h_r / scale)
  else:
    x, y, w, h = x_r, y_r, w_r, h_r

  # filter faces by minimum size
  if w < 128 or h < 128:
    print(f"Image too small ({w}x{h}), skipping")
    return None

  cropped_face = prep.crop_face_square(image, x, y, w, h)
  if cropped_face.size == 0 or min(cropped_face.shape[:2]) < 32:
    print(f"Invalid crop, skipping")
    return None

  # preprocessings
  resized_face = prep.resize_to_128x128(cropped_face)
  gray_face = prep.convert_to_grayscale(resized_face)
  equalized_face = prep.apply_histogram_equalization(gray_face)

  # feature extraction
  hog_features = feat.extract_hog_features(equalized_face).astype(np.float32)
  mblbp_features = feat.extract_mblbp_features(equalized_face).astype(np.float32)
  features = np.concatenate([hog_features, mblbp_features])
  
  # classification
  predicted_label = model.predict(features)

  # draw bbox biar keren
  bbox = draw_bbox(image, x, y, w, h, best_face[14])
  return to_uint8(bbox), to_uint8(cropped_face), to_uint8(resized_face), to_uint8(gray_face), to_uint8(equalized_face), predicted_label
