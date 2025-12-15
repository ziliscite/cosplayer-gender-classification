import numpy as np

class KNN:
  def __init__(self, k=5):
    self.k = k

  def fit(self, X, y):
    # X -> data fitur
    self.X_train = X
    # y -> label
    self.y_train = y

  def distance(self, x1, x2):
    # Euclidean distance = jarak langsung antara a dan b, yaitu akar((ax - bx)^2 + (ay - by)^2)
    # Karena tiap fitur memiliki lebih dari 1 data (plot point)
    # maka jarak dihitung dengan merata-rata selisih pangkat 2 antar tiap tiap plot point antara a dan b
    return np.sqrt(np.sum((x1 - x2)**2))

  def get_neighbours(self, pred):
    distances = []

    # Hitung tiap jarak antar data yang ingin diprediksi ke setiap data yang ada
    for (data, label) in zip(self.X_train, self.y_train):
      dist = self.distance(data, pred)
      distances.append((dist, label))

    # Sorting jarak (elemen pertama tuple)
    distances.sort(key=lambda x: x[0])

    # Ambil k jumlah tetangga
    neighbours = []
    for i in range(self.k):
      neighbours.append(distances[i][1])

    return neighbours

  def predict(self, pred):
    # Ambil k tetangga
    nearest_neighbours = self.get_neighbours(pred)

    # Hitung kelas:count
    counts = {}
    for cls in nearest_neighbours:
      counts[cls] = counts.get(cls, 0) + 1

    # Ambil kelas mayoritas
    majority = max(counts, key=counts.get)
    return majority

  def loocv(self):
    preds = []
    correct = 0

    # Ambil data uji dari tiap data fitur di list
    for i in range(len(self.X_train)):
      # Ambil data uji
      sample_to_predict = self.X_train[i]
      label = self.y_train[i]

      # Buang data uji dari list (sementara)
      X_temp = np.delete(self.X_train, i, axis=0)
      y_temp = np.delete(self.y_train, i, axis=0)

      # Backup list original
      X_train_backup = self.X_train
      y_train_backup = self.y_train

      # Set data training baru tanpa data uji
      self.X_train = X_temp
      self.y_train = y_temp

      # Predict data uji
      pred = self.predict(sample_to_predict)
      preds.append(pred)

      # Cek apakah prediksinya benar
      if pred == label:
          correct += 1

      # Kembalikan data fitur ke bentuk semula
      self.X_train = X_train_backup
      self.y_train = y_train_backup

    # Hitung akurasi
    accuracy = (correct / len(self.X_train)) * 100
    print(f"LOOCV Accuracy: {accuracy:.2f}% ({correct}/{len(self.X_train)})")

    return np.array(preds), accuracy