import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 載入 MNIST 數據集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 正規化圖片資料，使像素值範圍從 [0, 255] 變成 [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# 建立卷積神經網絡模型
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_images, train_labels, epochs=5)

# 評估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# 預測
predictions = model.predict(test_images)

# 顯示預測結果
print(f"Predicted label: {predictions[0].argmax()}")
print(f"True label: {test_labels[0]}")

# 顯示圖片
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()
