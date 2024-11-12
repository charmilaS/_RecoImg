import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Diretório de dados
data_dir = 'data/geometric_shapes'
shapes = ['circle', 'square', 'triangle', 'rectangle']

# Carregar e pré-processar as imagens
X = []
y = []
for shape in shapes:
    shape_path = os.path.join(data_dir, shape)
    for filename in os.listdir(shape_path):
        img = cv2.imread(os.path.join(shape_path, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))
            X.append(img)
            y.append(shapes.index(shape))

X = np.array(X) / 255.0
y = np.array(y)

# Dividir em conjunto de treinamento e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Checkpoint para salvar o melhor modelo
checkpoint = ModelCheckpoint('shape_classifier.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Treinar o modelo
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
