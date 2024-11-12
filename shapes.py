import cv2
import numpy as np
import os
import random

# Diretório para salvar imagens
output_dir = 'data/geometric_shapes'
shapes = ['circle', 'square', 'triangle', 'rectangle']
os.makedirs(output_dir, exist_ok=True)

# Função para desenhar formas
def draw_shape(shape_name):
    img = np.ones((64, 64, 3), dtype='uint8') * 255
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    if shape_name == 'circle':
        cv2.circle(img, (32, 32), 20, color, -1)
    elif shape_name == 'square':
        cv2.rectangle(img, (16, 16), (48, 48), color, -1)
    elif shape_name == 'triangle':
        points = np.array([[32, 10], [10, 54], [54, 54]], np.int32)
        cv2.fillPoly(img, [points], color)
    elif shape_name == 'rectangle':
        cv2.rectangle(img, (10, 20), (54, 44), color, -1)

    return img

# Gerar 500 imagens para cada forma
for shape in shapes:
    shape_dir = os.path.join(output_dir, shape)
    os.makedirs(shape_dir, exist_ok=True)
    for i in range(500):
        img = draw_shape(shape)
        filename = os.path.join(shape_dir, f'{shape}_{i}.png')
        cv2.imwrite(filename, img)

print("Synthetic images generated successfully.")
