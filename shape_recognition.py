import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import random
import time
from threading import Thread

# Inicializar pyttsx3 para síntese de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)
# Configurar voz em português
voices = engine.getProperty('voices')
for voice in voices:
    if "portuguese" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

model = load_model('shape_classifier.keras')
shapes = {
    'Circle': 'Circulo',
    'Square': 'Quadrado',
    'Triangle': 'Triangulo',
    'Rectangle': 'Retangulo'
}
shape_list = ['Circle', 'Square', 'Triangle', 'Rectangle']

# Variáveis para animações e efeitos visuais
animation_scale = 1.0
animation_growing = True
confidence_bar_width = 0
target_confidence_width = 0
button_hover = False
button_rect = None
last_spoken_time = 0
speak_cooldown = 1.0  # segundos

class AnimationState:
    def __init__(self):
        self.scale = 1.0
        self.growing = True
        self.color_index = 0
        self.colors = [
            (255, 182, 193),  # Rosa claro
            (255, 223, 186),  # Pêssego
            (186, 255, 228),  # Verde menta
            (173, 216, 230)   # Azul claro
        ]
        
    def update(self):
        if self.growing:
            self.scale += 0.02
            if self.scale >= 1.2:
                self.growing = False
        else:
            self.scale -= 0.02
            if self.scale <= 0.8:
                self.growing = True
                self.color_index = (self.color_index + 1) % len(self.colors)

    def get_current_color(self):
        return self.colors[self.color_index]

animation_state = AnimationState()

def speak_shape(shape_name):
    texto = f"Isto é um {shapes[shape_name]}"
    engine.say(texto)
    engine.runAndWait()

def speak_in_thread(shape_name):
    thread = Thread(target=speak_shape, args=(shape_name,))
    thread.daemon = True
    thread.start()

def preprocess_image(frame):
    img_resized = cv2.resize(frame, (64, 64))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

def draw_button(frame, text, position, size, is_hover=False):
    x, y = position
    width, height = size
    global button_rect
    button_rect = (x, y, width, height)
    
    # Efeito de hover
    color = (100, 200, 255) if is_hover else (70, 170, 225)
    
    # Desenhar botão com borda arredondada
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2, cv2.LINE_AA)
    
    # Centralizar texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def draw_confidence_bar(frame, confidence, y_position):
    global confidence_bar_width, target_confidence_width
    
    # Atualizar largura da barra de forma suave
    target_confidence_width = int(300 * confidence)
    confidence_bar_width += (target_confidence_width - confidence_bar_width) * 0.1
    
    # Desenhar barra de fundo
    cv2.rectangle(frame, (50, y_position), (350, y_position + 20), (70, 70, 70), -1)
    
    # Desenhar barra de progresso
    if confidence_bar_width > 0:
        # Cor baseada na confiança
        if confidence > 0.7:
            color = (0, 255, 0)  # Verde
        elif confidence > 0.5:
            color = (255, 165, 0)  # Laranja
        else:
            color = (0, 0, 255)  # Vermelho
            
        cv2.rectangle(frame, (50, y_position), 
                     (50 + int(confidence_bar_width), y_position + 20), 
                     color, -1)
    
    # Texto de porcentagem
    percentage = f"{int(confidence * 100)}%"
    cv2.putText(frame, percentage, (360, y_position + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def animate_shape(frame, shape, position, animation_state):
    scale = animation_state.scale
    color = animation_state.get_current_color()
    
    if shape == 'Circle':
        radius = int(50 * scale)
        cv2.circle(frame, position, radius, color, -1)
    elif shape == 'Square':
        side = int(100 * scale)
        top_left = (int(position[0] - side/2), int(position[1] - side/2))
        bottom_right = (int(position[0] + side/2), int(position[1] + side/2))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
    elif shape == 'Triangle':
        size = int(100 * scale)
        points = np.array([
            [position[0], position[1] - size],
            [position[0] - size, position[1] + size],
            [position[0] + size, position[1] + size]
        ], np.int32)
        cv2.fillPoly(frame, [points], color)
    elif shape == 'Rectangle':
        width = int(150 * scale)
        height = int(100 * scale)
        top_left = (int(position[0] - width/2), int(position[1] - height/2))
        bottom_right = (int(position[0] + width/2), int(position[1] + height/2))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)

def create_overlay(frame):
    overlay = frame.copy()
    # Adicionar um gradiente suave no topo
    height = 80
    for i in range(height):
        alpha = 1 - (i / height)
        cv2.line(overlay, (0, i), (frame.shape[1], i), 
                 (173, 216, 230), 1)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

# Inicialização da câmera e variáveis
cap = cv2.VideoCapture(0)
last_shape = None
speak_count = 0
shape_detected = False
detection_threshold = 0.7
detection_frames = 0
min_detection_frames = 5

# Função para lidar com eventos do mouse
def mouse_callback(event, x, y, flags, param):
    global button_hover, last_spoken_time
    if button_rect:
        button_x, button_y, button_w, button_h = button_rect
        button_hover = (button_x <= x <= button_x + button_w and 
                       button_y <= y <= button_y + button_h)
        
        if event == cv2.EVENT_LBUTTONDOWN and button_hover:
            current_time = time.time()
            if current_time - last_spoken_time >= speak_cooldown and last_shape:
                speak_in_thread(last_shape)
                last_spoken_time = current_time

# Criar janela e adicionar callback do mouse
cv2.namedWindow("Reconhecimento de Formas em Tempo Real")
cv2.setMouseCallback("Reconhecimento de Formas em Tempo Real", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Criar overlay com gradiente
    frame = create_overlay(frame)
    
    # Pré-processamento e predição
    img = preprocess_image(frame)
    predictions = model.predict(img, verbose=0)
    shape_idx = np.argmax(predictions)
    shape_label = shape_list[shape_idx]
    confidence = np.max(predictions)

    # Lógica de detecção
    if confidence > detection_threshold:
        detection_frames += 1
        if detection_frames >= min_detection_frames:
            shape_detected = True
            if shape_label != last_shape:
                speak_count = 0
                last_shape = shape_label
                speak_in_thread(shape_label)
                last_spoken_time = time.time()
    else:
        detection_frames = 0
        shape_detected = False

    # Interface visual
    # Desenhar botão de repetição
    if shape_detected and last_shape:
        draw_button(frame, "Repetir nome", (50, 100), (150, 40), button_hover)
        # Desenhar barra de confiança
        draw_confidence_bar(frame, confidence, 160)
        
        # Título com o nome da forma
        cv2.putText(frame, f"Forma: {shapes[last_shape]}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        
        # Animar a forma
        animation_state.update()
        animate_shape(frame, last_shape, (320, 240), animation_state)
    else:
        # Mensagem quando nenhuma forma é detectada
        cv2.putText(frame, "Aponte uma forma para a camera!", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Reconhecimento de Formas em Tempo Real!", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()