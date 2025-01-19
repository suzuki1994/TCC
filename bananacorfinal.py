import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Carregar o modelo YOLOv8
model = YOLO('roboflow/final completo/m100echos/m100echos/detect/train/weights/best.pt')

def analyze_banana_color(banana_image, min_confidence=0.5):
    # Realiza a detecção de objetos na imagem
    results = model(banana_image)

    # Filtra as detecções para manter apenas bananas com confiança mínima e classe correta
    bananas_detected = []
    for box in results[0].boxes:
        conf = box.conf.item()  # Converte a confiança para float
        class_id = int(box.cls.item())  # Converte o ID da classe para inteiro
        if conf >= min_confidence and class_id == 0:  # Classe 0 sendo 'banana'
            bananas_detected.append((box.xyxy[0], conf))  # Armazena a caixa e a confiança

    # Verificar se encontrou alguma banana
    if not bananas_detected:
        print("Nenhuma banana detectada com a confiança mínima.")
        return None, None, None
    
    # Escolhe a banana com maior confiança
    best_banana = max(bananas_detected, key=lambda x: x[1])
    (x1, y1, x2, y2) = map(int, best_banana[0])  # Pega as coordenadas da caixa

    # Extrai a região da banana detectada
    banana_image_cropped = banana_image[y1:y2, x1:x2]
    
    # Converte para o espaço de cor HSV e calcula o hue médio
    hsv = cv2.cvtColor(banana_image_cropped, cv2.COLOR_BGR2HSV)
    hue_mean = np.mean(hsv[:, :, 0])
    
    # Cálculo da maturação: 15 a 65, onde 15 = 100% (amarelo) e 65 = 0% (verde)
    if 15 <= hue_mean <= 65:
        maturity_percentage = (65 - hue_mean) / (65 - 20) * 100
    else:
        maturity_percentage = 100 if hue_mean < 20 else 0
    
    return maturity_percentage, hue_mean, (x1, y1, x2, y2)

# Função para desenhar a barra de maturação
def draw_maturity_bar(image, maturity_percentage, hue_mean, x1, y1, x2, y2):
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    bar_length = 500
    maturity_length = int(bar_length * (maturity_percentage / 100))
    
    # Desenha a barra de maturação
    cv2.rectangle(image, (50, 50), (50 + bar_length, 80), (255, 255, 255), -1)
    cv2.rectangle(image, (50, 50), (50 + maturity_length, 80), yellow, -1)
    cv2.rectangle(image, (50, 50), (50 + bar_length, 80), green, 2)

    # Exibe Hue e porcentagem de maturação
    cv2.putText(image, f'Hue: {hue_mean:.2f}', (50 + maturity_length + 5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(image, f'{maturity_percentage:.2f}%', (50 + maturity_length + 5, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Desenha a bounding box ao redor da banana
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image

# Processa imagens de uma pasta
def process_images_from_folder(input_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        image_resized = cv2.resize(image, (800, 600))
        
        max_maturity_percentage = 0
        max_hue_mean = 0
        max_box = None

        # Analisar cor da banana
        maturity_percentage, hue_mean, box = analyze_banana_color(image_resized)
        if maturity_percentage is not None:
            if maturity_percentage > max_maturity_percentage:
                max_maturity_percentage = maturity_percentage
                max_hue_mean = hue_mean
                max_box = box

        if max_box is not None:
            x1, y1, x2, y2 = max_box
            image_resized = draw_maturity_bar(image_resized, max_maturity_percentage, max_hue_mean, x1, y1, x2, y2)

        # Cria a pasta de saída caso não exista
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Salva a imagem com o mesmo nome
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, image_resized)
        print(f'Imagem salva em: {output_image_path}')

# Pasta de entrada e saída
input_folder = 'bananastest'
output_folder = 'resultados'

# Processar imagens
process_images_from_folder(input_folder, output_folder)
