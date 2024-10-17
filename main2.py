from ultralytics import YOLO
import cv2
import mediapipe as mp

# Carregar o modelo treinado do YOLO
model = YOLO('runs/detect/train28/weights/best.pt')  # Certifique-se de que o caminho está correto

# Inicializar MediaPipe para mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Abrir o vídeo (substitua 'caminho/do/video.mp4' pelo caminho real do seu vídeo)
video_path = 'd3.mp4'  # Coloque o caminho do seu vídeo aqui
cap = cv2.VideoCapture(video_path)

# Verifique se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Loop para processar o vídeo frame por frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar a inferência com YOLO para detectar mãos
    results = model(frame)

    # Filtrar as detecções para aquelas que têm confiança maior que 60%
    for r in results:
        for box in r.boxes:  # Cada box é uma detecção do YOLO

            # Extrair a confiança da detecção
            confidence = box.conf[0].item()

            # Apenas considere detecções com confiança maior que 60%
            if confidence >= 0.6:
                # Extrair as coordenadas da bounding box e garantir que são inteiros
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Recortar a região da mão da imagem original
                hand_region = frame[y1:y2, x1:x2]

                # Converter a região da mão para RGB, pois o MediaPipe usa RGB
                hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)

                # Processar a região da mão com o MediaPipe para detectar landmarks
                result = hands.process(hand_region_rgb)

                # Se landmarks forem detectados, desenhá-los na imagem original
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Desenhar os landmarks da mão na imagem original com o offset
                        mp_draw.draw_landmarks(frame[y1:y2, x1:x2], hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Exibir o frame processado com as detecções
    cv2.imshow('Detecção de Mãos em Vídeo', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()
