from ultralytics import YOLO
import cv2

# Carregar o modelo treinado
model = YOLO('runs/detect/train24/weights/best.pt')  # Certifique-se de que o caminho está correto

# Abrir a webcam
cap = cv2.VideoCapture(0)


# Verifique se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
# Loop para processar o vídeo frame por frame
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Realizar a inferência
    results = model(frame)

    # Filtrar as detecções para aquelas que têm confiança maior que 70%
    for r in results:
        for box in r.boxes:  # Cada box é uma detecção de mão do YOLO
            confidence = box.conf[0]  # Confiança da detecção

        
                # Extrair as coordenadas da bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box

                # Desenhar a bounding box no frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Exibir o percentual de confiança no frame
            label = f'Conf: {confidence:.2f}'  # Formatar o texto da confiança
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # Desenhar o texto

    # Exibir o frame processado com as detecções
    cv2.imshow('Detecção de Mãos em Tempo Real', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
