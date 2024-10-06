import cv2
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('/runs/detect/train3/weights/best.pt') # Se debe completar la ruta segun donde lo tengas almacenado

# Inicializar la cámara
cap = cv2.VideoCapture(0) # Por lo general la camara web esta en el 0 pero segun los puetos (USB) puede estar en 3 o 4

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se puede recibir frames. Saliendo...")
        break

    results = model(frame)

    for result in results:
        if hasattr(result, 'boxes'):
            detections = result.boxes

            for detection in detections:
                confidence = detection.conf.item()
                if confidence > 0.2:

                    startX, startY, endX, endY = map(int, detection.xyxy[0])
                    label = f"Class {int(detection.cls.item())}: {confidence:.2f}"

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar las detecciones
    cv2.imshow("Detecciones en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
