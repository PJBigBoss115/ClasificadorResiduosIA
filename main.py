from ultralytics import YOLO

# Cargar el modelo YOLOv8 pre-entrenado
model = YOLO('yolov8n.pt')  # yolov8n es el modelo más ligero

# Realizar la detección en una imagen de prueba
results = model('https://ultralytics.com/images/zidane.jpg')

# Mostrar los resultados usando plot
results[0].plot()  # results es una lista, accedemos al primer elemento