import sys
sys.path.insert(0, "/home/jiayuan/ultralytics-main/ultralytics")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/home/jiayuan/ultralytics-main/ultralytics/runs/best.pt')  # Validate the model
model.predict(source='/data/jiayuan/dash_camara_dataset/daytime', imgsz=(384,672), device=[3],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False)
