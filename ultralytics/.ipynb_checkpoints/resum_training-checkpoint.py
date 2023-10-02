import sys
sys.path.insert(0, "/home/jiayuan/yolom/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
model = YOLO('/home/jiayuan/ultralytics-main/ultralytics/runs/multi/bddmulti-v3-640/weights/last.pt', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(resume=True,batch=12, device=[1,2,3],epochs=200, imgsz=(640,640),name='bddmulti-v3-640', val=True, task='multi')
