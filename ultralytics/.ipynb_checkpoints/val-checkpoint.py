import sys
sys.path.insert(0, "/home/jiayuan/yolom/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# model = YOLO('yolov8s-seg.pt')
# number = 3 #input how many tasks in your work
model = YOLO('/home/jiayuan/ultralytics-main/ultralytics/runs/best.pt')  # 加载自己训练的模型# Validate the model
# metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[4],task='multi',name='v3-model-val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered

metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[3],task='multi',name='val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered
# for i in range(number):
#     print(f'This is for {i} work')
#     print(metrics[i].box.map)    # map50-95
#     print(metrics[i].box.map50)  # map50
#     print(metrics[i].box.map75)  # map75
#     print(metrics[i].box.maps)   # a list contains map50-95 of each category