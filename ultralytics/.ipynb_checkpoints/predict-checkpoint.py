import sys
sys.path.insert(0, "/home/jiayuan/yolom/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# model = YOLO('yolov8s-seg.pt')
number = 3 #input how many tasks in your work
model = YOLO('/home/jiayuan/ultralytics-main/ultralytics/runs/best.pt')  # 加载自己训练的模型# Validate the model
model.predict(source='/data/jiayuan/yolo8_multi/images/val2017', imgsz=(1280,740), device=[0,2],name='bdd-multi-pre', save=True, classes=[2,3,4,9,10,11], conf=0.25, iou=0.45, show_labels=False,boxes=False)
# metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[0,1,2,3],task='multi',classes=[2,3,4,9,10,11], iou=0.6,conf=0.001)  # no arguments needed, dataset and settings remembered
# for i in range(number):
#     print(f'This is for {i} work')
#     print(metrics[i].box.map)    # map50-95
#     print(metrics[i].box.map50)  # map50
#     print(metrics[i].box.map75)  # map75
#     print(metrics[i].box.maps)   # a list contains map50-95 of each category