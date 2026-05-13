from ultralytics import YOLO

weight = "weights/pruned.pt"

model = YOLO(weight)
model.train(
    data="ultralytics/cfg/datasets/coco.yaml", 
    epochs=200, 
    finetune=True, 
    device=0
)