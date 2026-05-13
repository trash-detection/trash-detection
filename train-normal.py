from ultralytics import YOLO


model = YOLO("models/trainedv8s.pt") # Pretrained weights
# Set L1 regularization coefficient to 0
model.train(
    sr=0,
    data="datasets/data.yaml",   # Dataset configuration
    epochs=1, 
    project='.', 
    name='runs/train-norm', 
    batch=48, 
    device="cpu"
)