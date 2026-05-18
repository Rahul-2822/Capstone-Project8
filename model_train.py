from ultralytics import YOLO

# ==========================================
# YOLO MODEL TRAINING ON EXDARK DATASET
# ==========================================

model = YOLO("best 1.pt")

print("YOLO model loaded successfully.")

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    lr0=0.001,
    patience=5,
    project="training_results",
    name="exdark_yolo_optimised_model"
)

print("\nYOLO training on ExDark dataset completed successfully.")