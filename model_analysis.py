from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# YOLO MODEL ANALYSIS ON EXDARK DATASET
# ==========================================

model = YOLO("best 1.pt")

print("YOLO model loaded successfully.")

confidence_values = [0.20, 0.25, 0.35, 0.45, 0.55]

analysis_results = []

for conf_value in confidence_values:
    print(f"\nAnalysing model with confidence threshold: {conf_value}")

    result = model.val(
        data="data.yaml",
        imgsz=640,
        batch=8,
        conf=conf_value,
        iou=0.5
    )

    analysis_results.append({
        "Confidence Threshold": conf_value,
        "Precision": float(result.box.mp),
        "Recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50-95": float(result.box.map)
    })

df = pd.DataFrame(analysis_results)

print("\nYOLO ExDark Analysis Results:")
print(df)

df.to_csv("exdark_yolo_analysis_results.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(df["Confidence Threshold"], df["Precision"], marker="o", label="Precision")
plt.plot(df["Confidence Threshold"], df["Recall"], marker="o", label="Recall")
plt.plot(df["Confidence Threshold"], df["mAP50"], marker="o", label="mAP50")
plt.plot(df["Confidence Threshold"], df["mAP50-95"], marker="o", label="mAP50-95")

plt.xlabel("Confidence Threshold")
plt.ylabel("Score")
plt.title("YOLO Performance Analysis on ExDark Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("exdark_yolo_analysis_graph.png")
plt.show()

print("\nAnalysis completed successfully.")
print("Saved: exdark_yolo_analysis_results.csv")
print("Saved: exdark_yolo_analysis_graph.png")