# 🏭 Rolled Steel Surface Defect Detection

This project demonstrates a robust pipeline for detecting defects on rolled steel surfaces using a combination of YOLO (You Only Look Once) for object detection and Anomalib (an anomaly detection library) for identifying unusual patterns. A user-friendly Gradio interface is provided for real-time inspection.

## ✨ Features

- **YOLO-based Defect Localization**: Utilizes a YOLOv8n model trained on the NEU-DET dataset to accurately locate and classify various surface defects such as crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.
- **Anomaly Detection (Anomalib)**: Integrates the Anomalib library (specifically PatchCore) for unsupervised anomaly detection, which can identify novel or subtle defects not explicitly trained in the YOLO model.
- **Interactive Web UI with Gradio**: A simple and intuitive web interface built with Gradio allows users to upload steel images and get immediate defect detection results and anomaly scores.

## 🚀 Technologies Used

- `ultralytics`: For YOLOv8 object detection model training and inference.
- `anomalib`: A library for state-of-the-art anomaly detection methods (PatchCore).
- `gradio`: For creating the interactive web demonstration interface.
- `pandas`: Data manipulation and analysis (though not extensively used in the final deployed code).
- `kaggle`: For downloading the NEU-DET dataset.
- `torch`: PyTorch backend for deep learning models.
- `opencv-python (cv2)`: Image processing utilities.
- `numpy`: Numerical operations.

## 🛠️ Setup and Installation

To set up the project locally or in a Colab environment, follow these steps:

1.  **Clone the Repository (if applicable)**: (Assuming this project is in a repo)

    ```bash
    git clone your_repository_url
    cd your_repository_name
    ```

2.  **Install Dependencies**: The project relies on several Python packages. You can install them using pip:

    ```python
    !pip install -q ultralytics anomalib gradio roboflow gdown
    !pip install kaggle
    ```

3.  **Download the Dataset**: The NEU-DET dataset is crucial for training the defect detection model. You can download it from Kaggle.

    ```python
    # Make sure you have your Kaggle API key set up (kaggle.json in ~/.kaggle/)
    !mkdir -p /content/datasets/
    !kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database --unzip -p /content/datasets/
    ```

    *(Note: The initial setup cells in the notebook also include a git clone for a related repository and dataset YAML creation. Ensure these steps are executed to prepare the dataset structure.)*

## 🏃 How to Run

### 1. Prepare Dataset Configuration

The `data.yaml` file defines the dataset paths and class names for YOLO. This was generated in a previous step:

```python
# This cell was run previously to create /content/datasets/NEU-DET/data.yaml
# containing paths and class names.
import os
os.makedirs("/content/datasets/NEU-DET", exist_ok=True)
# ... yaml_content writing logic ...
```

### 2. Train the YOLO Model

Load a pre-trained YOLOv8 nano model and fine-tune it on the NEU-DET dataset. Replace `coco8.yaml` with the path to your dataset config, i.e., `/content/datasets/NEU-DET/data.yaml` after setting up the dataset correctly.

```python
from ultralytics import YOLO

model_yolo = YOLO("yolov8n.pt")

results = model_yolo.train(
    data="/content/datasets/NEU-DET/data.yaml", # IMPORTANT: Use the correct data path
    epochs=25,
    imgsz=640,
    batch=16,
    project="steel_defect_yolo",
    name="run1",
)

metrics = model_yolo.val()
print("YOLO Validation mAP50-95:", metrics.box.map)
```

### 3. Setup Anomalib (Anomaly Detection)

This section sets up the Anomalib data module and model (PatchCore). Training for Anomalib is commented out but can be enabled.

```python
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

datamodule = Folder(
    name="steel_surface",
    root="/content/datasets/steel_anomaly", # Make sure this path exists and contains appropriate train/test splits
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    train_batch_size=8,
    eval_batch_size=8,
)

patchcore_model = Patchcore(backbone="wide_resnet50_2")

engine = Engine(
    max_epochs=1, default_root_dir="/content/results/anomalib", accelerator="gpu"
)

# Uncomment to train and test the Anomalib model:
# engine.fit(datamodule=datamodule, model=patchcore_model)
# engine.test(datamodule=datamodule, model=patchcore_model)
```

### 4. Launch the Gradio Interface

This script loads the trained YOLO model and creates a Gradio application to visually inspect steel surfaces for defects.

```python
import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

# Load the trained YOLO model (ensure this path is correct after training)
yolo_detector = YOLO("/content/runs/detect/steel_defect_yolo/run1/weights/best.pt")

def inspect_steel(image):
    if image is None:
        return None, "No image provided."

    yolo_res = yolo_detector.predict(image, conf=0.25)[0]
    annotated_img = yolo_res.plot()

    detected_labels = [
        yolo_detector.names[int(c)] for c in yolo_res.boxes.cls.tolist()
    ]
    defect_counts = {
        name: detected_labels.count(name) for name in set(detected_labels)
    }

    report = "### Inspection Summary:\n"
    if detected_labels:
        report += f"**Status:** DEFECT DETECTED ({len(detected_labels)} total)\n\n"
        for k, v in defect_counts.items():
            report += f"- **{k}**: {v}\n"
    else:
        report += "**Status:** PASS (No surface defect detected)\n"

    return annotated_img, report

with gr.Blocks() as demo:
    gr.Markdown("# 🏭 Rolled Steel Surface Defect Detection")
    gr.Markdown(
        "Upload a steel sheet image to test the detection pipeline in real-time."
    )

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Input Steel Image")
        output_image = gr.Image(type="numpy", label="YOLO Defect Localization")

    status_box = gr.Markdown()
    submit_btn = gr.Button("Inspect Surface", variant="primary")

    submit_btn.click(
        fn=inspect_steel,
        inputs=[input_image],
        outputs=[output_image, status_box],
    )

demo.launch(share=True, debug=True)
```

After running the Gradio cell, a public URL will be generated, allowing you to access the interactive demo in your web browser.<img width="972" height="622" alt="pic1" src="https://github.com/user-attachments/assets/3f6d7a53-9f20-4858-a153-359bd789e815" />

