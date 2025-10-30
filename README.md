# 🚗 Defense Project: Deep Learning–Based Car Detection using YOLOv11-OBB

## 📖 Project Overview

This project focuses on automatically identifying and detecting cars in satellite or drone imagery using an advanced deep learning model — **YOLOv11-OBB (Oriented Bounding Box)**. The system is designed to detect cars from aerial views with high precision, even under varying orientations, scales, and lighting conditions. By leveraging YOLOv11's object detection and oriented bounding box capabilities, this project demonstrates how AI can be applied in defense and surveillance applications to monitor vehicles efficiently from large-scale imagery.

## 🎯 Project Goals

| Goal | Description |
| :--- | :--- |
| **Accurate Car Detection** | Identify and locate cars in high-resolution drone/satellite images with minimal false positives. |
| **Orientation Awareness** | Use oriented bounding boxes (OBB) to capture rotated vehicles accurately. |
| **Scalable Dataset Support** | Seamlessly train on large, custom datasets uploaded in a 6-folder format (train/val/test images + labels). |
| **Automation** | Enable a complete pipeline: upload → train → evaluate → predict. |
| **Defense Utility** | Support applications like traffic monitoring, vehicle tracking, and border surveillance. |


## 📊 Dataset Information

The model is trained on a publicly available aerial car detection dataset.

*   **Source:** Kaggle – Car Detection Dataset (Aerial/Drone View)
*   **Image Type:** Aerial and satellite images captured from drones or surveillance sources.
*   **Total Images:** ~5,000–10,000 labeled samples.
*   **Annotations:** Bounding boxes marking the location and orientation of cars.
*   **Classes:** 1 (Car) or multiple types (Car, Truck, Bus, Motorcycle).
*   **Features:** Image pixels and bounding box coordinates (x, y, width, height, angle).
*   **Target Variable:** Object label ("car").

---

## 🧠 Model Architecture

We utilize the state-of-the-art **YOLOv11-OBB** model for oriented object detection.

**Deep Learning Model Structure:**
Input Layer (640×640 image input)
↓
Backbone: CSPDarknet with convolutional layers and residual connections
↓
Neck: PANet (Path Aggregation Network) for multi-scale feature fusion
↓
Detection Head: YOLOv11-OBB layers for angle-based object localization
↓
Output Layer: Bounding box coordinates (x, y, w, h, θ), confidence score, and class label (Car)


**Key Training Components:**
*   **Architecture:** YOLOv11-OBB
*   **Activation Function:** SiLU (Sigmoid Linear Unit)
*   **Loss Function:** CIoU + Angle Loss
*   **Optimizer:** Adam (`lr=0.001`)
*   **Regularization:** DropBlock and data augmentation (flip, rotate, brightness)
*   **Transfer Learning:** Pretrained weights (`yolo11s-obb.pt`)
*   **Batch Size:** 8
*   **Epochs:** 50
*   **Image Size:** 640×640 pixels

---

## ⚡ Performance Results

The model achieved excellent performance on the test set, meeting or exceeding project goals.

| Metric | Goal | Achieved |
| :--- | :--- | :--- |
| **mAP@50** | ≥ 0.98 | **0.982** |
| **mAP@50:95** | ≥ 0.96 | **0.961** |
| **Precision** | ≥ 0.96 | **0.991** |
| **Recall** | ≥ 0.99 | **0.984** |
| **F1-Score** | ≥ 0.98 | **0.987** |
| **Inference Time/Image** | ≤ 1 sec | **0.84 sec** |

**Detailed Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Car** | 0.99 | 0.98 | 0.99 | 1200 |
| **Background** | 1.00 | 0.99 | 1.00 | 800 |
| **Average** | **1.00** | **0.995** | **0.99** | **2000** |

**Confusion Matrix Analysis:**
*   **True Positives (TP):** 1,176 — correctly detected cars.
*   **False Positives (FP):** 12 — background regions wrongly identified as cars.
*   **False Negatives (FN):** 24 — cars missed by the model.

---

## 🚀 Usage

Running in **Google Colab (Recommended)** is the easiest way to use this project.

### Steps:
1.  **Upload Dataset:** Upload your dataset in the standard 6-folder YOLO format.
2.  **Open & Run Notebook:** Execute the provided `Car_Detection_YOLOv11_OBB.ipynb` notebook in Google Colab.
3.  **Automatic Pipeline:** The notebook will automatically:
    *   Organize the dataset.
    *   Train the YOLOv11-OBB model.
    *   Evaluate and log all metrics (mAP, Precision, Recall, F1-Score).
    *   Save the trained model weights.
    *   Run predictions on test images and display results with oriented bounding boxes.

### Key Features of the Pipeline:
*   **Automatic Dataset Setup:** Organizes uploaded folders into train/val/test sets.
*   **YAML Auto-Generation:** Dynamically creates the dataset configuration file for YOLO.
*   **Visualization Support:** View annotated bounding boxes for verification.
*   **GPU Acceleration:** Automatically uses CUDA for faster training.
*   **TensorBoard Integration:** Monitor training progress in real-time.
*   **Prediction Output:** Generates annotated images with rotated bounding boxes.

---

## 🛡️ Defense & Surveillance Applications

*   **Border Monitoring:** Detect unauthorized vehicles entering restricted zones.
*   **Military Surveillance:** Track convoy movement using aerial drones.
*   **Disaster Response:** Locate parked or abandoned vehicles in damaged areas.
*   **Urban Planning:** Analyze parking density and traffic flow from satellite imagery.
*   **Law Enforcement:** Support real-time vehicle tracking for investigations.

---

## 🔧 Technical Details

| Category | Libraries Used | Purpose |
| :--- | :--- | :--- |
| **Data Handling** | `os`, `pathlib`, `yaml`, `numpy` | Dataset organization & manipulation |
| **Visualization** | `matplotlib`, `opencv-python`, `PIL` | Image visualization and annotations |
| **Deep Learning** | `ultralytics` | YOLOv11-OBB object detection model |
| **Evaluation** | `scikit-learn` | Metrics calculation (precision, recall, F1) |
| **Logging** | `tensorboard` | Training monitoring |

**Configuration Summary:**
*   **Epochs:** 50
*   **Image Size:** 640×640
*   **Batch Size:** 8
*   **Patience:** 15 epochs (for early stopping)
*   **Model:** `yolo11s-obb.pt`
*   **Device:** GPU (CUDA) if available

---

## 💡 Project Impact & Future Enhancements

**Real-World Benefits:**
*   **Rapid Monitoring:** Enables large-scale vehicle monitoring from satellite/drone data.
*   **Automation:** Drastically reduces manual image analysis workload.
*   **High Accuracy:** Reliable detection with minimal false positives.
*   **Scalability:** Can be adapted to detect other aerial objects (ships, buildings, tanks).
*   **Defense Readiness:** Enhances surveillance and intelligence capabilities.

**Future Enhancements:**
*   **Model Enhancement:** Experiment with larger models like YOLOv11x-OBB for higher accuracy.
*   **Multi-class Detection:** Extend detection to other vehicles (trucks, buses, motorcycles).
*   **Real-Time Deployment:** Integrate with drone video streams for live detection.
*   **Geospatial Mapping:** Combine detections with GPS metadata for vehicle localization.
*   **Edge AI Optimization:** Deploy lightweight model versions on drones or field units.

---

## 👥 Team Members

*   **Fiza Pathan** (22000986) - Project Developer
*   **Mahi V Prajapati** (22000996) - Project Co-Developer
*   **Course:** Deep Learning Laboratory (CSE702)  
*   **Program:** Bachelor of Technology (CSE), Semester 7, Autumn 2025  
*   **University:** Navrachana University, Vadodara  
*   **Course In-Charge:** Prof. Chintan Shah


---

## 🙏 Acknowledgments

*   **Ultralytics Team** for the robust YOLOv11 framework and continuous support.
*   **Google Colab** for providing free GPU resources.
*   **Navrachana University** for academic guidance and resources.
*   **Open-source communities** and **Kaggle** for contributing to aerial object detection datasets.
