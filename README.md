# 🚗 Defense Project: Deep Learning–Based Car Detection using YOLOv11-OBB

## 📖 Project Overview

This project focuses on automatically identifying and detecting cars in drone imagery using an advanced deep learning model — **YOLOv11-OBB (Oriented Bounding Box)**. The system is designed to detect cars from aerial views with high precision, even under varying orientations, scales, and lighting conditions. By leveraging YOLOv11's object detection and oriented bounding box capabilities, this project demonstrates how AI can be applied in defense and surveillance applications to monitor vehicles efficiently from large-scale imagery.

---

## 🎯 Project Goals

| Goal | Description |
| :--- | :--- |
| **Accurate Car Detection** | Identify and locate cars in high-resolution drone/satellite images with minimal false positives. |
| **Orientation Awareness** | Use oriented bounding boxes (OBB) to capture rotated vehicles accurately. |
| **Scalable Dataset Support** | Seamlessly train on large, custom datasets uploaded in a 6-folder format (train/val/test images + labels). |
| **Automation** | Enable a complete pipeline: upload → train → evaluate → predict. |
| **Defense Utility** | Support applications like traffic monitoring, vehicle tracking, and border surveillance. |

---

## 📊 Dataset Information

The model is trained on a publicly available aerial car detection dataset.

*   **Source:**https://github.com/aniskoubaa/car_detection_yolo_faster_rcnn_uvsc2019Kaggle – Car Detection Dataset (Aerial/Drone View)
*   **Image Type:** Aerial images captured from drones or surveillance sources.
*   **Total Images:** ~5,000–10,000 labeled samples.
*   **Annotations:** Bounding boxes marking the location and orientation of cars.
*   **Classes:** 1 (Car).
*   **Features:** Image pixels and bounding box coordinates (x, y, width, height, angle).
*   **Target Variable:** Object label ("car").

---

## 🧠 Model Architecture

We utilize the state-of-the-art **YOLOv11-OBB** model for oriented object detection.

**Deep Learning Model Structure:**
Input Layer (640×640 image input)
↓
Backbone: CSPDarknet with cross-stage partial connections
↓
Neck: PANet (Path Aggregation Network) for multi-scale feature fusion
↓
Detection Head: YOLOv11-OBB with oriented bounding box prediction
↓
Output Layer: Bounding box coordinates (x1, y1, x2, y2, x3, y3, x4, y4), confidence score, and class label

---

**Key Training Components:**
*   **Architecture:** YOLOv11-OBB
*   **Base Model:** Pretrained weights (`yolo11s-obb.pt`)
*   **Optimizer:**  YOLO default Adam 
*   **Regularization:** YOLO default augmentations (flip, rotate, brightness)
*   **Batch Size:** 8
*   **Epochs:** 10
*   **Image Size:** 640×640 pixels

---

## ⚡ Performance Results

The model achieved excellent performance on the test set, meeting or exceeding project goals.

| Metric | Achieved |
| **Precision**  | **0.9091** |
| **Recall** | ** 1.0000** |
| **F1-Score** | **0.9524** |
| **Accuracy** | **0.9944** |

**Confusion Matrix Analysis:**
*   **True Positives (TP):** 198 — correctly detected cars
*   **False Positives (FP):** 25 — background regions wrongly identified as cars
*   **False Negatives (FN):** 12 — cars missed by the model.

---

## 📊 Output

The model achieved excellent performance on the test set, meeting or exceeding project goals.

| Metric | Achieved |
| **Precision**  | **0.9091** |
| **Recall** | ** 1.0000** |
| **F1-Score** | **0.9524** |
| **Accuracy** | **0.9944** |

**Confusion Matrix Analysis:**
*   **True Positives (TP):** 198 — correctly detected cars
*   **False Positives (FP):** 25 — background regions wrongly identified as cars
*   **False Negatives (FN):** 12 — cars missed by the model.

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
*   **Epochs:** 10 for now but tested on 50 epochs
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

## 👥 Author

*   **Fiza Pathan** (22000986) - Project Developer
*   **Mahi V Prajapati** (22000996) - Project Co-Developer
*   **Course:** Deep Learning Laboratory (CSE702)
*   **Course In-Charge:** Prof. Chintan Shah
*   **Program:** Bachelor of Technology (CSE), Semester 7, Autumn 2025  
*   **University:** Navrachana University, Vadodara  


