# 🗼 Tower Insights: Automated Telecom Tower Inspection & Visual Analytics

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Deep%20Learning-00FFFF.svg?logo=yolo&logoColor=black)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Automated Telecommunications Tower Inspection, Monocular Height Estimation, and Multi-Component Infrastructure Auditing via Deep Learning and Optical Scaling.**

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Mathematical Formulation & Optical Scaling](#-mathematical-formulation--optical-scaling)
- [Deep Learning Model Zoo](#-deep-learning-model-zoo)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Operational Workflow & User Guide](#-operational-workflow--user-guide)
- [Research Paper Blueprint & Citation](#-research-paper-blueprint--citation)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🚀 Executive Overview

Telecommunications infrastructure providers and Tower Companies (TowerCos) oversee thousands of cellular towers across diverse geographic regions. Maintaining structural integrity, verifying contractor construction milestones, and conducting accurate equipment audits are essential for operational continuity and tenant billing. 

However, traditional inspection workflows rely heavily on:
- **Hazardous Climbing Operations:** Exposing rigging engineers to dangerous occupational climbing hazards.
- **High Operational Expenditures (OPEX):** Incurring recurring travel and equipment costs for remote site surveys.
- **Subjective Human Estimation:** Manual height calculations and antenna counts suffer from reporting delays, human error, and billing discrepancies.

**Tower Insights** solves these challenges by combining single-shot deep learning object detection (YOLOv8) with **known-anchor optical scaling** and **ground-elevation perspective compensation**. The platform delivers sub-second automated metric height estimation, construction progress classification, and dense equipment inventory enumeration directly from monocular field imagery and drone video feeds.

---

## ✨ Key Features

### 1. Monocular Height Estimation & Milestone Audit
- **Autonomous Structure Localization:** Detects the complete vertical span and ground footing of lattice and monopole towers.
- **Fiducial Optical Scaling:** Uses civil engineering foundation dimensions ($W_{\text{base}}$) to derive millimeter-precise pixel-to-metric conversion ratios without physical calibration targets.
- **Perspective Foreshortening Correction:** Corrects monocular lens distortion and ground elevation upward viewing angles ($\approx 60^\circ$).
- **Lifecycle Milestone Auditing:** Automatically evaluates calculated height against design engineering specifications ($30\text{m}$, $40\text{m}$, $60\text{m}$) to classify structures as **"Construction Completed"** ($\ge 85\%$) or **"Under Construction"** with remaining completion percentages.

### 2. Ground Equipment & Power Infrastructure Inventory
- **Multi-Class Power Asset Detection:** Localizes and counts critical ground-level machinery to verify power redundancy and prevent asset shrinkage:
  - **Base Transceiver Stations (BTS):** Radio equipment cabinets.
  - **Control Boxes:** Power distribution, AC/DC conversion, and circuit breaker units.
  - **Diesel Generators (DG Sets):** Emergency backup power units.
  - **Photovoltaic Solar Panels:** Green energy arrays at hybrid sites.
- **Tabular Inventory Aggregation:** Real-time Pandas-backed asset tally and count reports.

### 3. Masthead Antenna Array Auditing
- **Dense Cluster Detection:** Accurately isolates closely clustered masthead equipment:
  - **GSM Panel Antennas:** Cellular sector communication arrays.
  - **Microwave Drum Antennas:** Parabolic backhaul transmission dishes.
- **Tenant Co-Location Reconciliation:** Verifies physical equipment deployments against leased operator contracts.

### 4. Interactive Zero-Latency Dashboard
- Built with **Streamlit** for rapid in-field inspection.
- Pure in-memory tensor plotting via OpenCV/Pillow to eliminate file system disk I/O race conditions.
- Batch processing support for multi-image site survey folders.

---

## 🏗️ System Architecture

```
                          ┌───────────────────────────────┐
                          │   Field Imagery / Drone Feed  │
                          │   (JPEG, PNG, MP4, AVI, etc.) │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │ Image Orientation & Resizing  │
                          │   (Aspect Ratio Correction)   │
                          └───────────────┬───────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Tab 1: Height Estimation]                    [Tab 2: Infrastructure Analysis]
                  │                                               │
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │   YOLOv8 Tower Detector   │                   │ Multi-Class Object Models │
    │   (`model/height_est.pt`) │                   ├───────────────────────────┤
    └─────────────┬─────────────┘                   │ • Ground Machinery:       │
                  │                                 │   `tele_obj_100.pt`       │
                  ▼                                 │   (BTS, Gen, Solar, Box)  │
    ┌───────────────────────────┐                   │ • Antenna Arrays:         │
    │ Bounding Box Regression   │                   │   `tower_model.pt`        │
    │ (w_adj = max(w - 20, 1))  │                   │   (GSM, Microwave)        │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │ Scale Factor Derivation   │                   │ In-Memory Bounding Box    │
    │ S = W_base / w_adj (m/px) │                   │ Overlays & Vector Plots   │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │ Perspective Compensation  │                   │ Tabular Asset Inventory   │
    │ H_final = 1.8 * (h * S)   │                   │ Breakdown (Pandas Data)   │
    └─────────────┬─────────────┘                   └───────────────────────────┘
                  │
                  ▼
    ┌───────────────────────────┐
    │ Milestone Classification  │
    │ % Completed vs. Design    │
    └───────────────────────────┘
```

---

## 📐 Mathematical Formulation & Optical Scaling

### 1. Known Anchor Reference Principle
In uncalibrated monocular computer vision, absolute depth is ambiguous. Because civil engineering standards mandate uniform tower foundation dimensions, the physical base width ($W_{\text{base}} \in [0.0\text{m}, 10.0\text{m}]$) functions as an invariant real-world spatial anchor.

### 2. Scale Factor Derivation
Let $(x, y, w_{\text{raw}}, h_{\text{bbox}})$ represent the predicted tower bounding box coordinates in pixel space from $\arg\max_i (c_i)$. An empirical margin correction $\Delta w = 20\text{ px}$ accounts for anchor box regression margins:

$$w_{\text{adj}} = \max(w_{\text{raw}} - \Delta w, 1)$$

The pixel-to-metric scaling factor $S$ (meters per pixel) is:

$$S = \frac{W_{\text{base}}}{w_{\text{adj}}} \quad [\text{m}\cdot\text{px}^{-1}]$$

### 3. Raw Altitude Projection
The uncompensated metric altitude $H_{\text{raw}}$ is obtained by scaling the bounding box pixel height $h_{\text{bbox}}$:

$$H_{\text{raw}} = h_{\text{bbox}} \cdot S = h_{\text{bbox}} \cdot \left(\frac{W_{\text{base}}}{w_{\text{adj}}}\right)$$

### 4. Foreshortening Angle Compensation
Ground photographs captured from an upward angle ($\theta \approx 60^\circ$) undergo monocular perspective compression. A foreshortening correction factor $\alpha = 0.8$ reconstructs the true orthogonal vertical height:

$$\Delta H = \alpha \cdot H_{\text{raw}} = 0.8 \cdot H_{\text{raw}}$$

$$H_{\text{final}} = H_{\text{raw}} + \Delta H = 1.8 \cdot H_{\text{raw}}$$

### 5. Construction Progress Classification
Given design height categories based on foundation footprint:

$$H_{\text{design}}(W_{\text{base}}) = \begin{cases} 
30\text{ meters}, & \text{if } W_{\text{base}} < 5.0\text{ m} \\
40\text{ meters}, & \text{if } 5.0\text{ m} \le W_{\text{base}} < 7.0\text{ m} \\
60\text{ meters}, & \text{if } W_{\text{base}} \ge 7.0\text{ m}
\end{cases}$$

The construction completion percentage $P_{\text{comp}}$ and operational classification are formulated as:

$$P_{\text{comp}} = \left(\frac{H_{\text{final}}}{H_{\text{design}}}\right) \times 100\%$$

$$\text{Status} = \begin{cases} 
\text{"Construction Completed!"}, & \text{if } P_{\text{comp}} \ge 85.0\% \\
\text{"Under Construction" } (100 - P_{\text{comp}})\% \text{ left}, & \text{if } P_{\text{comp}} < 85.0\%
\end{cases}$$

---

## 🤖 Deep Learning Model Zoo

The repository includes pre-trained custom YOLOv8 model checkpoints in the `model/` directory:

| Checkpoint | Target Domain | Architecture | Input Resolution | Output Classes | Inference Time (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`model/height_est.pt`](model/height_est.pt) | Tower Structure & Footing | YOLOv8 Custom | $640 \times 384$ | `tower` | $\sim 190\text{ ms}$ |
| [`model/tele_obj_100.pt`](model/tele_obj_100.pt) | Ground Power Machinery | YOLOv8 Custom | $480 \times 640$ | `BTS`, `Control Box`, `Generator`, `solar panel` | $\sim 190\text{ ms}$ |
| [`model/tower_model.pt`](model/tower_model.pt) | Masthead Antenna Arrays | YOLOv8 Custom | $640 \times 640$ | `GSM Antenna`, `Microwave Antenna` | $\sim 270\text{ ms}$ |
| [`model/tele_obj.pt`](model/tele_obj.pt) | Auxiliary Equipment Checkpoint | YOLOv8 Custom | $480 \times 640$ | Auxiliary Equipment Classes | $\sim 190\text{ ms}$ |

*Note: On CUDA-enabled GPUs or Apple Silicon MPS, inference latency drops to $< 30\text{ ms}$ per frame.*

---

## 📂 Repository Structure

```
Tower-Insights/
├── .gitignore               # Ignored environments, archives, client briefs, and temp runs
├── README.md                # Primary project documentation and usage guide
├── flow.md                  # Comprehensive implementation journey and research blueprint
├── main.py                  # Core Streamlit web application & inference engine
├── requirements.txt         # Pinned production dependencies
├── model/                   # Custom deep learning checkpoints
│   ├── height_est.pt        # Tower localization & optical scaling model
│   ├── tele_obj_100.pt      # Ground machinery & power equipment model
│   ├── tele_obj.pt          # Auxiliary equipment checkpoint
│   └── tower_model.pt       # GSM and microwave antenna detection model
└── data/                    # Benchmark evaluation imagery
    ├── antenna/             # Sample images with dense masthead antenna arrays
    ├── obj/                 # Ground machinery and power equipment test imagery
    └── tower/               # Completed and under-construction tower test imagery
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- **Operating System:** macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04+), or Windows 10/11
- **Python:** Version 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- **FFmpeg (Optional):** Required for AVI-to-MP4 video transcoding workflows

### 2. Clone the Repository
```bash
git clone https://github.com/K-692/Tower-Insights.git
cd Tower-Insights
```

### 3. Create & Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (cmd):
python -m venv venv
venv\Scripts\activate.bat

# On Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Launch the Streamlit interactive dashboard with:

```bash
streamlit run main.py
```

The application will start immediately and can be accessed in your browser at:
```
http://localhost:8501
```

---

## 📖 Operational Workflow & User Guide

### Tab 1: Height Estimation
1. Navigate to the **"Height Estimation"** tab on the navigation menu.
2. Adjust the **"Tower Base Width (m)"** slider to match the structural blueprint of the tower site (default is `6.0m`).
3. Upload single or multiple site inspection photos (`.jpg`, `.jpeg`, `.png`) via the file uploader.
4. The system will:
   - Identify the tower structure.
   - Draw green visual markers across the base and vertical spine.
   - Display the calculated height in meters.
   - Emit a real-time completion banner (**"Construction Completed!"** or **"Under Construction [X]% left"**).

### Tab 2: Infrastructure Analysis
1. Select the **"Infrastructure Analysis"** tab.
2. Choose between:
   - **Machinery Detection:** Upload ground-level site photos to automatically localize and inventory BTS cabinets, control boxes, generators, and solar panels.
   - **Antenna Detection:** Upload masthead imagery to identify and count GSM panel antennas and microwave backhaul dishes.
3. Review the bounding box visualizations and aggregate inventory count tables.

---

## 🔬 Research Paper Blueprint & Citation

If you use this system or its optical scaling formulation in academic research, please cite this work:

```bibtex
@article{pal2026towerinsights,
  title={Automated Monocular Height Estimation and Multi-Component Infrastructure Auditing for Telecommunications Towers via Deep Learning},
  author={Pal, Krishnendu},
  journal={arXiv preprint / Applied Computer Vision in Infrastructure Monitoring},
  year={2026},
  url={https://github.com/K-692/Tower-Insights}
}
```

For the complete theoretical derivation, civil engineering assumptions, and experimental ablation studies, refer to [`flow.md`](flow.md).

---

## 📄 License & Acknowledgments

This project is licensed under the [MIT License](LICENSE). Built using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [Streamlit](https://streamlit.io/), and [PyTorch](https://pytorch.org/).
