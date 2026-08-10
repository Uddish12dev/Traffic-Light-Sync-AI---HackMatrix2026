# 🚦 Traffic Light Sync AI

### AI-Powered Dynamic Traffic Signal Optimization

**Hackathon:** HackMatrix 2026
**Team Name:** Mindmates

---

## 📌 Project Title

# Traffic Light Sync AI

Traffic Light Sync AI is an intelligent traffic-control system that analyzes traffic conditions at a signal and dynamically determines the appropriate traffic-light state based on real-time traffic parameters.

Instead of relying on conventional fixed-duration traffic signals, the system uses Artificial Intelligence and traffic analysis to make signal decisions according to the current traffic situation.

---

## 🎯 Problem Statement

Traditional traffic-light systems generally operate using predefined timers. These fixed timings do not account for the actual traffic density at a particular junction.

This can result in:

* 🚗 Unnecessary waiting even when traffic is low
* 🚦 Long queues at heavily congested lanes
* ⛽ Increased fuel consumption
* 🕐 Increased travel and waiting time
* 🌫️ Higher emissions caused by vehicles idling at signals
* 🚧 Inefficient utilization of available road capacity

A traffic signal should ideally respond to the **actual traffic conditions**, rather than blindly following a fixed timer.

---

## 💡 Solution Overview

**Traffic Light Sync AI** provides a data-driven approach to traffic signal control.

The system analyzes traffic-related parameters such as:

* Number of vehicles
* Average vehicle speed
* Lane occupancy
* Traffic flow rate
* Time of day
* Estimated waiting time

These features are processed by an AI model that performs two tasks:

### 1. 🧠 Waiting-Time Prediction

The model predicts the expected waiting time for the current traffic conditions using a regression head.

### 2. 🚦 Traffic-Light Classification

The model classifies the recommended traffic-light state into:

| Class | Signal    | Meaning                            |
| ----- | --------- | ---------------------------------- |
| `0`   | 🔴 Red    | Low traffic / transition state     |
| `1`   | 🟡 Yellow | Moderate traffic / transition      |
| `2`   | 🟢 Green  | High traffic / prioritize clearing |

The system also contains a **YOLOv8-based vehicle detection pipeline** that can identify vehicles from traffic-camera frames. Detected vehicles can then be tracked and used to estimate traffic conditions.

The overall architecture therefore combines:

**Traffic Data → AI Model → Traffic Analysis → Signal Decision**

---

## 🏗️ System Architecture

```text
              Traffic Camera / Dataset
                       │
                       ▼
              Vehicle Detection
                   YOLOv8
                       │
                       ▼
              Traffic Parameters
        ┌──────────────┼──────────────┐
        │              │              │
   Vehicle Count   Lane Occupancy   Flow Rate
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                Data Preprocessing
                       │
                       ▼
              PyTorch AI Model
                 ┌─────┴─────┐
                 │           │
                 ▼           ▼
          Waiting Time    Signal State
           Prediction     Classification
                 │           │
                 └─────┬─────┘
                       ▼
             Dynamic Traffic Signal
```

---

## 📊 Machine Learning Model

The core AI component is a **multi-task neural network built using PyTorch**.

The model contains:

* Shared fully connected layers
* ReLU activation
* Dropout regularization
* Waiting-time regression head
* Traffic-light classification head

The model jointly optimizes:

```text
Total Loss = Regression Loss + 10 × Classification Loss
```

The dataset is split into:

* **80% Training**
* **20% Validation**

Numerical features are normalized using statistics calculated from the training data to reduce data leakage.

---

## 🛠️ Technology Stack

### Programming Language

* **Python**

### Artificial Intelligence / Machine Learning

* **PyTorch**
* Neural Networks
* Multi-task Learning
* Regression
* Classification

### Computer Vision

* **YOLOv8**
* **Ultralytics**
* Vehicle Detection
* Centroid-based Object Tracking

### Data Processing

* **NumPy**
* Python CSV processing
* Feature normalization
* One-hot encoding

### Dataset

* `traffic_dataset.csv`

### Development & Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
Traffic-Light-Sync-AI---HackMatrix2026/
│
├── model.py
│       └── AI model, data preprocessing,
│           training and vehicle-detection pipeline
│
├── traffic_dataset.csv
│       └── Traffic dataset used for model training
│
├── HackMatrix ProjectDocumentation.pdf
│       └── Project documentation
│
└── README.md
        └── Project documentation
```

The repository currently contains the main model implementation, traffic dataset, and HackMatrix project documentation.

---

## 👥 Team Members

| Name               | Role        |
| ------------------ | ----------- |
| **Uddish Agarwal** | Team Leader |
| **Harshraj Zala**  | Member      |
| **Aryan Sharma**   | Member      |

---

## 📑 PPT / Presentation

**Project Presentation:**
👉 https://drive.google.com/file/d/1z1rMiaL12Sa3QQr-XQfP6n3BidFuTlsU/view?usp=sharing
---

## 🎥 Live Demonstration

**Live Demo:**
👉 https://drive.google.com/file/d/167MKkDB4uFF3K7usSJVGOAYA_dCOefwp/view?usp=sharing

> If the project is currently demonstrated locally rather than through a deployed web application, this section can instead link to a demonstration video.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Uddish12dev/Traffic-Light-Sync-AI---HackMatrix2026.git
```

### 2. Navigate to the Project

```bash
cd Traffic-Light-Sync-AI---HackMatrix2026
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

Install the primary Python dependencies:

```bash
pip install numpy torch ultralytics
```

If your environment requires a specific PyTorch build, install the appropriate version for your CPU/GPU from the official PyTorch distribution.

### 5. Verify the Dataset

Make sure the following file is present in the project directory:

```text
traffic_dataset.csv
```

The dataset should contain the traffic features required by the model:

```text
vehicle_count
average_speed
lane_occupancy
flow_rate
time_of_day
waiting_time
```

### 6. Run the Project

Run the main Python program according to the available command-line options:

```bash
python model.py --help
```

This displays the available training, evaluation, and processing options.

---

## 🚦 How It Works

The system follows the following pipeline:

### Step 1: Collect Traffic Information

Traffic information is obtained from the dataset or traffic-camera input.

### Step 2: Detect Vehicles

The computer-vision pipeline uses **YOLOv8** to detect relevant vehicle classes such as:

* Cars
* Motorbikes
* Buses
* Trucks

### Step 3: Track Vehicles

Detected vehicles can be tracked between frames using centroid-based tracking.

### Step 4: Calculate Traffic Conditions

The system derives important traffic indicators including:

* Vehicle count
* Lane occupancy
* Flow rate
* Average speed
* Waiting time

### Step 5: AI-Based Decision

The PyTorch model processes the traffic features and produces:

* Predicted waiting time
* Recommended traffic-light state

### Step 6: Dynamic Signal Control

The recommended signal state is selected according to the current traffic situation, allowing the system to prioritize congested areas rather than relying solely on fixed timers.

---

## 🌟 Key Features

* 🚦 Dynamic traffic-light decision making
* 🤖 AI-based traffic analysis
* 🚗 Vehicle detection using YOLOv8
* 📈 Waiting-time prediction
* 🧠 Multi-task neural network
* 📊 Traffic-density analysis
* 🔄 Vehicle tracking
* ⚡ Data-driven signal optimization
* 🛣️ Potential for real-time traffic-camera integration

---

## 🔮 Future Scope

The system can be extended with:

* Real-time CCTV integration
* Multiple-intersection coordination
* Reinforcement-learning-based signal optimization
* Emergency vehicle priority
* Pedestrian-aware signal control
* Cloud-based traffic monitoring
* IoT-enabled physical traffic lights
* Real-time traffic dashboards
* Historical traffic prediction
* City-wide traffic signal synchronization

---

## 📜 Project Documentation

Additional project documentation is available in:

```text
HackMatrix ProjectDocumentation.pdf
```

---

## 🔗 Links

| Resource              | Link                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| 💻 GitHub Repository  | [Traffic Light Sync AI](https://github.com/Uddish12dev/Traffic-Light-Sync-AI---HackMatrix2026) |
| 📑 PPT                | https://drive.google.com/file/d/1FSmsV1gJ5iyyxHJbekcdMiObrf0g7iNd/view?usp=sharing             |
| 🎥 Live Demonstration | https://drive.google.com/file/d/167MKkDB4uFF3K7usSJVGOAYA_dCOefwp/view?usp=sharing             |

---

## 🏆 HackMatrix 2026

Built for **HackMatrix 2026** with the goal of making traffic management smarter, more adaptive, and more efficient through Artificial Intelligence.

**Traffic Light Sync AI**
*Smarter Signals. Smoother Traffic. 🚦*
