
# 📘 Smart Traffic Violation Detection System

*A complete AI-powered traffic violation detection system using YOLOv11, ByteTrack, and PaddleOCR with a Flask-based web dashboard.*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-2.x-black?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?logo=ai&logoColor=black" />
  <img src="https://img.shields.io/badge/ByteTrack-MOT-ff6600" />
  <img src="https://img.shields.io/badge/PaddleOCR-2.x-005ea5?logo=paddlepaddle&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite-database-07405e?logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

# 📄 1. Project Overview

The **Smart Traffic Violation Detection System** is a full-stack AI application that automatically detects, tracks, and documents traffic violations from images and videos.
It integrates cutting-edge computer vision models to build a complete intelligent traffic-monitoring workflow.

This project includes:

* **YOLOv11** for violation detection (helmet, red-light, stop-line, mobile phone, triple riding,…)
* **ByteTrack** for multi-object tracking in videos
* **PaddleOCR** for Vietnamese license plate recognition
* **Flask** web backend and user interface
* **SQLite** for structured record storage

Users can upload media, review detection output, manage violation records, and analyze statistics on an interactive dashboard.

---

# 🚀 2. Key Features

### 🚦 Violation Detection

* Red-light running
* Stop-line crossing
* No helmet
* Triple riding
* Using mobile phone
* Vehicle type detection

### 🎥 Video Tracking (ByteTrack)

* Multi-object ID tracking
* Frame-level decision making
* Automatic snapshot extraction

### 🔠 License Plate Recognition

* PaddleOCR VN license plate reading
* Regex-based validation
* Auto-cleaning & normalization

### 🗂 Record Management

* Approve / Reject violations
* Edit license plates
* View images/videos
* Export CSV files

### 📊 Dashboard Analytics

* Violation distribution chart
* Daily statistics
* Approval rates
* Top frequent license plates
* Date-based filtering

---

# 🧠 3. Technology Stack

| Category      | Technology        |
| ------------- | ----------------- |
| Backend       | Python, Flask     |
| Detection     | YOLOv11           |
| Tracking      | ByteTrack         |
| OCR           | PaddleOCR         |
| Frontend      | TailwindCSS, HTML |
| Database      | SQLite            |
| Visualization | Chart.js          |

---

# 🖼️ 4. Screenshots (UI Overview)

> Add images into `assets/` folder before using the paths below.

### 1️⃣ Upload Interface

![Upload](assets/upload.png)

### 2️⃣ Processed Image/Video Output

![Processed](assets/video_result.png)

### 3️⃣ Violation Records

![Records](assets/records.png)

### 4️⃣ Dashboard

![Dashboard](assets/dashboard.png)

---

# 📂 5. Project Structure

```
app/
│── app.py
│── bytetrack.yaml
│── model/
│   ├── DenDoV11_V3.pt
│   ├── nohelmet_V11.pt
│   └── Bienso_V11.pt
│── static/
│   ├── uploads/
│   ├── evidence/
│   ├── video_out/
│   └── favicon.ico
│── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── records.html
│   └── index.html
│── outputs/
requirements.txt
README.md
.gitignore
```

---

# ⚙️ 6. Installation

### Step 1 — Clone project

```bash
git clone https://github.com/<username>/smart-traffic-violation.git
cd smart-traffic-violation
```

### Step 2 — Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 7. Download Model Weights

Model weights are **not included** due to GitHub 100MB file limits.
Download from HuggingFace:

| Task                    | Model             | Download Link                                                                                                                                                      |
| ----------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| No Helmet Detection     | `nohelmet_V11.pt` | [https://huggingface.co/LePhuocThai003/nohelmet_V11/resolve/main/nohelmet_V11.pt](https://huggingface.co/LePhuocThai003/nohelmet_V11/resolve/main/nohelmet_V11.pt) |
| Red-Light / Stop-Line   | `DenDoV11_V3.pt`  | [https://huggingface.co/LePhuocThai003/DenDo_V11/resolve/main/DenDoV11_V3.pt](https://huggingface.co/LePhuocThai003/DenDo_V11/resolve/main/DenDoV11_V3.pt)         |
| License Plate Detection | `Bienso_V11.pt`   | [https://huggingface.co/LePhuocThai003/BienSo_V11/resolve/main/Bienso_V11.pt](https://huggingface.co/LePhuocThai003/BienSo_V11/resolve/main/Bienso_V11.pt)         |

Move them into:

```
app/model/
```

---

# ▶️ 8. Run the Application

```bash
cd app
python app.py
```

The application will start at:

```
http://127.0.0.1:5000/
```

---

# 🔌 9. API Endpoints

| Endpoint        | Method | Description                  |
| --------------- | ------ | ---------------------------- |
| `/detect_image` | POST   | Detect violations from image |
| `/detect_video` | POST   | Detect + track in video      |
| `/records`      | GET    | Retrieve saved records       |
| `/update_plate` | POST   | Update license plate         |
| `/approve`      | POST   | Approve a violation          |
| `/reject`       | POST   | Reject a violation           |

---

# 🗄️ 10. Database Schema

```
violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    type TEXT,
    vehicle TEXT,
    plate TEXT,
    frame INTEGER,
    evidence_path TEXT,
    status TEXT
)
```

---

# 📜 11. License

This project is distributed under the **MIT License**.
