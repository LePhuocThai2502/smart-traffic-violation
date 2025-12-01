# Application_last_version.py — Smart Traffic Violation Backend (Win + Py3.12)
# M1: red_light/stop_line/vehicle   M2: helmet/mobile/triple   M3: plate

import os, re, cv2, csv, time, sqlite3, datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import deque

from flask import Flask, request, render_template, send_from_directory, jsonify, send_file
from ultralytics import YOLO
import easyocr, torch
from paddleocr import PaddleOCR
from ultralytics import YOLO
import ultralytics, os
import subprocess, shutil  # <<< THÊM DÒNG NÀY

# DEBUG: in ra thông tin ultralytics và default.yaml
print("[DEBUG] ultralytics version:", ultralytics.__version__)
print("[DEBUG] ultralytics module:", ultralytics.__file__)

ultra_dir = os.path.dirname(ultralytics.__file__)
default_yaml_path = os.path.join(ultra_dir, "cfg", "default.yaml")
print("[DEBUG] default.yaml path:", default_yaml_path)

# ===== settings =====
try:
    from ultralytics.utils import SETTINGS
    SETTINGS.update({'fuse': False})
except Exception:
    pass

try:
    torch.set_num_threads(1)
except Exception:
    pass

APP_ROOT   = Path(__file__).parent.resolve()
STATIC_DIR = APP_ROOT / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
VIDEO_OUT  = STATIC_DIR / "video_out"
TEMPLATES  = APP_ROOT / "templates"
MODEL_DIR  = APP_ROOT / "model"

for p in (STATIC_DIR, UPLOAD_DIR, VIDEO_OUT, TEMPLATES, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

DB_PATH    = APP_ROOT / "violations.db"
BYTE_YAML  = APP_ROOT / "bytetrack.yaml"
if not BYTE_YAML.exists():
    BYTE_YAML.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.25\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.25\n"
        "track_buffer: 30\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n",
        encoding="utf-8"
    )


# ===== weights =====
M1_PATH = MODEL_DIR / "DenDoV11_V3.pt"
M2_PATH = MODEL_DIR / "nohelmet_V11.pt"
M3_PATH = MODEL_DIR / "Bienso_V11.pt"

for p in (M1_PATH, M2_PATH, M3_PATH):
    if not p.exists():
        raise FileNotFoundError(f"Weight not found: {p}")

# ===== runtime =====
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
torch.set_grad_enabled(False)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ===== config (H2 / R2 – strict) =====
IMG_SIZE            = 1024
CONF_M1, NMS_IOU_M1 = 0.04, 0.65

# M2:
# - CONF_M2: dùng cho ẢNH (analyze_image) → giữ thấp hơn để ảnh nhạy.
# - CONF_M2_VIDEO: dùng cho VIDEO (analyze_video) → cao hơn để giảm bắt nhầm không mũ.
CONF_M2, NMS_IOU_M2       = 0.08, 0.55          # ảnh
CONF_M2_VIDEO             = float(os.getenv("CONF_M2_VIDEO", "0.16"))  # video
CONF_M3, NMS_IOU_M3       = 0.35, 0.55
# NGƯỠNG RIÊNG cho box no_helmet trong VIDEO (lọc bớt nhiễu, không ảnh hưởng ảnh)
NO_HELMET_CONF_VIDEO      = float(os.getenv("NO_HELMET_CONF_VIDEO", "0.40"))

# ===== fallback stopline cho VIDEO =====
FALLBACK_STOPLINE_Y_FRAC  = float(os.getenv("FALLBACK_STOPLINE_Y_FRAC", "0.72"))
FALLBACK_STOPLINE_X1_FRAC = float(os.getenv("FALLBACK_STOPLINE_X1_FRAC", "0.12"))
FALLBACK_STOPLINE_X2_FRAC = float(os.getenv("FALLBACK_STOPLINE_X2_FRAC", "0.88"))

# ghép M2 vào xe: IOU phải cao hơn
IOU_MATCH_TH     = 0.20
OCR_INTERVAL     = 5
PLATE_STABLE_MIN = 2
VOTE_M2_CONSEC   = 4
VOTE_RED_CONSEC  = 2
COOLDOWN_S       = 12
DEDUP_WINDOW_S   = 30
RECENT_KEY_CACHE: Dict[str, float] = {}
SL_HISTORY       = deque(maxlen=20)

# HSV red-light
RED_S_MIN      = int(os.getenv("RED_S_MIN", "40"))
RED_V_MIN      = int(os.getenv("RED_V_MIN", "65"))
RED_RATIO_MIN  = 0.007  # dùng cho video

# ẢNH TĨNH: HSV dùng ngưỡng riêng, ROI hẹp bên phải phía trên stop-line
RED_RATIO_MIN_IMAGE = float(os.getenv("RED_RATIO_MIN_IMAGE", "0.0008"))

CAMERA_ID    = os.getenv("CAMERA_ID", "CAM01")
MIN_VEH_FRAC = 0.06

# bật debug video mặc định = 1 (vạch, box, HSV,…)
VIDEO_DEBUG  = os.getenv("VIDEO_DEBUG", "1") == "1"

# stop-line cố định cho VIDEO (kéo ngang cả khung)
FIXED_STOPLINE_VIDEO   = os.getenv("FIXED_STOPLINE_VIDEO", "1") == "1"
FIXED_STOPLINE_Y_FRAC  = float(os.getenv("FIXED_STOPLINE_Y_FRAC", str(FALLBACK_STOPLINE_Y_FRAC)))
FIXED_STOPLINE_X1_FRAC = float(os.getenv("FIXED_STOPLINE_X1_FRAC", "0.0"))
FIXED_STOPLINE_X2_FRAC = float(os.getenv("FIXED_STOPLINE_X2_FRAC", "1.0"))

# ===== guard line config =====
JUDGE_OFFSET_FRAC = float(os.getenv("JUDGE_FRAC", "0.035"))

JUDGE_MIN_PIX     = int(os.getenv("JUDGE_MIN", "12"))

# ===== pedestrian line config =====
PED_DELTA_FRAC = float(os.getenv("PED_FRAC", "0.04"))
PED_MIN_PIX    = int(os.getenv("PED_MIN", "10"))

# ===== vùng xét stop-line (lọc xe lề / xe giữa ngã tư)
LANE_INNER_LEFT_FRAC  = float(os.getenv("LANE_INNER_LEFT", "0.07"))
LANE_INNER_RIGHT_FRAC = float(os.getenv("LANE_INNER_RIGHT", "0.09"))
STOP_BAND_UP_FRAC     = float(os.getenv("STOP_BAND_UP", "0.06"))
STOP_BAND_DOWN_FRAC   = float(os.getenv("STOP_BAND_DOWN", "0.11"))

# ===== PLATE OCR (PaddleOCR) =====
PLATE_OCR_DEBUG = os.getenv("PLATE_OCR_DEBUG", "0") == "1"
PLATE_OCR_FORCE_CPU = os.getenv("PLATE_OCR_FORCE_CPU", "1") == "1"


def _plate_log(msg: str) -> None:
    if PLATE_OCR_DEBUG:
        print(f"[PLATE_OCR] {msg}")


def create_plate_ocr() -> PaddleOCR:
    """
    Khởi tạo PaddleOCR dùng chung cho cả app.

    - Mặc định ép chạy CPU nếu PLATE_OCR_FORCE_CPU=1 để tránh lỗi cudnn.
    - GIỮ det=True (mặc định) để Paddle tự detect nhiều dòng trên biển số.
    """
    if PLATE_OCR_FORCE_CPU:
        use_gpu = False
    else:
        use_gpu = torch.cuda.is_available()

    if use_gpu:
        try:
            _plate_log("Init PaddleOCR (GPU, det+rec)...")
            ocr = PaddleOCR(
                lang="en",
                use_gpu=True,
                show_log=PLATE_OCR_DEBUG,
                use_angle_cls=False,
            )
            _plate_log("PaddleOCR init OK (GPU)")
            return ocr
        except Exception as e:
            _plate_log(f"GPU init failed: {e} -> fallback CPU")

    _plate_log("Init PaddleOCR (CPU, det+rec)...")
    ocr = PaddleOCR(
        lang="en",
        use_gpu=False,
        show_log=PLATE_OCR_DEBUG,
        use_angle_cls=False,
    )
    _plate_log("PaddleOCR init OK (CPU)")
    return ocr


def _run_ocr(ocr_engine: Any, plate_bgr: np.ndarray):
    """
    Gọi OCR với PaddleOCR cho 1 ROI biển số.
    """
    try:
        if plate_bgr is None or plate_bgr.size == 0:
            _plate_log("empty plate roi in _run_ocr")
            return None

        img = plate_bgr
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        img = np.ascontiguousarray(img)

        res = ocr_engine.ocr(img, cls=False)
        _plate_log(f"raw PaddleOCR result: {res}")
        return res

    except Exception as e:
        _plate_log(f"ocr() Exception: {e}")
        return None


def read_plate_from_roi(
    plate_bgr: np.ndarray,
    ocr_engine: PaddleOCR,
) -> str:
    """
    Nhận ROI BGR của biển số và trả ra chuỗi text thô từ PaddleOCR.
    Sắp xếp theo toạ độ Y để giữ đúng thứ tự dòng.
    """
    if plate_bgr is None or plate_bgr.size == 0:
        _plate_log("empty plate roi")
        return ""

    res = _run_ocr(ocr_engine, plate_bgr)
    if not res:
        _plate_log(f"OCR result is empty/None: {res}")
        return ""

    try:
        items = []  # (y_center, text)

        def collect_items(obj):
            if (
                isinstance(obj, (list, tuple))
                and len(obj) >= 2
                and isinstance(obj[1], (list, tuple))
                and len(obj[1]) >= 1
                and isinstance(obj[1][0], str)
            ):
                text = str(obj[1][0])
                box = obj[0]
                yc = 0.0
                try:
                    pts = np.array(box).reshape(-1, 2)
                    yc = float(pts[:, 1].mean())
                except Exception:
                    pass
                items.append((yc, text))
                return

            if isinstance(obj, (list, tuple)):
                for sub in obj:
                    collect_items(sub)

        collect_items(res)

        if items:
            items.sort(key=lambda t: t[0])
            raw = " ".join(t for _, t in items if t).strip()
            _plate_log(f"raw ocr(sorted)='{raw}' from items={items}")
            return raw

    except Exception as e:
        _plate_log(f"structured parse error: {e}, fallback to flat parse")

    texts = []
    try:
        stack = [res]
        while stack:
            item = stack.pop()

            if isinstance(item, (list, tuple)):
                if (
                    len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                    and len(item[1]) >= 1
                    and isinstance(item[1][0], str)
                ):
                    texts.append(str(item[1][0]))
                    continue

                if len(item) >= 1 and isinstance(item[0], str):
                    texts.append(str(item[0]))
                    continue

                for sub in item:
                    stack.append(sub)

        raw = " ".join(t for t in texts if t).strip()
        _plate_log(f"raw ocr(flatten)='{raw}' from texts={texts}")
        return raw
    except Exception as e:
        _plate_log(f"parse OCR result error (flat): {e} | res={res}")
        return ""


def format_plate_text(raw: str, vehicle_label: Optional[str] = None) -> str:
    """
    Chuẩn hoá biển số theo quy tắc VN.
    """

    if not raw:
        return ""

    s0 = re.sub(r"[^A-Za-z0-9]", "", raw.upper())
    if len(s0) < 5:
        return ""

    chars = list(s0)

    DIGIT_FIX = {
        "O": "0", "Q": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }
    LETTER_FIX = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
    }

    def to_digit(c):
        if c.isdigit():
            return c
        return DIGIT_FIX.get(c, "0")

    def to_letter(c):
        if c.isalpha():
            return c
        return LETTER_FIX.get(c, "A")

    # 2 ký tự đầu luôn là số
    for i in range(min(2, len(chars))):
        chars[i] = to_digit(chars[i])

    # ký tự thứ 3 luôn là chữ
    if len(chars) >= 3:
        chars[2] = to_letter(chars[2])

    s = "".join(chars)

    # ====== XÁC ĐỊNH XE MÁY ======
    is_bike = False
    if vehicle_label:
        v = vehicle_label.lower()
        if any(k in v for k in ["motor", "bike", "bicycle"]):
            is_bike = True

    # ====== REGEX THEO FORMAT ======
    if is_bike:
        m = re.search(r"^(\d{2})([A-Z])([A-Z0-9])(\d{4,5})$", s)
        if not m:
            return ""
        prov, c3, c4, tail = m.groups()
        tail = tail[-5:]
        series = c3 + c4
        return f"{prov}{series}-{tail}"

    else:
        m = re.search(r"^(\d{2})([A-Z])(\d{4,5})$", s)
        if m:
            prov, c3, tail = m.groups()
            return f"{prov}{c3}-{tail[-5:]}"

        if len(s) >= 3:
            s = s[:2] + to_letter(s[2]) + s[3:]
        m = re.search(r"^(\d{2})([A-Z])(\d{4,5})$", s)
        if not m:
            return ""
        prov, c3, tail = m.groups()
        return f"{prov}{c3}-{tail[-5:]}"


def _norm(p: Path) -> str:
    return str(p.as_posix())


def _now(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)


def _fname(prefix: str, ext: str = ".jpg") -> str:
    return f"{prefix}_{int(time.time() * 1000)}{ext}"


# ===== app =====
app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATIC_DIR))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR.resolve())
print(f"[INFO] Device: {DEVICE}, FP16: {USE_FP16}")

# models
m1 = YOLO(_norm(M1_PATH))
m2 = YOLO(_norm(M2_PATH))
m3 = YOLO(_norm(M3_PATH))

# EasyOCR nếu bạn còn muốn dùng
ocr = easyocr.Reader(['en'])

# PaddleOCR chuyên cho biển số
plate_ocr_engine = create_plate_ocr()

# ===== labels =====
ALIAS = {
    "motor_scooter": "motorcycle",
    "red": "red_light",
    "stopline": "stop_line",
    "nohelmet": "no_helmet",
    "trafficlight_red": "red_light",
    "traffic_light_red": "red_light",
    "signal_red": "red_light",
    "redsignal": "red_light",
    "red_light_signal": "red_light",
    "traffic_signal_red": "red_light",
    "redsignal_light": "red_light",
    "traffic_red": "red_light",
    "using_mobile": "mobile_usage",
}

def normalize_label(s: str) -> str:
    s = str(s).strip().lower().replace("&", "_").replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return ALIAS.get(s, s)

VEHICLE_LABELS = {
    "car", "motorbike", "motorcycle", "bike", "bicycle",
    "bus", "truck", "ambulance", "vehicle", "vehicle_with_offence"
}
STOP_LINE_LABELS = {"stop_line"}
RED_LIGHT_LABELS = {"red_light"}
VIOLATION_LABELS = {
    "mobile_usage", "pillion_rider_not_wearing_helmet",
    "rider_and_pillion_not_wearing_helmet",
    "rider_not_wearing_helmet", "triple_riding", "no_helmet",
    "rider_helmet_invalid",
}
HELMET_LABELS = {
    "rider_not_wearing_helmet", "no_helmet", "rider_helmet_invalid",
    "pillion_rider_not_wearing_helmet", "rider_and_pillion_not_wearing_helmet",
}

def normalize_violation_labels(vios: set) -> set:
    """
    Gom pedestrian_line_violation -> red_light_violation.
    Nếu 1 xe dính cả 2 thì set vẫn chỉ còn 1 label red_light_violation.
    """
    out = set()
    for v in (vios or []):
        if v == "pedestrian_line_violation":
            out.add("red_light_violation")
        else:
            out.add(v)
    return out


def role_of(label: str) -> str:
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in STOP_LINE_LABELS:
        return "stop_line"
    if label in RED_LIGHT_LABELS:
        return "red_light"
    if label in VIOLATION_LABELS:
        return "violation"
    return "other"

# ===== db =====
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
c = conn.cursor()
c.execute(
    """
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT,
        source TEXT,
        frame_index INTEGER,
        timestamp TEXT,
        camera_id TEXT,
        vehicle_type TEXT,
        violation_types TEXT,
        license_plate TEXT,
        evidence_path TEXT,
        status TEXT DEFAULT 'pending'
    )
    """
)
c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON records(timestamp)")
c.execute("CREATE INDEX IF NOT EXISTS idx_lp ON records(license_plate)")
conn.commit()

# Migration: nếu DB cũ chưa có cột status thì ADD COLUMN
try:
    c.execute("ALTER TABLE records ADD COLUMN status TEXT DEFAULT 'pending'")
    conn.commit()
    print("[DB] Added status column to records")
except sqlite3.OperationalError:
    # cột đã tồn tại -> bỏ qua
    pass


def insert_record(
    kind: str, source: str, frame_index: Optional[int],
    vehicle_type: str, violation_types: str, plate: str, evidence_file_name: str,
):
    plate_norm = (plate or "").strip().upper()
    if not plate_norm:
        plate_norm = "UNKNOWN"

    c.execute(
        """
        INSERT INTO records(kind,source,frame_index,timestamp,camera_id,
                            vehicle_type,violation_types,license_plate,evidence_path)
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (kind, source, frame_index, _now(), CAMERA_ID,
         vehicle_type, violation_types, plate_norm, evidence_file_name),
    )
    conn.commit()
    return c.lastrowid


# ===== helpers =====
def compute_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    a1 = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    a2 = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (a1 + a2 - inter + 1e-6)


def center_in(inner, outer) -> bool:
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (X1 <= cx <= X2) and (Y1 <= cy <= Y2)



# ===== OCR biển số dùng PaddleOCR =====
def ocr_plate(roi_bgr: np.ndarray) -> str:
    if roi_bgr is None or roi_bgr.size == 0:
        return ""
    roi_bgr = enhance_lowlight(roi_bgr)
    raw = read_plate_from_roi(roi_bgr, plate_ocr_engine)
    if not raw:
        return ""
    plate = format_plate_text(raw)
    return plate or ""


def ocr_plate_multi(roi: np.ndarray) -> str:
    if roi is None or roi.size == 0:
        return ""

    candidates = []
    for s in (1.0, 1.5, 2.0):
        rsz = cv2.resize(roi, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        txt = ocr_plate(rsz)
        if txt:
            candidates.append(txt)

    if not candidates:
        return ""

    return max(set(candidates), key=candidates.count)


def enhance_lowlight(bgr):
    try:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if g.mean() < 70:
            gamma = 1.6
            table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
            bgr = cv2.LUT(bgr, table)
            yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return bgr
    except Exception:
        pass
    return bgr

def detect_red_signal_hsv(frame, stop_line_box, annotate=False):
    """
    VIDEO:
    - Bỏ phụ thuộc stop_line_box cho ROI đèn.
    - Quét cố định vùng GÓC TRÊN-PHẢI, nơi có cụm đèn giao thông DTV.
    """
    H, W = frame.shape[:2]

    # ROI góc trên-phải: 40% phải, 45% trên
    rx1 = int(0.60 * W)
    ry1 = 0
    rx2 = W
    ry2 = int(0.45 * H)

    if rx2 - rx1 < 10 or ry2 - ry1 < 10:
        return False, None, 0.0

    roi = frame[ry1:ry2, rx1:rx2]
    roi = enhance_lowlight(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # mask đỏ – 2 dải HUE
    lower1 = np.array([0,   RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper1 = np.array([10,  255,       255],       dtype=np.uint8)
    lower2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper2 = np.array([180, 255,       255],       dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # lọc nhiễu
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)

    red_ratio = mask.mean() / 255.0

    # nhạy hơn chút so với RED_RATIO_MIN mặc định
    is_red = red_ratio >= max(RED_RATIO_MIN * 0.8, 0.004)

    # lấy blob đỏ lớn nhất (nếu có) để vẽ box
    box = None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        gx1 = rx1 + x
        gy1 = ry1 + y
        gx2 = gx1 + w
        gy2 = gy1 + h
        box = [gx1, gy1, gx2, gy2]

    if annotate or VIDEO_DEBUG:
        # vẽ ROI HSV
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(
            frame,
            f"HSV-red:{red_ratio:.3f}",
            (rx1 + 5, ry1 + 18),
            0,
            0.6,
            (0, 255, 255),
            2,
        )
        # vẽ box đèn (nếu có)
        if box is not None:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    return is_red, box, red_ratio



# ===== runners =====
def _collect_dets(res, names_source):
    dets = []
    for r in res or []:
        if not getattr(r, "boxes", None):
            continue
        for box, cid in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int)
        ):
            label = normalize_label(
                (r.names if hasattr(r, "names") else names_source)[int(cid)]
            )
            dets.append((list(map(int, box.tolist())), label))
    return dets

def run_m1(bgr: np.ndarray, conf_override: Optional[float] = None):
    conf = CONF_M1 if conf_override is None else conf_override
    res = m1.predict(
        bgr, imgsz=IMG_SIZE, conf=conf, iou=NMS_IOU_M1,
        verbose=False, device=DEVICE, half=USE_FP16, workers=0,
    )
    return _collect_dets(res, m1.names)

def run_m1_light_stop(bgr: np.ndarray):
    return run_m1(bgr, conf_override=max(0.02, CONF_M1 * 0.6))

def run_m2(bgr: np.ndarray, conf_override: Optional[float] = None):
    conf = CONF_M2 if conf_override is None else conf_override
    res = m2.predict(
        bgr,
        imgsz=IMG_SIZE,
        conf=conf,
        iou=NMS_IOU_M2,
        verbose=False,
        device=DEVICE,
        half=USE_FP16,
        workers=0,
    )
    return _collect_dets(res, m2.names)

def run_m3(bgr: np.ndarray):
    res = m3.predict(
        bgr, imgsz=IMG_SIZE, conf=CONF_M3, iou=NMS_IOU_M3,
        verbose=False, device=DEVICE, half=USE_FP16,
        workers=0,
    )
    return _collect_dets(res, m3.names)

TWO_WHEELS = {"motorbike", "motorcycle", "bike", "bicycle"}

# ===== crossed_* functions, in_stop_region, red_violation_image_simple, compute_lane_front_flags,
# suppress_by_depth, assign_violations_to_vehicles,
# detect_red_signal_hsv_image, analyze_image, analyze_video,
# routes /, /records, /dashboard, /detect_image, /detect_video, /export_csv, /dashboard_data
# =====

# Do phần dưới bạn đã dán nguyên trong Application.py ở tin nhắn trước,
# giữ nguyên y chang (từ crossed_stopline(...) trở xuống không đổi).

# Để tránh vượt giới hạn tin nhắn ở đây, chỉ cần:
# 1. Copy từ hàm `crossed_stopline` trong file Application.py bạn vừa gửi
# 2. Dán toàn bộ phần còn lại vào đây ngay dưới comment này
# 3. Giữ nguyên đoạn `if __name__ == "__main__": ...` ở cuối

# ================== copy từ crossed_stopline(...) trở xuống ==================
# (toàn bộ đoạn bạn gửi ở trên, bắt đầu từ:
# def crossed_stopline(...):
# ...
# tới cuối file: if __name__ == "__main__": ...)

def crossed_stopline(
    vbox: List[int],
    sl_box: List[int],
    W: int,
    H: int,
    vehicle_label: str,
    judge_y: Optional[int] = None,
    mode: str = "video",  # "video" (mặc định), "image" (ảnh tĩnh)
) -> bool:
    """
    - VIDEO: chặt nhưng đã nới nhẹ cho xe 4 bánh.
    - IMAGE: nới lỏng hơn để các xe vừa đè / vừa qua vạch được bắt đúng.
    """
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = sl_box

    # Tính judge_y nếu chưa có
    if judge_y is None:
        offset = max(JUDGE_MIN_PIX, int(JUDGE_OFFSET_FRAC * H))
        judge_y = max(0, sy2 - offset)

    is_image = (mode == "image")

    # --------- điều kiện cơ bản theo chiều dọc ----------
    if is_image:
        # Ảnh tĩnh: chỉ cần đáy xe thấp hơn judge_y một ít
        if vy2 < judge_y + max(3, int(0.005 * H)):
            return False
    else:
        # Video: yêu cầu thân xe cắt rõ qua vùng quanh judge_y
        if vy1 > judge_y:
            return False
        if vy2 < judge_y + max(5, int(0.015 * H)):
            return False

    # --------- overlap ngang với stop-line ----------
    vw = max(1, vx2 - vx1)
    sw = max(1, sx2 - sx1)
    overlap = max(0, min(vx2, sx2) - max(vx1, sx1))
    frac_v = overlap / vw
    frac_s = overlap / sw

    # xe phải nằm trong vùng làn của stop-line
    cx = 0.5 * (vx1 + vx2)
    lane_pad = max(int(0.03 * W), 10)
    if not (sx1 - lane_pad <= cx <= sx2 + lane_pad):
        return False

    # --------- mức độ "đè vạch" theo chiều dọc ----------
    h = max(1, vy2 - vy1)
    penetrate_from_top = max(0, judge_y - vy1)
    pen_ratio_from_top = max(0.0, min(1.0, penetrate_from_top / h))
    penetrate_from_bottom = max(0, vy2 - judge_y)
    pen_ratio_from_bottom = max(0.0, min(1.0, penetrate_from_bottom / h))

    if vehicle_label in TWO_WHEELS:
        # NỚI LỎNG cho xe 2 bánh ở chế độ ảnh:
        if is_image:
            need_vx, need_sx = 0.16, 0.08   # trước là 0.18 / 0.10
            need_pen_top, need_pen_bottom = 0.03, 0.02
        else:
            need_vx, need_sx = 0.22, 0.12
            need_pen_top, need_pen_bottom = 0.08, 0.05
            
    else:
        # Xe 4 bánh trở lên:
        if is_image:
            # ẢNH: giữ như cũ, hơi chặt để ảnh không bắt nhầm
            need_vx, need_sx = 0.26, 0.12
            need_pen_top, need_pen_bottom = 0.05, 0.01
        else:
            # VIDEO: nới thêm cho xe 4 bánh
            # yêu cầu cắt vạch ít hơn một chút nhưng vẫn phải qua rõ ràng
            need_vx, need_sx = 0.26, 0.10   # trước 0.30 / 0.12
            need_pen_top, need_pen_bottom = 0.05, 0.01  # trước 0.07 / 0.01
  

    span_ok = frac_v >= need_vx and frac_s >= need_sx
    return span_ok and (pen_ratio_from_top >= need_pen_top) and (pen_ratio_from_bottom >= need_pen_bottom)





def crossed_pedline(
    vbox: List[int],
    ped_y: int,
    W: int,
    H: int,
    vehicle_label: str,
    mode: str = "video",
) -> bool:
    """
    - VIDEO: yêu cầu thân xe cắt rõ qua vạch người đi bộ (chặt hơn).
    - IMAGE: nới nhẹ nhưng vẫn giới hạn chỉ quanh vùng giao lộ
      để tránh bắt xe quá xa hoặc giữa ngã tư.
    """
    vx1, vy1, vx2, vy2 = vbox
    is_image = (mode == "image")

    # Chỉ xét xe nằm trong một dải dọc quanh vạch người đi bộ
    band_top = ped_y - max(int(0.12 * H), 35)
    band_bot = ped_y + max(int(0.18 * H), 35)
    if vy2 < band_top or vy1 > band_bot:
        return False

    # Điều kiện cơ bản theo chiều dọc
    if is_image:
        # Ảnh tĩnh: đáy xe phải thấp hơn ped_y một chút
        if vy2 < ped_y:
            return False
    else:
        # Video: đỉnh xe phải nằm phía trên ped_y
        if vy1 > ped_y:
            return False

    # Mức độ "đè" vạch theo chiều dọc
    h = max(1, vy2 - vy1)
    overlap = max(0, min(vy2, ped_y) - max(vy1, ped_y - h))
    pen_ratio = overlap / h

    if vehicle_label in TWO_WHEELS:
        need_pen = 0.15 if is_image else 0.18
    else:
        need_pen = 0.20 if is_image else 0.22

    return pen_ratio >= need_pen

def crossed_or_beyond_stopline_image(
    vbox: List[int],
    sl_box: List[int],
    W: int,
    H: int,
    vehicle_label: str,
    judge_y: int,
) -> bool:
    """
    ẢNH TĨNH:
    - Nếu xe đã nằm hoàn toàn phía sau vạch (gần camera hơn judge_y) trong một
      "dải" quanh stop-line => coi như đã vượt vạch.
    - Nếu chưa hoàn toàn phía sau thì dùng lại crossed_stopline(mode='image')
      để bắt các xe đang cắt qua vạch.
    """
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = sl_box

    # Giữ điều kiện xe nằm trong làn của stop-line
    cx = 0.5 * (vx1 + vx2)
    lane_pad = max(int(0.03 * W), 10)
    if not (sx1 - lane_pad <= cx <= sx2 + lane_pad):
        return False

    # Chỉ xét các xe nằm trong dải dọc quanh stop-line (tránh xe rất xa / giữa ngã tư)
    band_top = sy2 - max(int(0.18 * H), 40)
    band_bot = sy2 + max(int(0.22 * H), 40)
    if vy2 < band_top or vy1 > band_bot:
        return False

    # Trường hợp 1: xe đã nằm hoàn toàn sau judge_y (gần camera hơn) → đã vượt vạch
    if vy1 >= judge_y:
        return True

    # Trường hợp 2: xe đang cắt qua vạch → dùng lại logic chặt hơn
    return crossed_stopline(
        vbox, sl_box, W, H, vehicle_label,
        judge_y=judge_y, mode="image"
    )



def in_stop_region(vbox: List[int], sl_box: Optional[List[int]], W: int, H: int) -> bool:
    if sl_box is None:
        return True
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = sl_box
    if sx2 <= sx1:
        return True
    width = sx2 - sx1
    inner_x1 = sx1 + int(width * LANE_INNER_LEFT_FRAC)
    inner_x2 = sx2 - int(width * LANE_INNER_RIGHT_FRAC)
    cx = 0.5 * (vx1 + vx2)
    if cx < inner_x1 or cx > inner_x2:
        return False
    band_up = max(int(STOP_BAND_UP_FRAC * H), 20)
    band_down = max(int(STOP_BAND_DOWN_FRAC * H), 24)
    band_top = sy2 - band_up
    band_bot = sy2 + band_down
    return band_top <= vy2 <= band_bot


def red_violation_image_simple(vbox: List[int],
                               stop_line_box: List[int],
                               Wf: int,
                               Hf: int) -> bool:
    """
    Ảnh tĩnh:
    - Đèn đang đỏ.
    - Xe nào TRÊN / QUA vạch dừng (hướng vào giao lộ) thì vi phạm.
    - Xe còn ở phía dưới vạch (gần camera) thì không vi phạm.
    """
    vx1, vy1, vx2, vy2 = vbox
    sx1, sy1, sx2, sy2 = stop_line_box

    # Tâm xe
    cx = (vx1 + vx2) // 2
    cy = (vy1 + vy2) // 2

    # Bỏ các xe hoàn toàn ngoài vùng ngang stop-line (xe đỗ lề, làn khác)
    margin_x = max(8, int(0.02 * Wf))
    if cx < sx1 - margin_x or cx > sx2 + margin_x:
        return False

    # Chỉ xét các xe nằm trong một dải dọc quanh stop-line
    band_top = sy2 - max(int(0.20 * Hf), 40)
    band_bot = sy2 + max(int(0.18 * Hf), 40)
    if vy2 < band_top or vy1 > band_bot:
        return False

    # Guard line nằm hơi phía trước stop-line
    offset = max(JUDGE_MIN_PIX, int(JUDGE_OFFSET_FRAC * Hf))
    judge_y = max(0, sy2 - offset)

    # Ngưỡng nhiễu theo chiều dọc
    tol = max(3, int(0.004 * Hf))

    # TH1: toàn bộ xe đã nằm phía trong giao lộ (đáy xe cao hơn stop-line)
    if vy2 <= sy2 - tol:
        return True

    # TH2: xe đang đè vạch: thân xe cắt qua stop-line, đầu xe đã qua guard line
    if (vy1 <= sy2 + tol) and (vy1 <= judge_y) and (vy2 >= sy2 - tol):
        return True

    return False







# ===== lane front selection (VIDEO / IMAGE) =====
def compute_lane_front_flags(
    vehicles: List[Tuple[List[int], str]],
    judge_y: Optional[int],
    stop_line_box: Optional[List[int]],
    H: int,
    same_lane_x_iou: float = 0.45,
) -> List[bool]:
    n = len(vehicles)
    if n == 0 or judge_y is None or stop_line_box is None:
        return [True] * n

    sx1, sy1, sx2, sy2 = stop_line_box
    flags = [False] * n
    clusters: List[List[int]] = []

    for i, (vbox, _) in enumerate(vehicles):
        vx1, vy1, vx2, vy2 = vbox
        overlap_sl = max(0, min(vx2, sx2) - max(vx1, sx1))
        if overlap_sl <= 0:
            continue
        assigned = False
        for cl in clusters:
            j0 = cl[0]
            vbox0, _ = vehicles[j0]
            x1, y1, x2, y2 = vbox0
            inter_x = max(0, min(vx2, x2) - max(vx1, x1))
            union_x = (vx2 - vx1) + (x2 - x1) - inter_x + 1e-6
            if inter_x / union_x >= same_lane_x_iou:
                cl.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    for cl in clusters:
        best_idx = None
        best_dist = 1e9
        for idx in cl:
            vbox, _ = vehicles[idx]
            vx1, vy1, vx2, vy2 = vbox
            cy = vy2
            dist = abs(cy - judge_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            flags[best_idx] = True

    if any(flags) or n == 0:
        return flags

    best_idx, best_dist = 0, 1e9
    for i, (vbox, _) in enumerate(vehicles):
        vx1, vy1, vx2, vy2 = vbox
        cy = vy2
        dist = abs(cy - judge_y)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    flags[best_idx] = True
    return flags

def suppress_by_depth(
    cands: List[Tuple[List[int], str, set]],
    H: int, x_overlap_th: float = 0.45, y_margin_frac: float = 0.02,
):
    if len(cands) <= 1:
        return cands
    keep = [True] * len(cands)
    y_margin = int(y_margin_frac * H)
    for i in range(len(cands)):
        if not keep[i]:
            continue
        xi1, yi1, xi2, yi2 = cands[i][0]
        ci = 0.5 * (yi1 + yi2)
        for j in range(i + 1, len(cands)):
            if not keep[j]:
                continue
            xj1, yj1, xj2, yj2 = cands[j][0]
            cj = 0.5 * (yj1 + yj2)
            inter_x = max(0, min(xi2, xj2) - max(xi1, xj1))
            union_x = (xi2 - xi1) + (xj2 - xj1) - inter_x + 1e-6
            x_iou = inter_x / union_x
            if x_iou < x_overlap_th:
                continue
            if ci + y_margin < cj:
                keep[j] = False
            elif cj + y_margin < ci:
                keep[i] = False
            break
    return [c for k, c in enumerate(cands) if keep[k]]

# ===== gán M2 cho xe — H2 (rất chặt) =====
def assign_violations_to_vehicles(vehicles, violation_boxes):
    assigned = [set() for _ in vehicles]
    if not vehicles or not violation_boxes:
        return assigned

    for ibox, ilabel in violation_boxes:
        ix1, iy1, ix2, iy2 = ibox
        icy = 0.5 * (iy1 + iy2)

        candidates = []
        for idx, (vbox, vlabel) in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = vbox
            vcy = 0.5 * (vy1 + vy2)

            # Helmet / mobile chỉ gán cho xe 2 bánh
            if ilabel in HELMET_LABELS or ilabel == "mobile_usage":
                if vlabel not in TWO_WHEELS:
                    continue

            iou_score = compute_iou(vbox, ibox)
            vw = max(1, vx2 - vx1)

            # overlap ngang
            overlap_x = max(0, min(vx2, ix2) - max(vx1, ix1))
            frac_x = overlap_x / vw

            # điều kiện chung: phải cùng làn, không quá lệch ngang
            if frac_x < 0.12 and not (center_in(ibox, vbox) or iou_score > IOU_MATCH_TH):
                continue

            # ==== Lọc riêng cho Helmet ====
            if ilabel in HELMET_LABELS:
                # hộp mũ phải nằm phía trên hoặc quanh tâm xe
                if icy > vcy + 0.10 * (vy2 - vy1):
                    continue

                # head box rất nhỏ: chấp nhận IOU nhỏ nhưng yêu cầu overlap ngang đủ
                if iou_score < 0.03 and frac_x < 0.25:
                    continue

            # ==== Lọc riêng cho dùng điện thoại ====
            if ilabel == "mobile_usage":
                h = max(1, vy2 - vy1)
                # vị trí tương đối của box điện thoại trong chiều cao xe
                rel_y = (icy - vy1) / h

                # chỉ chấp nhận khi box nằm quanh vùng thân trên (tay/cổ):
                # từ 25% đến 75% chiều cao xe
                if rel_y < 0.25 or rel_y > 0.75:
                    continue

                # không nhận các box nằm rõ ràng dưới tâm xe (gần yên/đuôi)
                if icy > vcy + 0.05 * h:
                    continue

                # cần overlap ngang mạnh hơn để tránh dính xe kế bên
                if frac_x < 0.28:
                    continue

                # yêu cầu IOU tối thiểu với xe
                if iou_score < 0.04:
                    continue

            # ưu tiên xe gần head-box nhất
            dist_center = np.hypot(
                0.5 * (vx1 + vx2) - 0.5 * (ix1 + ix2),
                vcy - icy,
            )
            candidates.append((idx, iou_score, dist_center))

        if not candidates:
            continue

        # Ưu tiên IOU cao nhất, sau đó khoảng cách tâm nhỏ nhất
        best_idx = max(candidates, key=lambda t: (t[1], -t[2]))[0]
        assigned[best_idx].add(ilabel)

    return assigned


def detect_red_signal_hsv_image(
    frame: np.ndarray,
    stop_line_box: Optional[List[int]],
    annotate: bool = False,
):
    """
    HSV cho ẢNH TĨNH:
    - Chỉ quét vùng bên phải (x > 0.6W)
    - Chỉ quét phía trên stop-line (nơi đặt đèn)
    - Không quét dưới stop-line (tránh đèn hậu xe, banner, biển báo)
    """
    if stop_line_box is None:
        return False, None, 0.0

    H, W = frame.shape[:2]
    sx1, sy1, sx2, sy2 = stop_line_box

    # ROI theo chiều ngang (phía phải)
    rx1 = int(0.60 * W)
    rx2 = W

    # ROI theo chiều dọc (phía trên stop-line)
    margin = int(0.08 * H)
    ry1 = 0
    ry2 = max(0, sy1 - margin)

    if rx2 <= rx1 or ry2 <= ry1:
        return False, None, 0.0

    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Mask đỏ – 2 vùng hue
    lower1 = np.array([0,   RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper1 = np.array([10,  255,       255],       dtype=np.uint8)
    lower2 = np.array([160, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper2 = np.array([179, 255,       255],       dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_ratio = float(mask.mean() / 255.0)

    # Không đủ đỏ → không có đèn
    if red_ratio < RED_RATIO_MIN_IMAGE:
        return False, None, red_ratio

    # Lấy contour đèn
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, red_ratio

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Loại vật quá bè ngang (banner, biển)
    if w > 3 * h:
        return False, None, red_ratio

    gx1 = rx1 + x
    gy1 = ry1 + y
    gx2 = gx1 + w
    gy2 = gy1 + h
    box = [gx1, gy1, gx2, gy2]

    if annotate:
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"ROI red={red_ratio:.3f}",
            (gx1, max(0, gy1 - 6)),
            0,
            0.5,
            (0, 0, 255),
            1
        )

    return True, box, red_ratio


def analyze_image(frame: np.ndarray) -> Tuple[np.ndarray, str]:
    SL_HISTORY.clear()

    # tăng sáng chung cho frame
    frame = enhance_lowlight(frame)
    annotated = frame.copy()
    Hf, Wf = annotated.shape[:2]

    # ======================== PHA 1: M1 light — tìm stop_line ========================
    targeted = run_m1_light_stop(frame)
    vehicles: List[Tuple[List[int], str]] = []
    sl_boxes_all: List[List[int]] = []
    red_boxes_yolo: List[List[int]] = []

    for box, label in targeted:
        role = role_of(label)
        x1, y1, x2, y2 = box

        if role == "stop_line":
            sl_boxes_all.append(box)
            SL_HISTORY.append(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

        elif role == "red_light":
            red_boxes_yolo.append(box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # ======================== PHA 2: M1 chính ========================
    for box, label in run_m1(frame):
        x1, y1, x2, y2 = box
        role = role_of(label)

        if role == "vehicle":
            if (y2 - y1) / Hf < MIN_VEH_FRAC:
                continue
            vehicles.append((box, label))

        elif role == "stop_line":
            sl_boxes_all.append(box)
            SL_HISTORY.append(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

        elif role == "red_light":
            red_boxes_yolo.append(box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # ======================== STOP-LINE chính ========================
    stop_line_box: Optional[List[int]] = None
    main_sl_boxes: List[List[int]] = []

    if sl_boxes_all:
        sy2_list = [b[3] for b in sl_boxes_all]
        main_sy2 = max(sy2_list)
        band_y = max(int(0.04 * Hf), 12)

        main_sl_boxes = [b for b in sl_boxes_all if abs(b[3] - main_sy2) <= band_y]
        if not main_sl_boxes:
            main_sl_boxes = sl_boxes_all

        sx1 = min(b[0] for b in main_sl_boxes)
        sy1 = min(b[1] for b in main_sl_boxes)
        sx2 = max(b[2] for b in main_sl_boxes)
        sy2 = max(b[3] for b in main_sl_boxes)
        stop_line_box = [sx1, sy1, sx2, sy2]

    def in_our_lanes(vbox: List[int]) -> bool:
        if not main_sl_boxes:
            return True
        vx1, _, vx2, _ = vbox
        cx = 0.5 * (vx1 + vx2)
        margin_x = max(10, int(0.02 * Wf))
        for sx1, sy1, sx2, sy2 in main_sl_boxes:
            if sx1 - margin_x <= cx <= sx2 + margin_x:
                return True
        return False

    # ======================== VALIDATE STOP-LINE ========================
    has_valid_stop_line = False
    if stop_line_box is not None:
        sx1, sy1, sx2, sy2 = stop_line_box
        width_frac = (sx2 - sx1) / max(1, Wf)
        height_frac = (sy2 - sy1) / max(1, Hf)
        if width_frac >= 0.18 and height_frac <= 0.20:
            has_valid_stop_line = True
        else:
            stop_line_box = None
            SL_HISTORY.clear()

    # ======================== judge_y & ped_y ========================
    judge_y = None
    ped_y = None

    if has_valid_stop_line and stop_line_box is not None:
        sx1, sy1, sx2, sy2 = stop_line_box

        if len(SL_HISTORY) > 0:
            sy2 = int(np.median(SL_HISTORY))

        offset = max(JUDGE_MIN_PIX, int(JUDGE_OFFSET_FRAC * Hf))
        judge_y = max(0, sy2 - offset)
        cv2.line(annotated, (sx1, judge_y), (sx2, judge_y), (0, 0, 255), 2)

        ped_offset = max(PED_MIN_PIX, int(PED_DELTA_FRAC * Hf))
        ped_y = max(0, judge_y - ped_offset)
        cv2.line(annotated, (sx1, ped_y), (sx2, ped_y), (0, 165, 255), 1)

    # ======================== PHA 3: RED-LIGHT ========================
    red_light = False
    if has_valid_stop_line and stop_line_box is not None:
        ok_hsv, bb, red_ratio = detect_red_signal_hsv_image(
            annotated, stop_line_box, annotate=True
        )

        has_red_yolo = len(red_boxes_yolo) > 0

        if has_red_yolo and ok_hsv:
            red_light = True
        elif has_red_yolo and red_ratio > 0.0:
            red_light = True
        elif ok_hsv and red_ratio >= max(RED_RATIO_MIN_IMAGE * 1.6, 0.012):
            red_light = True
        else:
            red_light = False

    # ======================== PHA 4: M2 (helmet/mobile) ========================
    violation_boxes = [
        (b, l)
        for (b, l) in run_m2(frame, conf_override=max(0.02, CONF_M2 * 0.8))
        if role_of(l) == "violation"
    ]

    saved = 0
    last_evidence_filename = ""
    violator_types_set: set = set()

    # ======================== PHA 5: GÁN VI PHẠM ========================
    assigned_sets = assign_violations_to_vehicles(vehicles, violation_boxes)
    viol_candidates: List[Tuple[List[int], str, set]] = []

    for (vbox, vlabel), vios_init in zip(vehicles, assigned_sets):
        if not in_our_lanes(vbox):
            continue

        vios = set(vios_init)

        if has_valid_stop_line and stop_line_box is not None and red_light:
            if red_violation_image_simple(vbox, stop_line_box, Wf, Hf):
                vios.add("red_light_violation")

        if (
            has_valid_stop_line
            and ped_y is not None
            and stop_line_box is not None
            and red_light
        ):
            if crossed_pedline(vbox, ped_y, Wf, Hf, vlabel, mode="image"):
                vios.add("pedestrian_line_violation")

        if vios:
            viol_candidates.append((vbox, vlabel, vios))

    # ======================== PHA 6: BIỂN SỐ + OCR + LƯU DB ========================
    for vbox, vlabel, vios in viol_candidates:
        # dùng label normalize cho hiển thị + DB
        norm_vios = normalize_violation_labels(vios)

        vx1, vy1, vx2, vy2 = vbox
        plate_text = ""
        plate_box_global = None
        had_plate_box = False

        # cắt ROI theo xe
        roi = frame[max(0, vy1):max(0, vy2), max(0, vx1):max(0, vx2)]
        if roi.size:
            # chạy M3 trong ROI xe
            p_dets = run_m3(roi)
            if p_dets:
                idx_best, area_best = -1, -1
                for i, (pb, _) in enumerate(p_dets):
                    x1, y1, x2, y2 = pb
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area > area_best:
                        idx_best, area_best = i, area

                if idx_best >= 0:
                    had_plate_box = True
                    px1, py1, px2, py2 = p_dets[idx_best][0]
                    plate_box_global = (
                        int(px1 + vx1),
                        int(py1 + vy1),
                        int(px2 + vx1),
                        int(py2 + vy1),
                    )
                    plate_roi = frame[
                        max(0, plate_box_global[1]):max(0, plate_box_global[3]),
                        max(0, plate_box_global[0]):max(0, plate_box_global[2]),
                    ]

                    raw_lp = read_plate_from_roi(plate_roi, plate_ocr_engine) or ""
                    fmt_lp = format_plate_text(raw_lp, vlabel or "")
                    plate_text = fmt_lp if fmt_lp else raw_lp

        lp_show = ""
        if had_plate_box:
            lp_show = plate_text if plate_text else "UNKNOWN"

        # VẼ XE (tag dùng norm_vios)
        color = (0, 0, 255)
        thick = 3
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), color, thick)

        tag_vios = ",".join(sorted(norm_vios))
        tag = f"{vlabel}|{tag_vios}"
        if lp_show:
            tag += f"|LP:{lp_show}"
        cv2.putText(annotated, tag, (vx1, max(0, vy1 - 8)), 0, 0.6, color, 2)

        # VẼ BIỂN
        if had_plate_box and plate_box_global is not None:
            bx1, by1, bx2, by2 = plate_box_global
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            if lp_show:
                cv2.putText(
                    annotated,
                    lp_show,
                    (bx1, max(0, by1 - 4)),
                    0,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        violator_types_set.add(vlabel)

        # Lưu DB dùng norm_vios
        snap = _fname("image_vio")
        cv2.imwrite(str(UPLOAD_DIR / snap), annotated)
        insert_record(
            "image",
            "upload_image",
            None,
            vlabel,
            ",".join(sorted(norm_vios)),
            lp_show,
            snap,
        )

        last_evidence_filename = snap
        saved += 1

    # ======================== TỔNG KẾ ========================
    if violator_types_set:
        cv2.putText(
            annotated,
            "Violators: " + ",".join(sorted(violator_types_set)),
            (10, 26),
            0,
            0.8,
            (0, 0, 255),
            2,
        )

    if saved == 0:
        snap = _fname("image_safe")
        cv2.imwrite(str(UPLOAD_DIR / snap), annotated)
        insert_record("image", "upload_image", None, "", "", "", snap)
        last_evidence_filename = snap

    return annotated, last_evidence_filename

def transcode_to_h264_mp4(src_mp4: str) -> str:
    """
    Dùng ffmpeg mã hoá lại sang H.264 chuẩn cho browser (Chrome/Edge).
    Nếu không có ffmpeg thì trả luôn file gốc.
    """
    import subprocess
    import shutil
    from pathlib import Path

    ff = shutil.which("ffmpeg")
    if not ff:
        print("[WARN] ffmpeg not found -> browser có thể KHÔNG play được mp4 này.")
        return src_mp4   # fallback: dùng tạm file gốc

    src = Path(src_mp4)
    # đổi tên file: annotated_xxx_fixed.mp4
    out = src.with_name(src.stem + "_fixed.mp4")

    cmd = [
        ff, "-y",
        "-i", str(src),
        "-c:v", "libx264",      # mã hoá lại sang H.264
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",  # format chuẩn cho web
        "-an",                  # không cần audio
        "-movflags", "+faststart",
        str(out),
    ]
    print("[FFMPEG] encode H.264:", " ".join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # nếu output OK → dùng file fixed
    if out.exists() and out.stat().st_size > 0:
        return str(out)

    # nếu encode lỗi → quay lại file gốc
    return src_mp4





def analyze_video(video_path: str) -> Tuple[str, int]:
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap0.get(cv2.CAP_PROP_FPS) or 25
    if fps <= 1 or fps != fps:  # NaN hoặc quá thấp
        fps = 25

    W = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    if W <= 0 or H <= 0:
        raise RuntimeError(f"Invalid video size: {W}x{H}")

    # stop-line cố định cho toàn bộ video (chỉ dùng cho VIDEO)
    if FIXED_STOPLINE_VIDEO:
        sy_fixed = int(FIXED_STOPLINE_Y_FRAC * H)
        sx1_fixed = int(FIXED_STOPLINE_X1_FRAC * W)
        sx2_fixed = int(FIXED_STOPLINE_X2_FRAC * W)
    else:
        sy_fixed = sx1_fixed = sx2_fixed = None

    stem = Path(video_path).stem
    ts = int(time.time())

    # GHI TRỰC TIẾP RA MP4 (mp4v) CHO BROWSER
    out_name = f"annotated_{stem}_{ts}.mp4"
    out_full = str(VIDEO_OUT / out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_full, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Cannot create mp4 video writer (mp4v)")

    stream = m1.track(
        source=video_path, imgsz=IMG_SIZE, conf=CONF_M1, iou=NMS_IOU_M1,
        tracker=str(BYTE_YAML), persist=True, stream=True,
        device=DEVICE, half=USE_FP16, verbose=False,
    )

    tracked: Dict[int, Dict] = {}
    total, frame_idx = 0, 0
    SL_HISTORY.clear()

    for res in stream:
        frame_idx += 1
        frame = enhance_lowlight(res.orig_img.copy())

        stop_line_box, red_light = None, False

        xyxy, cids, ids = [], [], []
        if getattr(res, "boxes", None) and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cids = res.boxes.cls.cpu().numpy().astype(int)
            ids = (
                res.boxes.id.cpu().numpy().astype(int)
                if res.boxes.id is not None else np.full(len(xyxy), -1)
            )

        names = res.names if hasattr(res, "names") else m1.names
        vehicle_boxes_by_tid: List[Tuple[int, List[int]]] = []

        # ====== PHASE 1 — đọc M1 (xe, stop-line, red) ======
        for box, cid, tid in zip(xyxy, cids, ids):
            x1, y1, x2, y2 = map(int, box)
            label = normalize_label(names[int(cid)])
            role = role_of(label)

            if tid != -1 and tid not in tracked:
                tracked[tid] = {
                    "vehicle_type": None,
                    "violations": set(),
                    "votes": {},
                    "plate_votes": {},
                    "plate": None,
                    "last_ocr_frame": -999,
                    "last_save_time": 0.0,
                    "last_plate_box": None,
                }

            if role == "vehicle" and tid != -1:
                if (y2 - y1) / H < MIN_VEH_FRAC:
                    continue
                tracked[tid]["vehicle_type"] = label
                vehicle_boxes_by_tid.append((tid, [x1, y1, x2, y2]))

            elif role == "stop_line" and not FIXED_STOPLINE_VIDEO:
                if (y2 / H) < 0.45:
                    continue
                stop_line_box = [x1, y1, x2, y2]
                SL_HISTORY.append(y2)
                if VIDEO_DEBUG:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            elif role == "red_light":
                red_light = True
                if VIDEO_DEBUG:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ====== PHASE 2 — STOP-LINE FALLBACK / CỐ ĐỊNH ======
        if FIXED_STOPLINE_VIDEO:
            sy_f = sy_fixed
            sx1_f = sx1_fixed
            sx2_f = sx2_fixed
            stop_line_box = [sx1_f, sy_f - 4, sx2_f, sy_f + 4]
            if VIDEO_DEBUG:
                cv2.rectangle(
                    frame,
                    (sx1_f, sy_f - 4),
                    (sx2_f, sy_f + 4),
                    (255, 255, 0), 2,
                )
        else:
            if stop_line_box is None and len(SL_HISTORY) == 0:
                sy_f = int(FALLBACK_STOPLINE_Y_FRAC * H)
                sx1_f = int(FALLBACK_STOPLINE_X1_FRAC * W)
                sx2_f = int(FALLBACK_STOPLINE_X2_FRAC * W)
                stop_line_box = [sx1_f, sy_f - 4, sx2_f, sy_f + 4]
                if VIDEO_DEBUG:
                    cv2.rectangle(
                        frame,
                        (sx1_f, sy_f - 4),
                        (sx2_f, sy_f + 4),
                        (255, 255, 0), 2,
                    )

        # ====== PHASE 3 — judge_y & ped_y ======
        judge_y = None
        ped_y = None

        if FIXED_STOPLINE_VIDEO and stop_line_box is not None:
            sx1, sy1, sx2, sy2 = stop_line_box
        else:
            if stop_line_box is None:
                writer.write(frame)
                continue

            if len(SL_HISTORY) > 0:
                sy2 = int(np.median(SL_HISTORY))
                sx1, sy1, sx2, _ = stop_line_box
                stop_line_box = [sx1, sy2 - 4, sx2, sy2 + 4]
            else:
                sx1, sy1, sx2, sy2 = stop_line_box

        sx1, sy1, sx2, sy2 = stop_line_box

        offset = max(JUDGE_MIN_PIX, int(JUDGE_OFFSET_FRAC * H))
        judge_y = max(0, sy2 - offset)
        judge_y = max(0, judge_y - int(0.01 * H))

        ped_offset = max(PED_MIN_PIX, int(PED_DELTA_FRAC * H))
        ped_y = max(0, judge_y - ped_offset)

        cv2.line(frame, (sx1, sy2), (sx2, sy2), (0, 140, 255), 2)
        cv2.line(frame, (sx1, judge_y), (sx2, judge_y), (0, 0, 255), 2)
        cv2.line(frame, (sx1, ped_y), (sx2, ped_y), (0, 165, 255), 1)

        # ====== PHASE 4 — HSV fallback cho đèn đỏ ======
        if not red_light:
            ok_hsv, _, _ = detect_red_signal_hsv(frame, stop_line_box, annotate=VIDEO_DEBUG)
            if ok_hsv:
                red_light = True

        # ====== PHASE 5 — M2 (helmet/mobile) ======
        m2_res = m2.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_M2_VIDEO,
            iou=NMS_IOU_M2,
            verbose=False,
            device=DEVICE,
            half=USE_FP16,
            workers=0,
        )

        viol_boxes = []
        for r in m2_res or []:
            if not getattr(r, "boxes", None):
                continue

            xy   = r.boxes.xyxy.cpu().numpy()
            cl   = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            names2 = r.names if hasattr(r, "names") else m2.names
            for bb, cc, cf in zip(xy, cl, conf):
                lab   = normalize_label(names2[int(cc)])
                score = float(cf)

                if lab == "no_helmet" and score < NO_HELMET_CONF_VIDEO:
                    continue

                if role_of(lab) == "violation":
                    b1, b2, b3, b4 = list(map(int, bb.tolist()))
                    viol_boxes.append(([b1, b2, b3, b4], lab))

        veh_list = []
        tid_list = []
        for tid, vbox in vehicle_boxes_by_tid:
            label = tracked[tid]["vehicle_type"] or ""
            veh_list.append((vbox, label))
            tid_list.append(tid)

        assigned_sets = assign_violations_to_vehicles(veh_list, viol_boxes)

        for idx, tid in enumerate(tid_list):
            info = tracked.get(tid)
            if not info:
                continue
            for ilabel in assigned_sets[idx]:
                info["votes"][ilabel] = info["votes"].get(ilabel, 0) + 1

        lane_front_flags = compute_lane_front_flags(
            veh_list, judge_y, stop_line_box, H
        ) if veh_list else []

        if stop_line_box and red_light:
            for idx, (tid, vbox) in enumerate(vehicle_boxes_by_tid):
                info = tracked.get(tid)
                if not info:
                    continue
                vlabel = info["vehicle_type"] or ""
                is_front = lane_front_flags[idx] if idx < len(lane_front_flags) else True

                if not in_stop_region(vbox, stop_line_box, W, H):
                    continue

                if is_front and crossed_stopline(
                    vbox, stop_line_box, W, H, vlabel, judge_y=judge_y
                ):
                    info["votes"]["red_light_violation"] = info["votes"].get(
                        "red_light_violation", 0
                    ) + 1

                if ped_y is not None and is_front and crossed_pedline(
                    vbox, ped_y, W, H, vlabel
                ):
                    info["votes"]["pedestrian_line_violation"] = info["votes"].get(
                        "pedestrian_line_violation", 0
                    ) + 1

        for tid, info in tracked.items():
            for lab, cnt in list(info["votes"].items()):
                if lab == "no_helmet":
                    need = 3
                elif lab == "red_light_violation":
                    need = VOTE_RED_CONSEC
                else:
                    need = VOTE_M2_CONSEC

                if cnt >= need:
                    info["violations"].add(lab)

        # ====== PHASE 7 — OCR biển số (chỉ xe vi phạm) ======
        for tid, vbox in vehicle_boxes_by_tid:
            info = tracked.get(tid)
            if not info or not info["vehicle_type"]:
                continue

            if not info["violations"]:
                continue

            if frame_idx - info["last_ocr_frame"] < OCR_INTERVAL:
                continue

            x1, y1, x2, y2 = vbox
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                info["last_ocr_frame"] = frame_idx
                continue

            p_res = m3.predict(
                roi, imgsz=IMG_SIZE, conf=CONF_M3, iou=NMS_IOU_M3,
                verbose=False, device=DEVICE, half=USE_FP16, workers=0,
            )

            best_pb = None
            best_area = -1
            for pr in p_res or []:
                if not getattr(pr, "boxes", None) or len(pr.boxes) == 0:
                    continue
                for pb in pr.boxes.xyxy.cpu().numpy():
                    x1p, y1p, x2p, y2p = pb
                    area = max(0, x2p - x1p) * max(0, y2p - y1p)
                    if area > best_area:
                        best_area = area
                        best_pb = [int(x1p), int(y1p), int(x2p), int(y2p)]

            if best_pb:
                px1, py1, px2, py2 = best_pb
                px1g, py1g, px2g, py2g = (
                    px1 + x1, py1 + y1,
                    px2 + x1, py2 + y1,
                )
                info["last_plate_box"] = (px1g, py1g, px2g, py2g)
                plate_roi = frame[py1g:py2g, px1g:px2g]

                txt = ocr_plate_multi(plate_roi)
                if txt:
                    info["plate_votes"][txt] = info["plate_votes"].get(txt, 0) + 1
                    if info["plate_votes"][txt] >= PLATE_STABLE_MIN:
                        info["plate"] = txt

            info["last_ocr_frame"] = frame_idx

        # ====== PHASE 8 — vẽ + lưu DB ======
        for tid, vbox in vehicle_boxes_by_tid:
            info = tracked.get(tid, {})
            vx1, vy1, vx2, vy2 = vbox
            has_vio = bool(info.get("violations"))

            color = (0, 0, 255) if has_vio else (0, 255, 0)
            thick = 3 if has_vio else 2
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, thick)

            vset = info.get("violations") or set()
            norm_vset = normalize_violation_labels(vset)
            label_text = f"{info.get('vehicle_type') or 'vehicle'}#{tid}"
            if has_vio and norm_vset:
                label_text += " | " + ",".join(sorted(norm_vset))
            cv2.putText(frame, label_text, (vx1, max(0, vy1 - 8)), 0, 0.6, color, 2)

            if has_vio and info.get("last_plate_box"):
                bx1, by1, bx2, by2 = info["last_plate_box"]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

                plate_to_show = info.get("plate") or "UNKNOWN"
                cv2.putText(
                    frame,
                    plate_to_show,
                    (bx1, max(0, by1 - 4)),
                    0,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        now = time.time()
        for tid, info in list(tracked.items()):
            has_plate_box = info.get("last_plate_box") is not None
            if info.get("vehicle_type") and (info.get("violations") or has_plate_box):
                if now - info.get("last_save_time", 0) >= COOLDOWN_S:
                    vset = info.get("violations") or set()
                    norm_vset = normalize_violation_labels(vset)
                    vio_str = ",".join(sorted(norm_vset)) if norm_vset else ""

                    plate_str = info.get("plate")
                    if not plate_str and has_plate_box:
                        plate_str = "UNKNOWN"

                    key = f"{plate_str or ''}|{vio_str}"
                    ok_save = True
                    if key.strip("|"):
                        last = RECENT_KEY_CACHE.get(key, 0)
                        if now - last < DEDUP_WINDOW_S:
                            ok_save = False

                    if ok_save:
                        snap = f"video_{Path(video_path).stem}_{frame_idx}_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(str(UPLOAD_DIR / snap), frame)
                        insert_record(
                            "video",
                            str(Path(video_path)),
                            frame_idx,
                            info["vehicle_type"],
                            vio_str,
                            plate_str,
                            snap,
                        )
                        info["last_save_time"] = now
                        RECENT_KEY_CACHE[key] = now
                        total += 1

        writer.write(frame)

    writer.release()
    out_full = transcode_to_h264_mp4(out_full)
    return os.path.basename(out_full), total

    return out_name, total

# ===== routes =====
@app.route("/")
def index():
    rows = c.execute(
        """
        SELECT id,kind,source,frame_index,timestamp,camera_id,
               vehicle_type,violation_types,license_plate,evidence_path,status
        FROM records ORDER BY id DESC LIMIT 12
        """
    ).fetchall()
    return render_template("index.html", rows=rows, now_year=datetime.datetime.now().year)


@app.route("/records")
def records_page():
    rows = c.execute(
        """
        SELECT id,kind,source,frame_index,timestamp,camera_id,
               vehicle_type,violation_types,license_plate,evidence_path,status
        FROM records ORDER BY id DESC LIMIT 500
        """
    ).fetchall()
    return render_template("records.html", rows=rows, now_year=datetime.datetime.now().year)


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", now_year=datetime.datetime.now().year)

@app.route("/uploads/<path:fname>")
def uploads(fname):
    fname = os.path.basename(str(fname).replace("\\", "/"))
    return send_from_directory(app.config["UPLOAD_FOLDER"], fname)

@app.route("/video_out/<path:fname>")
def video_out(fname):
    fname = os.path.basename(str(fname).replace("\\", "/"))
    # file giờ là MP4/H.264
    return send_from_directory(str(VIDEO_OUT), fname, mimetype="video/mp4")


@app.route("/detect_image", methods=["POST"])
def detect_image_route():
    files = request.files.getlist("image")
    if not files:
        return jsonify(ok=False, error="No file"), 400
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = []
    for f in files:
        if not f or not getattr(f, "filename", ""):
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in allowed_ext:
            continue
        in_name = _fname("upload", ext)
        in_path = str(UPLOAD_DIR / in_name)
        f.save(in_path)
        frame = cv2.imread(in_path)
        if frame is None:
            continue
        annotated, last_snap = analyze_image(frame)
        out_name = f"annotated_{in_name}"
        out_path = str(UPLOAD_DIR / out_name)
        cv2.imwrite(out_path, annotated)
        items.append({
            "filename": f.filename,
            "out_url": f"/uploads/{out_name}",
            "evidence_url": f"/uploads/{last_snap}" if last_snap else None,
        })

    if not items:
        return jsonify(ok=False, error="Không xử lý được ảnh nào"), 400

    return jsonify(
        ok=True,
        items=items,
        out_url=items[0]["out_url"],
        last_evidence=items[0].get("evidence_url"),
        total=len(items),
    )

@app.route("/detect_video", methods=["POST"])
def detect_video_route():
    f = request.files.get("video")
    if not f:
        return jsonify(ok=False, error="No file"), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        return jsonify(ok=False, error="Unsupported video"), 400
    in_name = _fname("upload", ext)
    in_path = str(UPLOAD_DIR / in_name)
    f.save(in_path)
    try:
        annotated_video_name, count = analyze_video(in_path)
    except Exception as e:
        return jsonify(ok=False, error=f"Video error: {e}"), 500
    return jsonify(
        ok=True,
        out_url=f"/video_out/{annotated_video_name}",
        records_added=int(count),
    )

@app.route("/detect_video_batch", methods=["POST"])
def detect_video_batch_route():
    files = request.files.getlist("videos")
    if not files:
        return jsonify(ok=False, error="No files"), 400

    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    results = []

    for f in files:
        if not f or not getattr(f, "filename", ""):
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in allowed_ext:
            results.append({
                "filename": f.filename,
                "ok": False,
                "error": "Unsupported video",
            })
            continue

        in_name = _fname("upload", ext)
        in_path = str(UPLOAD_DIR / in_name)
        f.save(in_path)

        try:
            annotated_video_name, count = analyze_video(in_path)
            results.append({
                "filename": f.filename,
                "ok": True,
                "out_url": f"/video_out/{annotated_video_name}",
                "records_added": int(count),
            })
        except Exception as e:
            results.append({
                "filename": f.filename,
                "ok": False,
                "error": f"Video error: {e}",
            })

    if not results:
        return jsonify(ok=False, error="Không xử lý được video nào"), 400

    return jsonify(ok=True, results=results)


@app.route("/export_csv")
def export_csv():
    csv_path = APP_ROOT / "violations_export.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "id", "kind", "source", "frame_index", "timestamp", "camera_id",
            "vehicle_type", "violation_types", "license_plate", "evidence_file",
        ])
        for row in c.execute(
            """
            SELECT id,kind,source,frame_index,timestamp,camera_id,
                   vehicle_type,violation_types,license_plate,evidence_path
            FROM records ORDER BY id ASC
            """
        ):
            w.writerow(row)
    return send_file(
        csv_path, as_attachment=True,
        download_name="violations_export.csv", mimetype="text/csv",
    )

@app.route("/dashboard_data")
def dashboard_data():
    violation_counts: Dict[str, int] = {}
    for (v,) in c.execute(
        "SELECT violation_types FROM records "
        "WHERE violation_types IS NOT NULL AND violation_types<>''"
    ):
        for p in [p.strip() for p in v.split(",") if p.strip()]:
            violation_counts[p] = violation_counts.get(p, 0) + 1

    plate_counts: Dict[str, int] = {}
    for (lp,) in c.execute(
        "SELECT license_plate FROM records "
        "WHERE license_plate IS NOT NULL AND license_plate<>''"
    ):
        plate_counts[lp] = plate_counts.get(lp, 0) + 1

    top_plates = sorted(plate_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    per_day: Dict[str, int] = {}
    for (ts,) in c.execute("SELECT timestamp FROM records"):
        d = (ts or "").split(" ")[0]
        if d:
            per_day[d] = per_day.get(d, 0) + 1
    days_sorted = sorted(per_day.items(), key=lambda x: x[0])

    return jsonify({
        "violations": violation_counts,
        "top_plates": [{"plate": p, "count": n} for p, n in top_plates],
        "timeline": [{"date": d, "count": n} for d, n in days_sorted],
    })

    
@app.route("/api/record/update_plate", methods=["POST"])
def api_update_plate():
    """
    Cập nhật biển số cho 1 bản ghi trong bảng records.

    Frontend gửi: {id, plate}
    - id: ID bản ghi (cột id)
    - plate: chuỗi biển số người dùng nhập (raw)
    """
    data = request.get_json(force=True, silent=True) or {}
    rid = data.get("id")
    raw = (data.get("plate") or "").strip()

    if not rid:
        return jsonify(ok=False, error="Missing id"), 400

    # Lấy vehicle_type để format chuẩn hơn (xe máy / ô tô)
    row = c.execute(
        "SELECT vehicle_type FROM records WHERE id=?",
        (rid,)
    ).fetchone()
    vehicle_label = row[0] if row else None

    # Chuẩn hoá biển số
    norm = ""
    valid = False
    if raw:
        norm = format_plate_text(raw, vehicle_label)
        if norm:
            valid = True

    # Lưu lại vào cột license_plate
    # - nếu format được: lưu dạng chuẩn (norm)
    # - nếu không format được: vẫn lưu raw để đỡ bị mất
    to_store = norm if norm else raw

    c.execute(
        "UPDATE records SET license_plate=? WHERE id=?",
        (to_store, rid)
    )
    conn.commit()

    return jsonify(
        ok=True,
        plate_norm=norm,          # dùng cho hiển thị (data.plate_norm)
        plate_valid=valid,
    )
@app.route("/api/record/update_status", methods=["POST"])
def api_update_status():
    """
    Cập nhật trạng thái bản ghi: pending / approved / rejected
    Frontend gửi: {id, status}
    """
    data = request.get_json(force=True, silent=True) or {}
    rid = data.get("id")
    status = (data.get("status") or "").strip().lower()

    if not rid or status not in {"pending", "approved", "rejected"}:
        return jsonify(ok=False, error="Invalid params"), 400

    try:
        rid_int = int(rid)
    except Exception:
        return jsonify(ok=False, error="Invalid id"), 400

    c.execute(
        "UPDATE records SET status=? WHERE id=?",
        (status, rid_int)
    )
    conn.commit()

    return jsonify(ok=True, id=rid_int, status=status)

if __name__ == "__main__":
    print(
        f"[INFO] Ready. Open http://127.0.0.1:5000  "
        f"(GPU: {DEVICE == 'cuda'}, FP16: {USE_FP16})"
    )
    print(f"[INFO] VIDEO_DEBUG={int(VIDEO_DEBUG)} (1=ON, 0=OFF)")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

