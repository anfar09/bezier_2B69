# 🔬 3D Point Cloud Surface Reconstruction

> **Bezier Cross-Hatching** — การซ่อมแซมรูโหว่บนพื้นผิว 3 มิติด้วยเส้นโค้ง Bezier แบบตาข่ายไขว้ พร้อมระบบ Surface Densification

---

## 📋 สารบัญ

- [ภาพรวม](#-ภาพรวม)
- [สถาปัตยกรรมระบบ](#-สถาปัตยกรรมระบบ)
- [ขั้นตอนการทำงาน (Pipeline)](#-ขั้นตอนการทำงาน-pipeline)
- [อัลกอริทึมหลัก](#-อัลกอริทึมหลัก)
- [โครงสร้างไฟล์](#-โครงสร้างไฟล์)
- [การติดตั้ง](#-การติดตั้ง)
- [การใช้งาน](#-การใช้งาน)
- [Chamfer Distance — ผลการทดสอบ](#-chamfer-distance--ผลการทดสอบ)
- [ข้อจำกัดและแนวทางพัฒนา](#-ขอจำกัดและแนวทางพัฒนา)

---

## 🎯 ภาพรวม

โปรเจกต์นี้เป็นเครื่องมือวิจัยสำหรับ **ซ่อมแซมรูโหว่บนพื้นผิว 3D Point Cloud** ที่เกิดจากการสแกนไม่สมบูรณ์ โดยใช้แนวทาง:

1. **Cross-Hatching Slicing** — หั่น point cloud เป็นแผ่น 2D ตามแกน PCA สองแกน
2. **Cubic Bezier Curve** — สร้างเส้นโค้ง G1-continuous เชื่อมขอบรูแต่ละ slice
3. **Surface Densification** — เติมจุดระหว่างเส้น Bezier ให้ดูเป็นพื้นผิวจริง

ผลลัพธ์คือ point cloud ที่มีรูโหว่ถูกเติมเต็มอย่างเรียบ โดยมี **Chamfer Distance ลดลงเฉลี่ย ~55%** เมื่อเทียบกับ Ground Truth

### ✨ จุดเด่น

| Feature | รายละเอียด |
|---|---|
| 🎯 **Hermite-to-Bezier** | คำนวณ control points จาก tangent vectors โดยตรง — ไม่ต้อง ray intersection |
| 📐 **Polyfit Tangent** | ใช้ Linear Regression (5 จุด) หา tangent ที่ robust, ไม่ diverge |
| 🧩 **Surface Densification** | สร้างเส้นกลาง + scatter จุดสุ่มสม่ำเสมอระหว่าง Bezier curves |
| 🌙 **Light/Dark Theme** | UI สลับ theme ได้ทันที พร้อม glassmorphism design |
| 🌡️ **3D Error Heatmap** | Chamfer Distance แสดงผลเป็น 3D heatmap บน point cloud จริง |
| ⚡ **Auto-Tuning** | Slice thickness, gap threshold, curve density ปรับอัตโนมัติจากข้อมูล |

---

## 🏗 สถาปัตยกรรมระบบ

```
┌──────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                     │
│  Tab 01: บทนำ  │  Tab 02: ประมวลผล  │  Tab 03: Deep-Dive 2D     │
│  Tab 04: ผลลัพธ์ 3D  │  Tab 05: Chamfer Distance Analysis       │
└──────────────────┬───────────────────────────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │    hole_filler.py (Core)    │
    │  ┌────────────────────────┐ │
    │  │ 1. PCA Alignment       │ │
    │  │ 2. Statistical Outlier │ │
    │  │    Removal (SOR)       │ │
    │  │ 3. Cross-Hatch Slicing │ │
    │  │ 4. Gap Detection       │ │
    │  │ 5. Bezier Curve Fill   │ │
    │  │ 6. Surface Densify     │ │
    │  │ 7. Inverse PCA + Merge │ │
    │  └────────────────────────┘ │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  metrics.py (Evaluation)    │
    │  • Chamfer Distance         │
    │  • Hausdorff Distance       │
    │  • RMSE                     │
    │  • Surface Roughness        │
    │  • Per-point Error Heatmap  │
    └─────────────────────────────┘
```

---

## 🔄 ขั้นตอนการทำงาน (Pipeline)

### Step 1: PCA Alignment
```
Input Point Cloud → Center (ลบ mean) → PCA → จัดแกนตาม Variance
```
- คำนวณ covariance matrix → eigenvectors
- จัดวาง: **PCA-X** = variance สูงสุด, **PCA-Y** = รอง, **PCA-Z** = ต่ำสุด (ความหนา)
- ทำให้การ slice ตรงกับโครงสร้างหลักของพื้นผิว

### Step 2: Statistical Outlier Removal (SOR)
```
PCA Points → หา k-NN mean distance → ตัดจุดที่ > μ + 2σ
```
- ใช้ k=10 nearest neighbors
- คำนวณ mean distance ของแต่ละจุด
- กรองจุดที่อยู่ไกลผิดปกติ (noise, outliers)

### Step 3: Auto-Tuning Parameters
```
Cleaned Points → cKDTree → avg_spacing → slice_thickness, gap_threshold
```
- **avg_point_spacing**: ค่าเฉลี่ยระยะห่างระหว่างจุดที่ใกล้ที่สุด (k=2)
- **slice_thickness**: `avg_spacing × 3.0` — ความหนาของแต่ละ slice
- **gap_threshold**: `avg_spacing × 3.0` — ระยะที่ถือว่าเป็น "รู"

### Step 4: Cross-Hatch Slicing + Gap Detection
```
PCA-X Slicing:  |---|---|---|---|---|  → detect gaps ในแต่ละ slice
PCA-Y Slicing:  ═══╤═══╤═══╤═══╤═══  → detect gaps ในแต่ละ slice
```
- สำหรับแต่ละแกน (X, Y):
  - แบ่งจุดตาม axis value เป็น slices
  - ในแต่ละ slice: sort จุดตามแกนที่ 2 → หา gap ที่ > threshold
  - Record: `(p_left, p_right, distance, axis_val)` สำหรับแต่ละรู

### Step 5: Bezier Curve Gap Filling

#### 5.1 Tangent Estimation (Linear Regression)
```python
# ฝั่งซ้าย: fit เส้นตรงผ่าน 5 จุดข้างๆ P0 → ได้ slope
coeffs = np.polyfit(left_pts[:, 0], left_pts[:, 1], deg=1)
v_left = [1.0, slope]  # ชี้ขวาเข้ารูเสมอ

# ฝั่งขวา: fit เส้นตรงผ่าน 5 จุดข้างๆ P3 → ได้ slope  
v_right = [-1.0, -slope]  # ชี้ซ้ายเข้ารูเสมอ
```
**ข้อดี**: `v_left` มี x เป็นบวกเสมอ, `v_right` มี x เป็นลบเสมอ → **รับประกัน convergence**

#### 5.2 Hermite-to-Bezier Control Points
```
P1 = P0 + (gap_length / 3) × v_left    ← control point ซ้าย (อยู่ในรูเสมอ)
P2 = P3 + (gap_length / 3) × v_right   ← control point ขวา (อยู่ในรูเสมอ)
```
ไม่ต้อง ray intersection → **ไม่มีกรณี parallel/diverge ที่จะทำให้ผิดพลาด**

#### 5.3 Cubic Bezier Curve
```
B(t) = (1-t)³·P0 + 3(1-t)²t·P1 + 3(1-t)t²·P2 + t³·P3
```
คุณสมบัติ: **G1 continuous** ที่ขอบรู (tangent ต่อเนื่อง)

### Step 6: Surface Densification
```
Slice N  : ────Bezier Curve────
             ↓ interpolate ↓
Middle   : ····interp curve····   ← เส้นกลางใหม่
             ↓ interpolate ↓  
Slice N+1: ────Bezier Curve────
             + scatter random pts ← จุดสุ่มสม่ำเสมอ
```
1. **Group** curves ที่อยู่ใน slice ติดกันและปิดรูเดียวกัน
2. **Interpolate** เส้นกลาง (linear lerp ระหว่าง curve pairs)
3. **Scatter** จุดสุ่มแบบ bilinear ในพื้นที่ quad ระหว่าง rails (ความหนาแน่นตาม `avg_spacing²`)

### Step 7: Inverse PCA + Merge
```
PCA points → ×R^T + mean → Original space → Merge with distance check
```
- แปลงกลับจาก PCA space → พิกัดจริง
- Merge: จุดที่อยู่ใกล้กว่า `avg_spacing × 0.2` จะถูกรวม (average)

---

## 🧮 อัลกอริทึมหลัก

### ทำไมถึงใช้ Polyfit (Linear Regression)?

| วิธี | ปัญหา | แก้ไขได้? |
|---|---|---|
| **1 จุดข้างเคียง** | จุดติดกันมาก → tangent เพี้ยน | ❌ |
| **PCA (eigenvalue)** | ดึง neighbor จาก slice อื่น → ผิดทิศ | ❌ |
| **Average 5 vectors** | Diverge ได้ → เส้นไม่ตัดกัน | ❌ |
| **Polyfit (เส้นตรง)** | Robust, ทิศตรง, convergence 100% | ✅ |

### ทำไมถึงใช้ Hermite แทน Apex Intersection?

| วิธี | กรณี Diverge | กรณี Parallel | ข้อจำกัด |
|---|---|---|---|
| **Ray Intersection (Apex)** | Fallback ไม่ดี → apex ผิดที่ | Singular matrix | ต้องการ ray ตัดกันใน gap |
| **Hermite-to-Bezier** | ✅ ไม่เกิด | ✅ ไม่เกิด | ไม่มี — ใช้ได้ทุกกรณี |

---

## 📁 โครงสร้างไฟล์

```
bezier_2B69/
├── app.py               # Streamlit UI (Light/Dark theme, 5 tabs)
├── hole_filler.py       # Core algorithm (PCA → SOR → Slice → Bezier → Densify → Merge)
├── metrics.py           # Chamfer Distance, Hausdorff, RMSE, Surface Roughness
├── slicer.py            # File I/O (.xyz, .txt, .pts)
├── comparative.py       # Alternative fill methods (linear, spline, etc.)
├── requirements.txt     # Dependencies
├── Dataset/
│   ├── hole/            # Input: point clouds WITH holes (H1.xyz ~ H10.xyz)
│   └── before/          # Ground truth: COMPLETE point clouds (H1.xyz ~ H10.xyz)
```

### ไฟล์หลัก

| ไฟล์ | บรรทัด | หน้าที่ |
|---|---|---|
| `hole_filler.py` | ~950 | Pipeline ทั้งหมด: PCA, SOR, Slicing, Gap Detection, Bezier Fill, Densification, Merge |
| `app.py` | ~580 | Streamlit UI: 5 tabs, theme system, 2D/3D visualization, Chamfer Distance analysis |
| `metrics.py` | ~320 | Quantitative metrics: CD, Hausdorff, RMSE, Surface Roughness, Per-point Error |
| `slicer.py` | ~60 | โหลดไฟล์ point cloud (.xyz, .txt, .pts) |

---

## 🚀 การติดตั้ง

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone
git clone <repository-url>
cd bezier_2B69

# Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install Dependencies
pip install -r requirements.txt
```

### จัดเตรียมข้อมูล
วางไฟล์ point cloud ในโฟลเดอร์:

```
Dataset/
├── hole/          # ไฟล์ที่มีรู (input)
│   ├── H1.xyz
│   └── ...
└── before/        # ไฟล์ต้นฉบับ (ground truth) — ไม่บังคับ ใช้สำหรับ Chamfer Distance
    ├── H1.xyz
    └── ...
```

**รูปแบบไฟล์** `.xyz`:
```
x1 y1 z1
x2 y2 z2
...
```

---

## 💻 การใช้งาน

### เปิด Web Application
```bash
streamlit run app.py
```
เปิดเบราว์เซอร์ไปที่ `http://localhost:8501`

### Tab ต่างๆ

| Tab | ฟังก์ชัน |
|---|---|
| **📍 01. บทนำ** | แสดง raw point cloud, ตั้งค่าการแสดงผล |
| **⚙️ 02. ประมวลผล** | กดปุ่ม "เริ่มประมวลผล" → รัน pipeline ทั้งหมด |
| **🔗 03. การเชื่อมต่อ** | Deep-dive: ดู Bezier curve ทีละ slice (2D) |
| **💎 04. ผลลัพธ์** | แสดง 3D reconstruction, download .xyz |
| **📐 05. Chamfer Distance** | 3D error heatmap, metrics, comparison charts |

### ใช้ผ่าน Python โดยตรง
```python
from hole_filler import process_point_cloud
from slicer import load_points_from_file
import numpy as np

# โหลดข้อมูล
points = np.array(load_points_from_file("Dataset/hole/H1.xyz"))

# ประมวลผล (auto-tune ทุกค่า)
result = process_point_cloud(points, verbose=True)

# ผลลัพธ์
merged = result['merged_points']        # point cloud ที่ซ่อมแล้ว
filled = result['combined_bezier_pts']  # เฉพาะจุดที่เติมใหม่

# บันทึก
np.savetxt("output.xyz", merged, fmt='%.8f')
```

### ปรับค่าด้วยตนเอง
```python
result = process_point_cloud(
    points,
    slice_thickness=0.005,   # default: auto (avg_spacing × 3)
    gap_threshold=0.008,     # default: auto (avg_spacing × 3)    
    verbose=True
)
```

---

## 📊 Chamfer Distance — ผลการทดสอบ

### ผลการทดสอบกับ Dataset 10 ไฟล์

| File | Hole pts | Before pts | Merged pts | Filled pts | CD Input→GT | CD Repaired→GT | **Improvement** | RMSE | Time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 621 | 690 | 1,202 | 643 | 8.96e-06 | 2.61e-06 | **70.8%** | 0.00213 | 0.05s |
| H2 | 670 | 779 | 1,113 | 513 | 2.20e-05 | 6.73e-06 | **69.4%** | 0.00332 | 0.04s |
| H3 | 1,283 | 1,363 | 1,866 | 685 | 6.04e-06 | 3.33e-06 | **44.8%** | 0.00254 | 0.05s |
| H4 | 1,523 | 1,709 | 2,550 | 1,145 | 1.81e-05 | 3.97e-06 | **78.1%** | 0.00267 | 0.06s |
| H5 | 461 | 543 | 744 | 327 | 1.20e-05 | 3.83e-06 | **68.2%** | 0.00253 | 0.02s |
| H6 | 1,039 | 1,257 | 2,113 | 1,199 | 2.30e-05 | 5.54e-06 | **76.0%** | 0.00290 | 0.05s |
| H7 | 666 | 750 | 1,183 | 602 | 1.27e-05 | 8.32e-06 | **34.4%** | 0.00336 | 0.03s |
| H8 | 595 | 644 | 993 | 467 | 4.82e-06 | 5.42e-06 | **-12.4%** | 0.00300 | 0.03s |
| H9 | 490 | 554 | 774 | 323 | 8.70e-06 | 4.53e-06 | **47.9%** | 0.00309 | 0.02s |
| H10 | 1,148 | 1,447 | 2,824 | 1,851 | 2.90e-05 | 1.05e-05 | **63.8%** | 0.00386 | 0.07s |

### สรุปผล

| Metric | ค่า |
|---|---|
| **CD Improvement เฉลี่ย** | **54.1%** (9/10 ไฟล์ดีขึ้น) |
| **CD Improvement สูงสุด** | **78.1%** (H4) |
| **RMSE เฉลี่ย (จุดเติม)** | **0.00294** |
| **เวลาประมวลผลเฉลี่ย** | **0.04s** |
| **จุดเติมเฉลี่ย** | ~700 จุดต่อไฟล์ |

---

## 📝 ข้อจำกัดและแนวทางพัฒนา

### ข้อจำกัดปัจจุบัน

1. **H8 ได้ค่าแย่ลง (-12.4%)** — เมื่อรูมีขนาดเล็กมากและอยู่ในบริเวณ flat การเติมจุดอาจเพิ่ม noise มากกว่าแก้ปัญหา
2. **Gap detection ขึ้นกับ sorting order** — ถ้าพื้นผิวมีรูปร่างซับซ้อน (undercut) การ sort 1D อาจพลาด gap
3. **Linear tangent** — ใช้ polyfit degree 1 ซึ่งไม่จับ curvature ท้องถิ่น (degree 2 อาจดีกว่าถ้ามีจุดพอ)

### แนวทางพัฒนาต่อ

- **Weighted regression**: ให้น้ำหนักจุดใกล้ขอบรูมากกว่า
- **Adaptive density**: ปรับจำนวน scatter points ตาม curvature ท้องถิ่น
- **Multi-scale slicing**: ใช้หลาย slice thickness สำหรับรูขนาดต่างกัน
- **Normal estimation**: เพิ่มการคำนวณ surface normal เพื่อ constrain Bezier ให้อยู่บนพื้นผิว

---

## 📜 License

MIT License

---

## 🙏 Acknowledgements

- **PCA / SOR**: scikit-learn, scipy
- **Visualization**: Plotly, Streamlit
- **Evaluation**: Chamfer Distance, Hausdorff Distance — standard 3D reconstruction metrics