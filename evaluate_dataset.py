import os
import time
import numpy as np
from slicer import load_points_from_file
from hole_filler import process_point_cloud
from metrics import compute_chamfer_distance, compute_rmse

hole_dir = 'Dataset/hole'
before_dir = 'Dataset/before'

print("| File | Hole pts | Before pts | Merged pts | Filled pts | CD Input→GT | CD Repaired→GT | **Improvement** | RMSE | Time |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

improvements = []
rmses = []
times = []
filled_pts = []

for i in range(1, 11):
    fname = f"H{i}.xyz"
    hole_path = os.path.join(hole_dir, fname)
    before_path = os.path.join(before_dir, fname)
    
    if not os.path.exists(hole_path) or not os.path.exists(before_path):
        continue
        
    pts_hole = load_points_from_file(hole_path)
    pts_before = load_points_from_file(before_path)
    
    t0 = time.time()
    result = process_point_cloud(pts_hole, verbose=False)
    t1 = time.time()
    
    merged = result['merged_points']
    filled = result['combined_bezier_pts']
    
    cd_input = compute_chamfer_distance(pts_hole, pts_before)
    cd_repair = compute_chamfer_distance(merged, pts_before)
    
    imp = (cd_input - cd_repair) / cd_input * 100
    
    if len(filled) > 0:
        rmse = compute_rmse(filled, pts_before)
    else:
        rmse = 0.0
        
    print(f"| H{i} | {len(pts_hole):,} | {len(pts_before):,} | {len(merged):,} | {len(filled):,} | {cd_input:.2e} | {cd_repair:.2e} | **{imp:.1f}%** | {rmse:.5f} | {t1-t0:.2f}s |")
    
    improvements.append(imp)
    if len(filled) > 0:
        rmses.append(rmse)
    times.append(t1-t0)
    filled_pts.append(len(filled))

print("\n### สรุปผล")
print("| Metric | ค่า |")
print("|---|---|")
better_count = sum(1 for i in improvements if i > 0)
worst_idx = np.argmin(improvements)
worst_name = f"H{worst_idx+1}"
worst_val = improvements[worst_idx]

print(f"| 🏆 **CD Improvement เฉลี่ย** | **{np.mean(improvements):.1f}%** ({better_count}/{len(improvements)} ไฟล์ดีขึ้น) |")
print(f"| �� **CD Improvement สูงสุด** | **{np.max(improvements):.1f}%** (H{np.argmax(improvements)+1}) |")
print(f"| 📉 **กรณีแย่ลง** | {worst_name} ({worst_val:.1f}%) |")
print(f"| 📐 **RMSE เฉลี่ย (จุดเติม)** | **{np.mean(rmses):.5f}** |")
print(f"| ⚡ **เวลาประมวลผลเฉลี่ย** | **{np.mean(times):.2f}s** |")
print(f"| 🔢 **จุดเติมเฉลี่ย** | ~{int(np.mean(filled_pts))} จุดต่อไฟล์ |")

