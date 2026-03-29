import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import io

from slicer import load_points_from_file
from hole_filler import (process_point_cloud, apply_inverse_pca,
                         estimate_tangent_2d, find_apex_2d,
                         axis_col, compute_g1_angle)
from metrics import (compute_all_metrics, chamfer_distance, surface_roughness,
                     point_density_uniformity)

# --- Page Config ---
st.set_page_config(
    page_title="3D Point Cloud Reconstruction — Research",
    page_icon="🔬", layout="wide", initial_sidebar_state="collapsed"
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Sarabun', sans-serif; }
.main { background-color: #F8FAFC; color: #1E293B; }
:root { --primary: #3B82F6; --border: #E2E8F0; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background-color: transparent; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { height: 46px; background-color: #FFF; border-radius: 10px 10px 0 0; padding: 0 18px; color: #64748B; border: 1px solid var(--border); font-weight: 500; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background-color: #EFF6FF !important; color: var(--primary) !important; border-bottom: 3px solid var(--primary) !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
.title-main { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0; letter-spacing: -1px; }
.step-header { font-size: 1.3rem; color: #1E293B; border-left: 5px solid var(--primary); padding-left: 15px; margin-top: 20px; margin-bottom: 12px; font-weight: 600; }
.method-card { background: #FFF; padding: 18px; border-radius: 14px; border: 1px solid var(--border); box-shadow: 0 4px 6px -1px rgb(0 0 0/0.05); margin-bottom: 14px; }
.stMetric { background: #FFF; padding: 16px; border-radius: 10px; border: 1px solid var(--border); box-shadow: 0 1px 3px 0 rgb(0 0 0/0.1); }
.stButton>button { background: white; color: var(--primary); border: 2px solid var(--primary); padding: 8px 24px; border-radius: 10px; font-weight: 600; }
.stButton>button:hover { background: var(--primary); color: white; box-shadow: 0 10px 15px -3px rgba(59,130,246,0.3); }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
HOLE_FOLDER = "Dataset/hole"
BEFORE_FOLDER = "Dataset/before"
os.makedirs(HOLE_FOLDER, exist_ok=True)
os.makedirs(BEFORE_FOLDER, exist_ok=True)

hole_files = sorted([f for f in os.listdir(HOLE_FOLDER) if f.endswith(('.xyz', '.txt', '.pts'))])
before_files = sorted([f for f in os.listdir(BEFORE_FOLDER) if f.endswith(('.xyz', '.txt', '.pts'))])

if "hf_result" not in st.session_state: st.session_state["hf_result"] = None
if "hf_file" not in st.session_state: st.session_state["hf_file"] = None

# --- Header ---
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown('<p class="title-main">3D Reconstruction Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; font-size:1rem;">การเติมเต็มพื้นผิวและซ่อมแซมรูโหว่ด้วย Bezier Cross-Hatching</p>', unsafe_allow_html=True)

with col_t2:
    if not hole_files:
        st.error("ไม่พบไฟล์ข้อมูลใน Dataset/hole/")
        st.stop()
    selected_file = st.selectbox("เลือกข้อมูล (hole)", hole_files, label_visibility="collapsed")
    file_path = os.path.join(HOLE_FOLDER, selected_file)

    @st.cache_data
    def load_data(fp): return np.array(load_points_from_file(fp))

    if st.session_state["hf_file"] != selected_file:
        st.session_state["points_raw"] = load_data(file_path)
        st.session_state["hf_file"] = selected_file
        st.session_state["hf_result"] = None
    points_raw = st.session_state["points_raw"]

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# TABS  (reordered: intro → process → deep-dive → result → metrics)
# =====================================================================
tab_intro, tab_analysis, tab_deepdive, tab_result, tab_metrics = st.tabs([
    "📍 01. บทนำ",
    "⚙️ 02. ประมวลผล",
    "🔗 03. การเชื่อมต่อ",
    "💎 04. ผลลัพธ์",
    "📐 05. Metrics",
])

# =====================================================================
# TAB 1: INTRODUCTION
# =====================================================================
with tab_intro:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown('<p class="step-header">ภาพรวมของโครงการ</p>', unsafe_allow_html=True)
        st.write("""
        การซ่อมแซมพื้นผิว 3 มิติ (3D Surface Repair) เป็นส่วนสำคัญในงานอุตสาหกรรมสมัยใหม่
        ช่วยให้เราสามารถกู้คืนโมเดลที่เสียหายจากการสแกน
        
        **หัวข้อหลัก:**
        - **Data Cleaning:** การลบจุดรบกวน (SOR)
        - **Geometric Alignment:** การจัดวาง PCA
        - **Surface Generation:** การสร้างพื้นผิวใหม่ด้วย Cross-Hatching
        """)
        st.markdown('<p class="step-header">ตั้งค่าการแสดงผล</p>', unsafe_allow_html=True)
        p_size = st.slider("ขนาดจุด", 1.0, 5.0, 2.0)
        p_color = st.selectbox("โทนสี", ["Blues", "Greens", "Viridis", "Cividis"])

    with col_r:
        fig1 = go.Figure(data=[go.Scatter3d(
            x=points_raw[:, 0], y=points_raw[:, 1], z=points_raw[:, 2],
            mode='markers',
            marker=dict(size=p_size, color=points_raw[:, 2], colorscale=p_color, opacity=0.6),
            name="Raw Input"
        )])
        fig1.update_layout(
            scene=dict(aspectmode='data',
                       xaxis=dict(gridcolor='#E2E8F0', backgroundcolor='white'),
                       yaxis=dict(gridcolor='#E2E8F0', backgroundcolor='white'),
                       zaxis=dict(gridcolor='#E2E8F0', backgroundcolor='white')),
            margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white", height=600
        )
        st.plotly_chart(fig1, use_container_width=True)

# =====================================================================
# TAB 2: PROCESSING
# =====================================================================
with tab_analysis:
    col_m, col_p = st.columns([1, 2])
    with col_m:
        st.markdown('<p class="step-header">Methodology</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="method-card"><b>1. Statistical Removal (SOR)</b><br>กรองจุดที่ไม่ใช่ส่วนของพื้นผิวจริง</div>
        <div class="method-card"><b>2. Axis Alignment (PCA)</b><br>คำนวณ Variance เพื่อหาแกนหลัก</div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ พารามิเตอร์")
        s_th = st.text_input("Slice Thickness", placeholder="Auto", key="s_th")
        g_th = st.text_input("Gap Threshold", placeholder="Auto", key="g_th")
        
        if st.button("🔍 เริ่มประมวลผล", use_container_width=True):
            with st.spinner("กำลังวิเคราะห์..."):
                s_val = float(s_th) if s_th.strip() else None
                g_val = float(g_th) if g_th.strip() else None
                st.session_state["hf_result"] = process_point_cloud(
                    points_raw, slice_thickness=s_val, gap_threshold=g_val, verbose=False
                )

    with col_p:
        result = st.session_state.get("hf_result")
        if result:
            st.markdown('<p class="step-header">ผลการตรวจหาจุดโหว่</p>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("จุดที่ผ่านการกรอง", f"{len(result['original_inlier_points']):,}")
            m2.metric("จำนวนรูโหว่", f"{len(result['gaps_axis1']) + len(result['gaps_axis2'])}")
            m3.metric("ความหนาแน่นเฉลี่ย", f"{result['avg_point_spacing']:.4f}")
            m4.metric("เวลารวม", f"{result.get('timings', {}).get('total', 0):.2f}s")

            orig = result['original_inlier_points']
            bound = result['combined_boundary_pts']
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter3d(
                x=orig[:, 0], y=orig[:, 1], z=orig[:, 2],
                mode='markers', marker=dict(size=1.5, color='#94A3B8', opacity=0.2), name="Base"
            ))
            if len(bound) > 0:
                fig2.add_trace(go.Scatter3d(
                    x=bound[:, 0], y=bound[:, 1], z=bound[:, 2],
                    mode='markers', marker=dict(size=4, color='#EF4444', opacity=0.8,
                                                line=dict(width=1, color='white')),
                    name="Hole Boundary"
                ))
            fig2.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0),
                               paper_bgcolor="white", height=600)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("กรุณากด 'เริ่มประมวลผล'")

# =====================================================================
# TAB 3: DEEP-DIVE — 2D SLICE VISUALIZATION
# =====================================================================
with tab_deepdive:
    result = st.session_state.get("hf_result")
    if result:
        st.markdown('<p class="step-header">🔗 กระบวนการเชื่อมต่อ (Bezier Curve Deep-Dive)</p>', unsafe_allow_html=True)
        st.write("เลือก Slice ที่พบรูโหว่ แล้วดูว่า Bezier Curve ถูกสร้างขึ้นมาเชื่อมจุดอย่างไร (มุมมอง 2D)")

        # Group gaps by slice
        from collections import defaultdict
        gaps_1 = result.get('gaps_axis1', [])
        gaps_2 = result.get('gaps_axis2', [])
        slices_with_holes = defaultdict(list)
        for g in gaps_1:
            _, _, dist, axis_val = g
            slices_with_holes[f"Axis {result['axis_1']} | slice={axis_val:.4f}"].append(g)
        for g in gaps_2:
            _, _, dist, axis_val = g
            slices_with_holes[f"Axis {result['axis_2']} | slice={axis_val:.4f}"].append(g)

        if not slices_with_holes:
            st.info("ไม่พบ Slice ที่มีรูโหว่")
        else:
            col_ctrl, col_viz = st.columns([1, 2.5])

            with col_ctrl:
                slice_keys = list(slices_with_holes.keys())
                selected_slice = st.selectbox("เลือก Slice", slice_keys, key="dd_slice")
                gap_list = slices_with_holes[selected_slice]

                st.markdown(f"**พบ {len(gap_list)} ช่องว่าง (gaps)**")

                gap_idx = st.selectbox("เลือก Gap", range(len(gap_list)),
                                       format_func=lambda i: f"Gap {i+1}: dist={gap_list[i][2]:.4f}",
                                       key="dd_gap")

                st.markdown("---")
                st.markdown("### 🎬 ขั้นตอน")
                step = st.radio("เลือกขั้นตอน", [
                    "① Slice Points (จุดทั้งหมด)",
                    "② Gap Detection (ระบุขอบรู)",
                    "③ Control Points (จุดควบคุม)",
                    "④ Bezier Curve (เส้นโค้ง)",
                ], key="dd_step")
                n_curve_pts = st.slider("จำนวนจุดบนเส้นโค้ง", 5, 50, 20, key="dd_npts")

            with col_viz:
                p_left_pca, p_right_pca, gap_dist, axis_val = gap_list[gap_idx]

                # Determine slice axis
                if "Axis X" in selected_slice: slice_axis = 'X'
                elif "Axis Y" in selected_slice: slice_axis = 'Y'
                else: slice_axis = 'Z'

                pts_pca = result['pts_clean_pca']
                rot_mat = result['rotation_matrix']
                mean_pt = result['mean_pt']
                thickness = result['slice_thickness']

                # Get points in this slice (PCA space)
                col_idx = axis_col(slice_axis)
                vals = pts_pca[:, col_idx]
                half_t = thickness / 2.0
                slice_mask = (vals >= axis_val - half_t) & (vals < axis_val + half_t)
                slice_pts_pca = pts_pca[slice_mask]

                # Determine 2D axes (the two axes that are NOT the slice axis)
                axes_3d = [0, 1, 2]
                axes_3d.remove(col_idx)
                ax_h, ax_v = axes_3d[0], axes_3d[1]
                axis_names = ['PCA-X', 'PCA-Y', 'PCA-Z']

                # === TRIANGULATION ALGORITHM ===
                from scipy.spatial import KDTree
                tree = KDTree(pts_pca)
                k_nb = min(6, len(pts_pca))
                _, idx_l = tree.query(p_left_pca, k=k_nb)
                _, idx_r = tree.query(p_right_pca, k=k_nb)
                nb_left = pts_pca[[i for i in idx_l if not np.allclose(pts_pca[i], p_left_pca)]]
                nb_right = pts_pca[[i for i in idx_r if not np.allclose(pts_pca[i], p_right_pca)]]

                p_l_2d = p_left_pca[[ax_h, ax_v]]
                p_r_2d = p_right_pca[[ax_h, ax_v]]
                
                slice_points_2d = np.empty((0, 2))
                if len(nb_left) > 0 and len(nb_right) > 0:
                    slice_points_2d = np.vstack([nb_left[:, [ax_h, ax_v]], nb_right[:, [ax_h, ax_v]]])

                v_left = estimate_tangent_2d(slice_points_2d, p_l_2d, opposite_2d=p_r_2d)
                v_right = estimate_tangent_2d(slice_points_2d, p_r_2d, opposite_2d=p_l_2d)

                p_c_2d = find_apex_2d(p_l_2d, v_left, p_r_2d, v_right, gap_dist)
                
                # Auto Curve Tension logic (mirroring hole_filler.py)
                midpoint = (p_l_2d + p_r_2d) / 2.0
                offset = np.linalg.norm(p_c_2d - midpoint)
                bulge_ratio = offset / gap_dist if gap_dist > 1e-12 else 0.0
                c_tension = 0.666
                if bulge_ratio > 0.05:
                    factor = np.clip((bulge_ratio - 0.05) / 0.20, 0.0, 1.0)
                    c_tension = 0.666 - (factor * 0.45)

                # Step 2: Control points with auto ratio in 2D
                P0 = p_l_2d
                P1 = c_tension * p_c_2d + (1.0 - c_tension) * p_l_2d
                P2 = c_tension * p_c_2d + (1.0 - c_tension) * p_r_2d
                P3 = p_r_2d

                arrow_scale = gap_dist * 0.6
                v_left_scaled = v_left * arrow_scale
                v_right_scaled = v_right * arrow_scale

                # Compute Bezier curve in 2D then map to 3D
                t_vals = np.linspace(0.0, 1.0, n_curve_pts)[:, np.newaxis]
                u = 1.0 - t_vals
                curve_2d = u**3*P0 + 3*u**2*t_vals*P1 + 3*u*t_vals**2*P2 + t_vals**3*P3
                
                curve_pca = np.zeros((n_curve_pts, 3))
                curve_pca[:, col_idx] = axis_val
                curve_pca[:, ax_h] = curve_2d[:, 0]
                curve_pca[:, ax_v] = curve_2d[:, 1]
                
                ctrl_pts_pca = np.zeros((4, 3))
                ctrl_pts_pca[:, col_idx] = axis_val
                ctrl_pts_pca[:, ax_h] = [P0[0], P1[0], P2[0], P3[0]]
                ctrl_pts_pca[:, ax_v] = [P0[1], P1[1], P2[1], P3[1]]

                # === BUILD 2D FIGURE ===
                fig = go.Figure()

                # Step 1+: All slice points (2D projection)
                if len(slice_pts_pca) > 0:
                    fig.add_trace(go.Scatter(
                        x=slice_pts_pca[:, ax_h], y=slice_pts_pca[:, ax_v],
                        mode='markers', marker=dict(size=5, color='#3B82F6', opacity=0.7),
                        name=f"Slice Points ({len(slice_pts_pca)})"
                    ))

                # Step 2+: Gap boundary + slope vectors
                if "②" in step or "③" in step or "④" in step:
                    fig.add_trace(go.Scatter(
                        x=[P0[0], P3[0]], y=[P0[1], P3[1]],
                        mode='markers+text',
                        marker=dict(size=14, color='#EF4444', symbol='diamond',
                                    line=dict(width=2, color='white')),
                        text=['P0 (pl_n)', 'P3 (pr_m)'], textposition='top center',
                        textfont=dict(size=12, color='#EF4444'),
                        name="Gap Boundary"
                    ))
                    # Dashed gap line
                    fig.add_trace(go.Scatter(
                        x=[P0[0], P3[0]], y=[P0[1], P3[1]],
                        mode='lines', line=dict(color='#EF4444', width=2, dash='dash'),
                        name=f"Gap (dist={gap_dist:.4f})"
                    ))
                    # Slope vector arrows (v_l, v_r)
                    tip_l = P0 + v_left_scaled
                    tip_r = P3 + v_right_scaled
                    fig.add_trace(go.Scatter(
                        x=[P0[0], tip_l[0]], y=[P0[1], tip_l[1]],
                        mode='lines+markers',
                        line=dict(color='#8B5CF6', width=3),
                        marker=dict(size=[0, 10], color='#8B5CF6', symbol=['circle', 'arrow-up']),
                        name="v_l (slope left)"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[P3[0], tip_r[0]], y=[P3[1], tip_r[1]],
                        mode='lines+markers',
                        line=dict(color='#EC4899', width=3),
                        marker=dict(size=[0, 10], color='#EC4899', symbol=['circle', 'arrow-up']),
                        name="v_r (slope right)"
                    ))

                # Step 3+: Apex point + control points + 1:2 ratio lines
                if "③" in step or "④" in step:
                    # Apex point p_c_2d (star)
                    fig.add_trace(go.Scatter(
                        x=[p_c_2d[0]], y=[p_c_2d[1]],
                        mode='markers+text',
                        marker=dict(size=18, color='#F59E0B', symbol='star',
                                    line=dict(width=2, color='#92400E')),
                        text=['p_c (Apex)'], textposition='top center',
                        textfont=dict(size=13, color='#F59E0B', family='Inter'),
                        name="Apex Point (p_c)"
                    ))
                    # Lines from apex to boundaries (showing triangulation)
                    fig.add_trace(go.Scatter(
                        x=[P0[0], p_c_2d[0], P3[0]],
                        y=[P0[1], p_c_2d[1], P3[1]],
                        mode='lines', line=dict(color='#F59E0B', width=2, dash='dot'),
                        name="Triangulation"
                    ))
                    # Control points (1:2 ratio)
                    fig.add_trace(go.Scatter(
                        x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                        mode='markers+text',
                        marker=dict(size=12, color='#F97316', symbol='cross',
                                    line=dict(width=1, color='#7C2D12')),
                        text=['P1 (2/3 apex)', 'P2 (1/3 apex)'],
                        textposition='bottom center',
                        textfont=dict(size=10, color='#F97316'),
                        name="Control Points (1:2)"
                    ))

                # Step 4: Bezier curve
                if "④" in step:
                    fig.add_trace(go.Scatter(
                        x=curve_2d[:, 0], y=curve_2d[:, 1],
                        mode='lines+markers',
                        line=dict(color='#10B981', width=4),
                        marker=dict(size=5, color='#10B981'),
                        name=f"Bezier Curve ({n_curve_pts} pts)"
                    ))

                fig.update_layout(
                    xaxis_title=axis_names[ax_h], yaxis_title=axis_names[ax_v],
                    xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor='#F1F5F9'),
                    yaxis=dict(gridcolor='#F1F5F9'),
                    paper_bgcolor='white', plot_bgcolor='#FAFBFC',
                    height=550, margin=dict(l=60, r=20, t=30, b=60),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Info card
                p_c_pca = np.zeros(3)
                p_c_pca[col_idx] = axis_val
                p_c_pca[ax_h] = p_c_2d[0]
                p_c_pca[ax_v] = p_c_2d[1]

                p_left_orig = apply_inverse_pca(ctrl_pts_pca[0].reshape(1, 3), rot_mat, mean_pt)[0]
                p_right_orig = apply_inverse_pca(ctrl_pts_pca[3].reshape(1, 3), rot_mat, mean_pt)[0]
                p_c_orig = apply_inverse_pca(p_c_pca.reshape(1, 3), rot_mat, mean_pt)[0]
                ctrl_orig = apply_inverse_pca(ctrl_pts_pca, rot_mat, mean_pt)

                st.markdown(f"""
                <div class="method-card">
                <b>📝 Triangulation + 1:2 Ratio</b><br>
                <b>Slice:</b> {slice_axis}={axis_val:.4f} &nbsp;|&nbsp;
                <b>Gap:</b> {gap_dist:.4f}<br><br>
                <b>🔺 Triangulation:</b><br>
                <b>P0 (pl_n):</b> [{p_left_orig[0]:.4f}, {p_left_orig[1]:.4f}, {p_left_orig[2]:.4f}]<br>
                <b>P3 (pr_m):</b> [{p_right_orig[0]:.4f}, {p_right_orig[1]:.4f}, {p_right_orig[2]:.4f}]<br>
                <b>⭐ p_c (Apex):</b> [{p_c_orig[0]:.4f}, {p_c_orig[1]:.4f}, {p_c_orig[2]:.4f}]<br><br>
                <b>📐 Control Points (1:2 ratio):</b><br>
                <b>P1 = ⅔·p_c + ⅓·pl_n:</b> [{ctrl_orig[1,0]:.4f}, {ctrl_orig[1,1]:.4f}, {ctrl_orig[1,2]:.4f}]<br>
                <b>P2 = ⅓·p_c + ⅔·pr_m:</b> [{ctrl_orig[2,0]:.4f}, {ctrl_orig[2,1]:.4f}, {ctrl_orig[2,2]:.4f}]<br>
                <b>Slice Points:</b> {len(slice_pts_pca)} &nbsp;|&nbsp;
                <b>Curve Points:</b> {n_curve_pts}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("กรุณาทำขั้นตอน Tab 02 ก่อน")

# =====================================================================
# TAB 4: RECONSTRUCTION RESULT
# =====================================================================
with tab_result:
    result = st.session_state.get("hf_result")
    if result:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            st.markdown('<p class="step-header">การฟื้นฟูพื้นผิว</p>', unsafe_allow_html=True)

            st.subheader("🔄 Cross-Hatching Logic")
            show_axis1 = st.checkbox(f"แสดงแกน {result['axis_1']} (Primary)", value=True)
            show_axis2 = st.checkbox(f"แสดงแกน {result['axis_2']} (Secondary)", value=True)

            st.markdown("---")
            progress = st.slider("Reconstruction Step", 0, 100, 100)

            st.markdown("---")
            st.markdown("### 📥 ส่งออกข้อมูล")
            out_buf = io.StringIO()
            np.savetxt(out_buf, result['merged_points'], fmt='%.8f %.8f %.8f')
            st.download_button("Download Reconstructed XYZ", data=out_buf.getvalue(),
                               file_name=f"{selected_file.split('.')[0]}_final.xyz",
                               use_container_width=True)

        with col_f2:
            orig = result['original_inlier_points']
            fill_1 = result['bezier_pts_axis1'] if show_axis1 else np.empty((0,3))
            fill_2 = result['bezier_pts_axis2'] if show_axis2 else np.empty((0,3))
            
            n_show_1 = int(len(fill_1) * (progress / 100.0))
            n_show_2 = int(len(fill_2) * (progress / 100.0))
            active_1 = fill_1[:n_show_1]
            active_2 = fill_2[:n_show_2]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter3d(
                x=orig[:, 0], y=orig[:, 1], z=orig[:, 2],
                mode='markers', marker=dict(size=1.5, color='#CBD5E1', opacity=0.3), name="Original"
            ))
            if len(active_1) > 0:
                fig3.add_trace(go.Scatter3d(
                    x=active_1[:, 0], y=active_1[:, 1], z=active_1[:, 2],
                    mode='markers', marker=dict(size=2.5, color='#3B82F6', opacity=0.9), name="Primary Axis"
                ))
            if len(active_2) > 0:
                fig3.add_trace(go.Scatter3d(
                    x=active_2[:, 0], y=active_2[:, 1], z=active_2[:, 2],
                    mode='markers', marker=dict(size=2.5, color='#F59E0B', opacity=0.9), name="Secondary (Loft)"
                ))
            fig3.update_layout(
                scene=dict(aspectmode='data', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False)),
                margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white", height=700,
                legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig3, use_container_width=True)
            if progress < 100:
                st.caption(f"แสดงผล: {progress}%")
    else:
        st.warning("กรุณาทำขั้นตอน Tab 02 ก่อน")

# =====================================================================
# TAB 5: METRICS — with Chamfer Distance
# =====================================================================
with tab_metrics:
    result = st.session_state.get("hf_result")
    if result:
        st.markdown('<p class="step-header">📐 Quantitative Evaluation Metrics</p>', unsafe_allow_html=True)

        # --- Pipeline metrics ---
        metrics_data = compute_all_metrics(result)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Gaps Detected", metrics_data.get('total_gaps_detected', 0))
        col2.metric("🔵 Fill Points", f"{metrics_data.get('total_fill_points', 0):,}")
        col3.metric("📈 Fill Rate", f"{metrics_data.get('fill_rate', 0):.1f}%")
        col4.metric("⏱️ Total Time", f"{result.get('timings', {}).get('total', 0):.3f}s")

        st.markdown("---")

        # --- Chamfer Distance against ground truth ---
        st.markdown("### 📏 Chamfer Distance (ค่าความถูกต้อง)")

        # Try to find matching ground truth file in before/
        gt_file = os.path.join(BEFORE_FOLDER, selected_file.replace("hole", "before"))
        # Also try exact same name
        gt_file_exact = os.path.join(BEFORE_FOLDER, selected_file)
        # Also try "before.xyz" if current is "hole.xyz"
        gt_file_default = os.path.join(BEFORE_FOLDER, "before.xyz")

        gt_path = None
        if os.path.exists(gt_file):
            gt_path = gt_file
        elif os.path.exists(gt_file_exact):
            gt_path = gt_file_exact
        elif os.path.exists(gt_file_default) and "hole" in selected_file.lower():
            gt_path = gt_file_default

        # Let user also manually select
        if before_files:
            manual_gt = st.selectbox("เลือกไฟล์ Ground Truth (before)", before_files, key="gt_select")
            gt_path = os.path.join(BEFORE_FOLDER, manual_gt)

        if gt_path and os.path.exists(gt_path):
            gt_pts = np.array(load_points_from_file(gt_path))
            merged_pts = result['merged_points']
            filled_pts = result.get('combined_bezier_pts', np.empty((0, 3)))

            # Chamfer: merged (repaired) vs before (ground truth)
            cd_merged = chamfer_distance(merged_pts, gt_pts)
            # Chamfer: hole (input) vs before (ground truth)
            cd_input = chamfer_distance(points_raw, gt_pts)

            st.markdown("#### เปรียบเทียบความถูกต้อง")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("CD (Input → GT)", f"{cd_input['symmetric']:.8f}",
                       help="Chamfer Distance ระหว่าง input (มีรู) กับ ground truth")
            cc2.metric("CD (Repaired → GT)", f"{cd_merged['symmetric']:.8f}",
                       help="Chamfer Distance ระหว่างผลซ่อม กับ ground truth")
            improvement = cd_input['symmetric'] - cd_merged['symmetric']
            cc3.metric("Improvement", f"{improvement:.8f}",
                       delta=f"{improvement:.8f}" if improvement > 0 else f"{improvement:.8f}",
                       delta_color="normal")

            st.markdown("#### รายละเอียด Chamfer Distance")
            cd_df = pd.DataFrame({
                'Comparison': ['Input (hole) → GT', 'Repaired → GT'],
                'CD Forward': [f"{cd_input['forward']:.8f}", f"{cd_merged['forward']:.8f}"],
                'CD Backward': [f"{cd_input['backward']:.8f}", f"{cd_merged['backward']:.8f}"],
                'CD Symmetric': [f"{cd_input['symmetric']:.8f}", f"{cd_merged['symmetric']:.8f}"],
                'Points': [f"{len(points_raw):,}", f"{len(merged_pts):,}"],
            })
            st.dataframe(cd_df, use_container_width=True, hide_index=True)

            # Bar chart comparison
            fig_cd = go.Figure(data=[
                go.Bar(name='Input (hole)', x=['Forward', 'Backward', 'Symmetric'],
                       y=[cd_input['forward'], cd_input['backward'], cd_input['symmetric']],
                       marker_color='#EF4444'),
                go.Bar(name='Repaired', x=['Forward', 'Backward', 'Symmetric'],
                       y=[cd_merged['forward'], cd_merged['backward'], cd_merged['symmetric']],
                       marker_color='#10B981'),
            ])
            fig_cd.update_layout(
                barmode='group', yaxis_title="Chamfer Distance",
                paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                height=350, margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_cd, use_container_width=True)
        else:
            st.info("ไม่พบไฟล์ Ground Truth ใน Dataset/before/ — ใส่ไฟล์ before.xyz เพื่อคำนวณ Chamfer Distance")

        st.markdown("---")

        # --- Surface Quality ---
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### 🔍 Surface Quality")
            quality_df = pd.DataFrame({
                'Metric': ['Surface Roughness (Mean)', 'Surface Roughness (Std)',
                           'Density Uniformity (CV)', 'Mean Point Spacing',
                           'Original Points', 'Merged Points'],
                'Value': [
                    f"{metrics_data.get('surface_roughness_mean', 0):.6f}",
                    f"{metrics_data.get('surface_roughness_std', 0):.6f}",
                    f"{metrics_data.get('density_uniformity_cv', 0):.4f}",
                    f"{metrics_data.get('mean_point_spacing', 0):.6f}",
                    f"{metrics_data.get('original_points', 0):,}",
                    f"{metrics_data.get('merged_points', 0):,}",
                ]
            })
            st.dataframe(quality_df, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("### ⏱️ Per-Step Timing")
            timings = result.get('timings', {})
            if timings:
                t_labels = [k.upper() for k in timings if k != 'total']
                t_values = [v for k, v in timings.items() if k != 'total']
                fig_t = go.Figure(data=[
                    go.Bar(x=t_labels, y=t_values,
                           marker_color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'][:len(t_labels)],
                           text=[f"{v:.4f}s" for v in t_values], textposition='auto')
                ])
                fig_t.update_layout(yaxis_title="Seconds", paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                                    height=350, margin=dict(l=40, r=20, t=20, b=40))
                st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.warning("กรุณาทำขั้นตอน Tab 02 ก่อน")
