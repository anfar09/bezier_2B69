import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import io

from slicer import load_points_from_file
from hole_filler import process_point_cloud, apply_inverse_pca, axis_col
from metrics import chamfer_distance, per_point_error, rmse, hausdorff_distance

# --- Page Config ---
st.set_page_config(
    page_title="3D Point Cloud Reconstruction — Research",
    page_icon="🔬", layout="wide", initial_sidebar_state="collapsed"
)

# --- Theme State ---
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def toggle_theme():
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"

is_dark = st.session_state["theme"] == "dark"

# --- Theme-aware Plotly defaults ---
def plotly_colors():
    if is_dark:
        return dict(
            paper="#0F172A", plot="#1E293B", grid="#334155",
            text="#E2E8F0", surface_bg="#1E293B",
            card="#1E293B", accent="#60A5FA"
        )
    return dict(
        paper="#FFFFFF", plot="#FAFBFC", grid="#E2E8F0",
        text="#1E293B", surface_bg="#FFFFFF",
        card="#FFFFFF", accent="#3B82F6"
    )

pc = plotly_colors()

def to_pca(pts, result):
    """Transform points from original space to PCA space."""
    return (np.asarray(pts) - result['mean_pt']) @ result['rotation_matrix']

def themed_3d_layout(**overrides):
    base = dict(
        scene=dict(
            aspectmode='data',
            xaxis=dict(gridcolor=pc['grid'], backgroundcolor=pc['surface_bg'],
                       color=pc['text'], showbackground=True),
            yaxis=dict(gridcolor=pc['grid'], backgroundcolor=pc['surface_bg'],
                       color=pc['text'], showbackground=True),
            zaxis=dict(gridcolor=pc['grid'], backgroundcolor=pc['surface_bg'],
                       color=pc['text'], showbackground=True),
        ),
        paper_bgcolor=pc['paper'], height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color=pc['text']),
        legend=dict(font=dict(color=pc['text']))
    )
    base.update(overrides)
    return base

def themed_2d_layout(**overrides):
    base = dict(
        paper_bgcolor=pc['paper'], plot_bgcolor=pc['plot'],
        font=dict(color=pc['text']),
        xaxis=dict(gridcolor=pc['grid'], color=pc['text']),
        yaxis=dict(gridcolor=pc['grid'], color=pc['text']),
    )
    base.update(overrides)
    return base

# --- CSS Theme System ---
THEME_CSS = {
    "dark": """
    :root {
        --bg-primary: #0F172A; --bg-card: #1E293B; --bg-surface: #334155;
        --text-primary: #F1F5F9; --text-secondary: #94A3B8; --text-muted: #64748B;
        --accent: #60A5FA; --accent-rgb: 96,165,250; --accent-hover: #93C5FD;
        --border: #334155; --border-light: #475569;
        --success: #34D399; --danger: #F87171; --warning: #FBBF24;
        --gradient-start: #1E293B; --gradient-end: #0F172A;
        --shadow: rgba(0,0,0,0.4); --glow: rgba(96,165,250,0.15);
        --metric-border: #3B82F6;
    }
    """,
    "light": """
    :root {
        --bg-primary: #F8FAFC; --bg-card: #FFFFFF; --bg-surface: #F1F5F9;
        --text-primary: #0F172A; --text-secondary: #64748B; --text-muted: #94A3B8;
        --accent: #3B82F6; --accent-rgb: 59,130,246; --accent-hover: #2563EB;
        --border: #E2E8F0; --border-light: #CBD5E1;
        --success: #10B981; --danger: #EF4444; --warning: #F59E0B;
        --gradient-start: #FFFFFF; --gradient-end: #F8FAFC;
        --shadow: rgba(0,0,0,0.08); --glow: rgba(59,130,246,0.08);
        --metric-border: #3B82F6;
    }
    """
}

COMMON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sarabun:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Sarabun', sans-serif !important;
}

.main { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }
.block-container { padding-top: 1.5rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: transparent; flex-wrap: wrap;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: 44px; background: transparent; border-radius: 8px 8px 0 0;
    padding: 0 20px; color: var(--text-secondary);
    border: none; font-weight: 500; font-size: 0.85rem;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--accent); background: var(--glow);
}
.stTabs [aria-selected="true"] {
    background: var(--glow) !important;
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    font-weight: 600;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, var(--gradient-start), var(--bg-card));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 24px var(--shadow);
}
.app-header .title {
    font-size: 1.8rem; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px; margin: 0;
    background: linear-gradient(135deg, var(--accent), var(--accent-hover));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-header .subtitle {
    font-size: 0.9rem; color: var(--text-secondary); margin: 4px 0 0 0;
}

/* Cards */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 4px 16px var(--shadow);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    margin-bottom: 14px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px var(--shadow);
}

.method-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 12px var(--shadow);
    margin-bottom: 14px;
}

/* Step headers */
.step-header {
    font-size: 1.2rem; color: var(--text-primary);
    border-left: 4px solid var(--accent);
    padding-left: 14px;
    margin: 20px 0 12px 0;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--metric-border);
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 12px var(--shadow);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-card .metric-label {
    font-size: 0.75rem; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 6px; font-weight: 500;
}
.metric-card .metric-value {
    font-size: 1.5rem; font-weight: 700;
    color: var(--text-primary);
}
.metric-card .metric-delta {
    font-size: 0.8rem; margin-top: 4px; font-weight: 500;
}
.metric-card .metric-delta.positive { color: var(--success); }
.metric-card .metric-delta.negative { color: var(--danger); }

.metric-card.accent-blue   { --metric-border: #3B82F6; }
.metric-card.accent-green  { --metric-border: #10B981; }
.metric-card.accent-orange { --metric-border: #F59E0B; }
.metric-card.accent-red    { --metric-border: #EF4444; }
.metric-card.accent-purple { --metric-border: #8B5CF6; }

/* Buttons */
.stButton>button {
    background: transparent; color: var(--accent);
    border: 2px solid var(--accent);
    padding: 8px 24px; border-radius: 10px;
    font-weight: 600; transition: all 0.2s ease;
}
.stButton>button:hover {
    background: var(--accent); color: white;
    box-shadow: 0 8px 20px rgba(var(--accent-rgb), 0.3);
}

/* Streamlit overrides */
.stMetric { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.stMetric label { color: var(--text-secondary) !important; }
.stMetric [data-testid="stMetricValue"] { color: var(--text-primary) !important; }
.stSelectbox label, .stTextInput label, .stSlider label, .stRadio label, .stCheckbox label {
    color: var(--text-primary) !important;
}
.stMarkdown, .stMarkdown p, .stWrite { color: var(--text-primary) !important; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; }
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Section dividers */
.section-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 24px 0;
}

/* Hide Streamlit deploy toolbar */
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 1rem !important; }
"""

st.markdown(f"<style>{THEME_CSS[st.session_state['theme']]}{COMMON_CSS}</style>", unsafe_allow_html=True)

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
col_h1, col_h2, col_h3 = st.columns([3, 1, 0.3])
with col_h1:
    st.markdown("""
    <div class="app-header">
        <div>
            <p class="title">3D Reconstruction Analysis</p>
            <p class="subtitle">การเติมเต็มพื้นผิวและซ่อมแซมรูโหว่ด้วย Bezier Cross-Hatching</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
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

with col_h3:
    theme_icon = "☀️" if is_dark else "🌙"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle", help="Toggle Light/Dark Theme")

# =====================================================================
# TABS
# =====================================================================
tab_intro, tab_analysis, tab_deepdive, tab_result, tab_metrics = st.tabs([
    "📍 01. บทนำ",
    "⚙️ 02. ประมวลผล",
    "🔗 03. การเชื่อมต่อ",
    "💎 04. ผลลัพธ์",
    "📐 05. Chamfer Distance",
])

# =====================================================================
# TAB 1: INTRODUCTION
# =====================================================================
with tab_intro:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown('<p class="step-header">ภาพรวมของโครงการ</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card">
        <b>🔬 3D Surface Repair</b><br><br>
        การซ่อมแซมพื้นผิว 3 มิติ เป็นส่วนสำคัญในงานอุตสาหกรรมสมัยใหม่
        ช่วยให้สามารถกู้คืนโมเดลที่เสียหายจากการสแกน<br><br>
        <b>📋 ขั้นตอนหลัก:</b><br>
        • <b>Data Cleaning:</b> การลบจุดรบกวน (SOR)<br>
        • <b>Geometric Alignment:</b> การจัดวาง PCA<br>
        • <b>Surface Generation:</b> Bezier Cross-Hatching + Surface Densification
        </div>
        """, unsafe_allow_html=True)
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
        fig1.update_layout(**themed_3d_layout())
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
        <div class="method-card"><b>3. Surface Densification</b><br>เติมจุดระหว่าง Bezier curves เพื่อสร้างพื้นผิวสมจริง</div>
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

            coord_mode_2 = st.radio("🔄 ระบบพิกัด", ["Original", "PCA"], horizontal=True, key="coord_tab2")
            use_pca_2 = coord_mode_2 == "PCA"

            orig = to_pca(result['original_inlier_points'], result) if use_pca_2 else result['original_inlier_points']
            bound_raw = result['combined_boundary_pts']
            bound = to_pca(bound_raw, result) if (use_pca_2 and len(bound_raw) > 0) else bound_raw

            axis_labels = dict(xaxis_title="PCA-X", yaxis_title="PCA-Y", zaxis_title="PCA-Z") if use_pca_2 else {}
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
            layout_2 = themed_3d_layout()
            if axis_labels:
                layout_2['scene'].update(axis_labels)
            fig2.update_layout(**layout_2)
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

                if "Axis X" in selected_slice: slice_axis = 'X'
                elif "Axis Y" in selected_slice: slice_axis = 'Y'
                else: slice_axis = 'Z'

                pts_pca = result['pts_clean_pca']
                rot_mat = result['rotation_matrix']
                mean_pt = result['mean_pt']
                thickness = result['slice_thickness']

                col_idx = axis_col(slice_axis)
                vals = pts_pca[:, col_idx]
                half_t = thickness / 2.0
                slice_mask = (vals >= axis_val - half_t) & (vals < axis_val + half_t)
                slice_pts_pca = pts_pca[slice_mask]

                axes_3d = [0, 1, 2]
                axes_3d.remove(col_idx)
                ax_h, ax_v = axes_3d[0], axes_3d[1]
                axis_names = ['PCA-X', 'PCA-Y', 'PCA-Z']

                # === TANGENT VIA LINEAR REGRESSION (polyfit) ===
                p_l_2d = p_left_pca[[ax_h, ax_v]]
                p_r_2d = p_right_pca[[ax_h, ax_v]]

                slice_2d_all = slice_pts_pca[:, [ax_h, ax_v]]
                sort_order = np.argsort(slice_2d_all[:, 0])
                sorted_2d = slice_2d_all[sort_order]

                dists_to_p0 = np.linalg.norm(sorted_2d - p_l_2d, axis=1)
                p0_idx = int(np.argmin(dists_to_p0))

                dists_to_p3 = np.linalg.norm(sorted_2d - p_r_2d, axis=1)
                p3_idx = int(np.argmin(dists_to_p3))

                n_adj = 5

                k_left = min(n_adj, p0_idx)
                if k_left >= 2:
                    left_pts = sorted_2d[p0_idx - k_left : p0_idx + 1]
                    coeffs_l = np.polyfit(left_pts[:, 0], left_pts[:, 1], deg=1)
                    slope_l = coeffs_l[0]
                    v_left = np.array([1.0, slope_l])
                    v_left = v_left / np.linalg.norm(v_left)
                elif k_left == 1:
                    adj = sorted_2d[p0_idx - 1]
                    v_left = p_l_2d - adj
                    nl = np.linalg.norm(v_left)
                    v_left = v_left / nl if nl > 1e-12 else np.array([1.0, 0.0])
                else:
                    v_left = np.array([1.0, 0.0])

                k_right = min(n_adj, len(sorted_2d) - 1 - p3_idx)
                if k_right >= 2:
                    right_pts = sorted_2d[p3_idx : p3_idx + k_right + 1]
                    coeffs_r = np.polyfit(right_pts[:, 0], right_pts[:, 1], deg=1)
                    slope_r = coeffs_r[0]
                    v_right = np.array([-1.0, -slope_r])
                    v_right = v_right / np.linalg.norm(v_right)
                elif k_right == 1:
                    adj = sorted_2d[p3_idx + 1]
                    v_right = p_r_2d - adj
                    nr = np.linalg.norm(v_right)
                    v_right = v_right / nr if nr > 1e-12 else np.array([-1.0, 0.0])
                else:
                    v_right = np.array([-1.0, 0.0])

                # === HERMITE-TO-BEZIER CONTROL POINTS ===
                hermite_scale = gap_dist / 3.0

                P0 = p_l_2d
                P1 = P0 + hermite_scale * v_left
                P2 = p_r_2d + hermite_scale * v_right
                P3 = p_r_2d

                dir_01 = P1 - P0
                dir_32 = P2 - P3
                try:
                    A = np.column_stack([dir_01, -dir_32])
                    ts = np.linalg.solve(A, P3 - P0)
                    p_c_2d = P0 + ts[0] * dir_01
                except np.linalg.LinAlgError:
                    p_c_2d = (P1 + P2) / 2.0

                arrow_scale = gap_dist * 0.6
                v_left_scaled = v_left * arrow_scale
                v_right_scaled = v_right * arrow_scale

                # Compute Bezier curve
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

                if len(slice_pts_pca) > 0:
                    fig.add_trace(go.Scatter(
                        x=slice_pts_pca[:, ax_h], y=slice_pts_pca[:, ax_v],
                        mode='markers', marker=dict(size=5, color=pc['accent'], opacity=0.7),
                        name=f"Slice Points ({len(slice_pts_pca)})"
                    ))

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
                    fig.add_trace(go.Scatter(
                        x=[P0[0], P3[0]], y=[P0[1], P3[1]],
                        mode='lines', line=dict(color='#EF4444', width=2, dash='dash'),
                        name=f"Gap (dist={gap_dist:.4f})"
                    ))
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

                if "③" in step or "④" in step:
                    fig.add_trace(go.Scatter(
                        x=[P0[0], P1[0]], y=[P0[1], P1[1]],
                        mode='lines', line=dict(color='#F59E0B', width=2, dash='dot'),
                        name="Control Arm (P0→P1)"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[P3[0], P2[0]], y=[P3[1], P2[1]],
                        mode='lines', line=dict(color='#F59E0B', width=2, dash='dot'),
                        name="Control Arm (P3→P2)"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[P1[0], P2[0]], y=[P1[1], P2[1]],
                        mode='markers+text',
                        marker=dict(size=12, color='#F97316', symbol='cross',
                                    line=dict(width=1, color='#7C2D12')),
                        text=['P1 (Hermite)', 'P2 (Hermite)'],
                        textposition='bottom center',
                        textfont=dict(size=10, color='#F97316'),
                        name="Control Points (Hermite)"
                    ))

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
                    xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor=pc['grid']),
                    yaxis=dict(gridcolor=pc['grid']),
                    paper_bgcolor=pc['paper'], plot_bgcolor=pc['plot'],
                    font=dict(color=pc['text']),
                    height=550, margin=dict(l=60, r=20, t=30, b=60),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5,
                                font=dict(color=pc['text'])),
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
                <b>📝 Hermite-to-Bezier Construction</b><br>
                <b>Slice:</b> {slice_axis}={axis_val:.4f} &nbsp;|&nbsp;
                <b>Gap:</b> {gap_dist:.4f} &nbsp;|&nbsp;
                <b>Hermite Scale:</b> {hermite_scale:.4f}<br><br>
                <b>🔹 Boundary Points:</b><br>
                <b>P0 (pl_n):</b> [{p_left_orig[0]:.4f}, {p_left_orig[1]:.4f}, {p_left_orig[2]:.4f}]<br>
                <b>P3 (pr_m):</b> [{p_right_orig[0]:.4f}, {p_right_orig[1]:.4f}, {p_right_orig[2]:.4f}]<br><br>
                <b>📐 Control Points (Hermite: P + gap/3 · tangent):</b><br>
                <b>P1 = P0 + (gap/3)·v_left:</b> [{ctrl_orig[1,0]:.4f}, {ctrl_orig[1,1]:.4f}, {ctrl_orig[1,2]:.4f}]<br>
                <b>P2 = P3 + (gap/3)·v_right:</b> [{ctrl_orig[2,0]:.4f}, {ctrl_orig[2,1]:.4f}, {ctrl_orig[2,2]:.4f}]<br>
                <b>Tangent method:</b> Linear Regression (polyfit, {n_adj} pts) &nbsp;|&nbsp;
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
            coord_mode_4 = st.radio("🔄 ระบบพิกัด", ["Original", "PCA"], horizontal=True, key="coord_tab4")
            progress = st.slider("Reconstruction Step", 0, 100, 100)

            st.markdown("---")
            st.markdown("### 📥 ส่งออกข้อมูล")
            out_buf = io.StringIO()
            np.savetxt(out_buf, result['merged_points'], fmt='%.8f %.8f %.8f')
            st.download_button("Download Reconstructed XYZ", data=out_buf.getvalue(),
                               file_name=f"{selected_file.split('.')[0]}_final.xyz",
                               use_container_width=True)

        with col_f2:
            use_pca_4 = coord_mode_4 == "PCA"

            orig_raw = result['original_inlier_points']
            fill_1_raw = result['bezier_pts_axis1'] if show_axis1 else np.empty((0,3))
            fill_2_raw = result['bezier_pts_axis2'] if show_axis2 else np.empty((0,3))

            orig = to_pca(orig_raw, result) if use_pca_4 else orig_raw
            fill_1 = to_pca(fill_1_raw, result) if (use_pca_4 and len(fill_1_raw) > 0) else fill_1_raw
            fill_2 = to_pca(fill_2_raw, result) if (use_pca_4 and len(fill_2_raw) > 0) else fill_2_raw

            n_show_1 = int(len(fill_1) * (progress / 100.0))
            n_show_2 = int(len(fill_2) * (progress / 100.0))
            active_1 = fill_1[:n_show_1]
            active_2 = fill_2[:n_show_2]

            axis_labels_4 = dict(xaxis_title="PCA-X", yaxis_title="PCA-Y", zaxis_title="PCA-Z") if use_pca_4 else {}
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
            layout_4 = themed_3d_layout(
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=1,
                            font=dict(color=pc['text']))
            )
            if axis_labels_4:
                layout_4['scene'].update(axis_labels_4)
            fig3.update_layout(**layout_4)
            st.plotly_chart(fig3, use_container_width=True)
            if progress < 100:
                st.caption(f"แสดงผล: {progress}%")
    else:
        st.warning("กรุณาทำขั้นตอน Tab 02 ก่อน")

# =====================================================================
# TAB 5: CHAMFER DISTANCE — 3D COMPARISON & METRICS
# =====================================================================
with tab_metrics:
    result = st.session_state.get("hf_result")
    if result:
        st.markdown('<p class="step-header">📐 3D Chamfer Distance Analysis</p>', unsafe_allow_html=True)

        # --- Ground Truth selection ---
        gt_path = None
        if before_files:
            manual_gt = st.selectbox("🗂️ เลือกไฟล์ Ground Truth (before)", before_files, key="gt_select")
            gt_path = os.path.join(BEFORE_FOLDER, manual_gt)
        else:
            st.info("ไม่พบไฟล์ Ground Truth ใน Dataset/before/ — ใส่ไฟล์เพื่อคำนวณ Chamfer Distance")

        if gt_path and os.path.exists(gt_path):
            gt_pts = np.array(load_points_from_file(gt_path))
            merged_pts = result['merged_points']
            filled_pts = result.get('combined_bezier_pts', np.empty((0, 3)))

            # === COMPUTE ALL METRICS ===
            cd_merged = chamfer_distance(merged_pts, gt_pts)
            cd_input = chamfer_distance(points_raw, gt_pts)
            hd_merged = hausdorff_distance(merged_pts, gt_pts)
            rmse_val = rmse(filled_pts, gt_pts) if len(filled_pts) > 0 else 0.0
            improvement = cd_input['symmetric'] - cd_merged['symmetric']
            imp_pct = (improvement / cd_input['symmetric'] * 100) if cd_input['symmetric'] > 1e-15 else 0

            # === ERROR RATE HERO SECTION ===
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # Compute error statistics early
            errors_merged = per_point_error(merged_pts, gt_pts)
            mean_err = float(np.mean(errors_merged))
            median_err = float(np.median(errors_merged))
            std_err = float(np.std(errors_merged))
            p95 = float(np.percentile(errors_merged, 95))
            p99_val = float(np.percentile(errors_merged, 99))
            max_err = float(np.max(errors_merged))

            # Error rate: % of fill points above a threshold
            # ใช้เฉพาะจุดที่ "สร้างขึ้นมาใหม่" (filled_pts) ไม่ให้มาผสมรวมกับจุดเดิมฐานที่ตรงอยู่แล้ว 100%
            if len(filled_pts) > 0:
                errors_filled = per_point_error(filled_pts, gt_pts)
                err_threshold = result.get('avg_point_spacing', 0.003) * 2.0
                n_high_err = int(np.sum(errors_filled > err_threshold))
                error_rate = (n_high_err / len(errors_filled)) * 100.0
                total_filled = len(errors_filled)
            else:
                err_threshold = result.get('avg_point_spacing', 0.003) * 2.0
                n_high_err = 0
                error_rate = 0.0
                total_filled = 0

            accuracy_rate = 100.0 - error_rate

            # Gauge color
            if accuracy_rate >= 90:
                gauge_color = "#10B981"
                gauge_label = "ดีมาก"
            elif accuracy_rate >= 70:
                gauge_color = "#FBBF24"
                gauge_label = "ปานกลาง"
            else:
                gauge_color = "#EF4444"
                gauge_label = "ต้องปรับปรุง"

            st.markdown(f"""
<div class="glass-card" style="text-align:center; padding:28px;">
    <div style="font-size:0.85rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
        🎯 ความแม่นยำในการซ่อมพื้นผิว (เฉพาะส่วนที่เติมใหม่)
    </div>
    <div style="font-size:3.5rem; font-weight:800; color:{gauge_color}; line-height:1.1;">
        {accuracy_rate:.1f}%
    </div>
    <div style="font-size:1rem; color:{gauge_color}; font-weight:600; margin:4px 0 12px 0;">
        {gauge_label}
    </div>
    <div style="font-size:0.8rem; color:var(--text-muted);">
        จุดเติมใหม่ที่ตรงตาม GT (error ≤ {err_threshold:.4f}) = <b>{total_filled - n_high_err:,}</b> จาก <b>{total_filled:,}</b> จุด
        &nbsp;|&nbsp; พื้นที่ส่วนเกินที่สร้างขึ้นผิดพลาด (Hallucinated) = <b style="color:#EF4444;">{n_high_err:,}</b> ({error_rate:.1f}%)
    </div>
</div>
            """, unsafe_allow_html=True)

            st.markdown("", unsafe_allow_html=True)

            # === METRIC CARDS WITH THAI DESCRIPTIONS ===
            c1, c2, c3, c4, c5 = st.columns(5)
            def metric_html(label, value, desc, accent="blue", delta=None, delta_positive=True):
                delta_html = ""
                if delta is not None:
                    cls = "positive" if delta_positive else "negative"
                    arrow = "▲" if delta_positive else "▼"
                    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta}</div>'
                return f'''<div class="metric-card accent-{accent}">
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
{delta_html}
<div style="font-size:0.7rem; color:var(--text-muted); margin-top:8px; line-height:1.4;">{desc}</div>
</div>'''

            c1.markdown(metric_html(
                "CD (Input→GT)", f"{cd_input['symmetric']:.6f}",
                "ระยะห่างเฉลี่ยยกกำลังสอง ระหว่าง input (มีรู) กับ ground truth — ยิ่งสูง ยิ่งต่างจากต้นฉบับมาก",
                "red"), unsafe_allow_html=True)
            c2.markdown(metric_html(
                "CD (Repaired→GT)", f"{cd_merged['symmetric']:.6f}",
                "ระยะห่างเฉลี่ยยกกำลังสอง ระหว่างผลซ่อม กับ ground truth — ยิ่งต่ำ ยิ่งใกล้ต้นฉบับ",
                "green"), unsafe_allow_html=True)
            c3.markdown(metric_html(
                "ดีขึ้น", f"{imp_pct:.1f}%",
                "CD ลดลงกี่ % หลังซ่อม — ค่าบวก = ดีขึ้น, ค่าลบ = แย่ลง",
                "blue", delta=f"{improvement:.6f}", delta_positive=improvement > 0), unsafe_allow_html=True)
            c4.markdown(metric_html(
                "RMSE (จุดเติม)", f"{rmse_val:.6f}",
                "ค่าเฉลี่ยราก (Root Mean Square) ของระยะห่างจุดที่เติม ไปยัง GT ที่ใกล้ที่สุด",
                "orange"), unsafe_allow_html=True)
            c5.markdown(metric_html(
                "Hausdorff", f"{hd_merged['symmetric']:.6f}",
                "ระยะห่างสูงสุดที่เลวร้ายที่สุด — จุดที่ไกลจาก GT มากที่สุด (worst-case error)",
                "purple"), unsafe_allow_html=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # === 3D ERROR HEATMAP ===
            st.markdown("### 🌡️ แผนที่ความร้อน 3D — ระยะห่างแต่ละจุดจาก Ground Truth")
            st.caption("จุดซ่อมถูกลงสีตามระยะห่างจาก Ground Truth (เขียว=ใกล้=ดี, แดง=ไกล=ไม่ดี)")

            coord_mode_5 = st.radio("🔄 ระบบพิกัด", ["Original", "PCA"], horizontal=True, key="coord_tab5")
            use_pca_5 = coord_mode_5 == "PCA"

            display_merged = to_pca(merged_pts, result) if use_pca_5 else merged_pts
            display_gt = to_pca(gt_pts, result) if use_pca_5 else gt_pts
            axis_labels_5 = dict(xaxis_title="PCA-X", yaxis_title="PCA-Y", zaxis_title="PCA-Z") if use_pca_5 else {}

            col_heat, col_gt = st.columns(2)

            with col_heat:
                st.markdown("**ผลซ่อม — Error Heatmap**")
                p99_clamp = np.percentile(errors_merged, 99) if len(errors_merged) > 0 else 1.0

                fig_heat = go.Figure(data=[go.Scatter3d(
                    x=display_merged[:, 0], y=display_merged[:, 1], z=display_merged[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2.0, color=errors_merged,
                        colorscale=[[0, '#10B981'], [0.3, '#FBBF24'], [0.6, '#F97316'], [1.0, '#EF4444']],
                        cmin=0, cmax=p99_clamp,
                        colorbar=dict(
                            title=dict(text="ระยะห่าง", font=dict(color=pc['text'])),
                            tickfont=dict(color=pc['text']),
                            len=0.6, thickness=15,
                        ),
                        opacity=0.9,
                    ),
                    name="Error",
                    hovertemplate="x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<br>Error: %{marker.color:.6f}<extra></extra>"
                )])
                layout_heat = themed_3d_layout(height=550)
                if axis_labels_5:
                    layout_heat['scene'].update(axis_labels_5)
                fig_heat.update_layout(**layout_heat)
                st.plotly_chart(fig_heat, use_container_width=True)

            with col_gt:
                st.markdown("**ต้นฉบับ — Ground Truth (Before)**")
                fig_gt = go.Figure(data=[go.Scatter3d(
                    x=display_gt[:, 0], y=display_gt[:, 1], z=display_gt[:, 2],
                    mode='markers',
                    marker=dict(size=1.5, color='#60A5FA', opacity=0.5),
                    name="Ground Truth"
                )])
                layout_gt = themed_3d_layout(height=550)
                if axis_labels_5:
                    layout_gt['scene'].update(axis_labels_5)
                fig_gt.update_layout(**layout_gt)
                st.plotly_chart(fig_gt, use_container_width=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # === ERROR DISTRIBUTION + STATS ===
            col_hist, col_stats = st.columns([1.5, 1])

            with col_hist:
                st.markdown("### 📊 การกระจายตัวของ Error")
                st.caption("แสดงจำนวนจุดที่มี error ในแต่ละช่วง — กองซ้ายมาก = ดี (error ต่ำ)")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=errors_merged, nbinsx=80,
                    marker_color=pc['accent'], opacity=0.85,
                    name="Per-point error"
                ))
                fig_hist.add_vline(x=mean_err, line_dash="dash", line_color="#F59E0B",
                                  annotation_text=f"ค่าเฉลี่ย: {mean_err:.6f}",
                                  annotation_font=dict(color="#F59E0B"))
                fig_hist.add_vline(x=median_err, line_dash="dash", line_color="#10B981",
                                  annotation_text=f"มัธยฐาน: {median_err:.6f}",
                                  annotation_font=dict(color="#10B981"))
                fig_hist.add_vline(x=err_threshold, line_dash="dot", line_color="#EF4444",
                                  annotation_text=f"Threshold: {err_threshold:.4f}",
                                  annotation_font=dict(color="#EF4444"))
                fig_hist.update_layout(
                    **themed_2d_layout(),
                    xaxis_title="ระยะห่างจาก GT", yaxis_title="จำนวนจุด",
                    height=400, margin=dict(l=50, r=20, t=30, b=50),
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_stats:
                st.markdown("### 📋 สถิติ Error")
                stats_df = pd.DataFrame({
                    'สถิติ': [
                        'ค่าเฉลี่ย (Mean)', 'มัธยฐาน (Median)', 'ส่วนเบี่ยงเบน (Std)',
                        'Percentile 95', 'Percentile 99', 'สูงสุด (Max)',
                        '──────────',
                        'Accuracy Rate', 'Error Rate',
                        '──────────',
                        'จุด Input', 'จุดหลังซ่อม', 'จุด Ground Truth', 'จุดที่เติม',
                    ],
                    'ค่า': [
                        f"{mean_err:.8f}", f"{median_err:.8f}", f"{std_err:.8f}",
                        f"{p95:.8f}", f"{p99_val:.8f}", f"{max_err:.8f}",
                        '──────────',
                        f"✅ {accuracy_rate:.1f}%", f"❌ {error_rate:.1f}%",
                        '──────────',
                        f"{len(points_raw):,}", f"{len(merged_pts):,}",
                        f"{len(gt_pts):,}", f"{len(filled_pts):,}",
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # === CHAMFER DISTANCE DETAIL + BAR CHART ===
            col_table, col_bar = st.columns([1, 1])

            with col_table:
                st.markdown("### 📏 Chamfer Distance — รายละเอียด")
                st.caption("""
                **Forward** = แต่ละจุดใน A หาจุดที่ใกล้ที่สุดใน B → เฉลี่ย |
                **Backward** = กลับกัน |
                **Symmetric** = รวมทั้งสองทิศทาง
                """)
                cd_df = pd.DataFrame({
                    'ตัวชี้วัด': [
                        'CD Forward (A→B)', 'CD Backward (B→A)', 'CD Symmetric (รวม)',
                        'Hausdorff Forward', 'Hausdorff Backward', 'Hausdorff Symmetric'],
                    'Input → GT': [
                        f"{cd_input['forward']:.8f}", f"{cd_input['backward']:.8f}", f"{cd_input['symmetric']:.8f}",
                        "—", "—", "—"
                    ],
                    'Repaired → GT': [
                        f"{cd_merged['forward']:.8f}", f"{cd_merged['backward']:.8f}", f"{cd_merged['symmetric']:.8f}",
                        f"{hd_merged['forward']:.8f}", f"{hd_merged['backward']:.8f}", f"{hd_merged['symmetric']:.8f}",
                    ],
                })
                st.dataframe(cd_df, use_container_width=True, hide_index=True)

            with col_bar:
                st.markdown("### 📊 เปรียบเทียบ Input vs Repaired")
                fig_cd = go.Figure(data=[
                    go.Bar(name='ก่อนซ่อม (Input)', x=['Forward', 'Backward', 'Symmetric'],
                           y=[cd_input['forward'], cd_input['backward'], cd_input['symmetric']],
                           marker_color='#EF4444'),
                    go.Bar(name='หลังซ่อม (Repaired)', x=['Forward', 'Backward', 'Symmetric'],
                           y=[cd_merged['forward'], cd_merged['backward'], cd_merged['symmetric']],
                           marker_color='#10B981'),
                ])
                fig_cd.update_layout(
                    **themed_2d_layout(),
                    barmode='group', yaxis_title="Chamfer Distance",
                    height=350, margin=dict(l=50, r=20, t=20, b=50),
                    legend=dict(font=dict(color=pc['text']))
                )
                st.plotly_chart(fig_cd, use_container_width=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # === METRIC EXPLANATION ===
            st.markdown("### 📖 อธิบายตัวชี้วัด")
            ex1, ex2 = st.columns(2)
            with ex1:
                st.markdown("""
                <div class="method-card">
                <b>📏 Chamfer Distance (CD)</b><br>
                วัดว่า point cloud สองชุดใกล้กันแค่ไหน โดยหาจุดที่ใกล้ที่สุดของแต่ละจุด แล้วเฉลี่ยระยะทาง²<br><br>
                <b>สูตร:</b> CD = (1/|A|) Σ min‖a−b‖² + (1/|B|) Σ min‖b−a‖²<br>
                <b>ค่าดี:</b> ยิ่ง <b>ต่ำ</b> ยิ่งดี (0 = เหมือนกันทุกจุด)<br>
                <b>หน่วย:</b> ระยะทาง² (เช่น mm²)
                </div>

                <div class="method-card">
                <b>📐 RMSE (Root Mean Square Error)</b><br>
                วัดค่าเฉลี่ยของ "ความผิดพลาด" ของจุดที่เติมใหม่ เทียบกับพื้นผิวต้นฉบับ<br><br>
                <b>สูตร:</b> RMSE = √(Σ dᵢ² / N)<br>
                <b>ค่าดี:</b> ยิ่ง <b>ต่ำ</b> ยิ่งดี<br>
                <b>ใช้ดู:</b> จุดที่เราสร้างใหม่อยู่ใกล้พื้นผิวจริงแค่ไหน
                </div>
                """, unsafe_allow_html=True)
            with ex2:
                st.markdown(f"""
                <div class="method-card">
                <b>🎯 Accuracy / Error Rate</b><br>
                เปอร์เซ็นต์ของจุดที่มี error ≤ threshold (2× avg spacing = {err_threshold:.4f})<br><br>
                <b>Accuracy:</b> จุดที่อยู่ใกล้ GT พอ → ✅ {accuracy_rate:.1f}%<br>
                <b>Error Rate:</b> จุดที่ไกลเกิน threshold → ❌ {error_rate:.1f}%<br>
                <b>ค่าดี:</b> Accuracy ยิ่ง <b>สูง</b> ยิ่งดี
                </div>

                <div class="method-card">
                <b>📏 Hausdorff Distance</b><br>
                วัด "กรณีเลวร้ายที่สุด" — จุดที่ไกลจาก GT มากที่สุด<br><br>
                <b>สูตร:</b> H = max(max min‖a−b‖, max min‖b−a‖)<br>
                <b>ค่าดี:</b> ยิ่ง <b>ต่ำ</b> ยิ่งดี<br>
                <b>ใช้ดู:</b> มีจุดหลุดออกไปไกลมากไหม (outlier detection)
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # === TIMING ===
            st.markdown("### ⏱️ เวลาประมวลผลแต่ละขั้นตอน")
            timings = result.get('timings', {})
            if timings:
                t_labels = [k.upper() for k in timings if k != 'total']
                t_values = [v for k, v in timings.items() if k != 'total']
                fig_t = go.Figure(data=[
                    go.Bar(x=t_labels, y=t_values,
                           marker_color=[pc['accent'], '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'][:len(t_labels)],
                           text=[f"{v:.4f}s" for v in t_values], textposition='auto',
                           textfont=dict(color=pc['text']))
                ])
                fig_t.update_layout(**themed_2d_layout(), yaxis_title="Seconds",
                                    height=300, margin=dict(l=50, r=20, t=20, b=50))
                st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.warning("ไม่พบไฟล์ Ground Truth — กรุณาเพิ่มไฟล์ใน Dataset/before/")
    else:
        st.warning("กรุณาทำขั้นตอน Tab 02 ก่อน")
