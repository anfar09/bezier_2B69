import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import io
import time

from slicer import load_points_from_file, get_bounds
from hole_filler import process_point_cloud

# --- Page Config ---
st.set_page_config(
    page_title="3D Point Cloud Reconstruction",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Soft & Professional Light Theme CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Sarabun', sans-serif;
    }
    
    .main {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    
    /* Soft Professional Palette: Sage, Slate, and Sky Blue */
    :root {
        --primary-soft: #3B82F6;
        --secondary-soft: #10B981;
        --bg-card: #FFFFFF;
        --text-main: #334155;
        --border-color: #E2E8F0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #FFFFFF;
        border-radius: 12px 12px 0px 0px;
        padding: 0px 30px;
        color: #64748B;
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: var(--primary-soft) !important;
        border-bottom: 3px solid var(--primary-soft) !important;
        box-shadow: 0px 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .presentation-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    
    .step-header {
        font-size: 1.4rem;
        color: #1E293B;
        border-left: 5px solid var(--primary-soft);
        padding-left: 15px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: 600;
    }

    .methodology-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
        margin-bottom: 20px;
    }

    .stMetric {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    
    .stButton>button {
        background: white;
        color: var(--primary-soft);
        border: 2px solid var(--primary-soft);
        padding: 8px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: var(--primary-soft);
        color: white;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }

    /* Info boxes styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        background-color: #F1F5F9;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
DATASET_FOLDER = "Dataset"
os.makedirs(DATASET_FOLDER, exist_ok=True)
db_files = sorted([f for f in os.listdir(DATASET_FOLDER) if f.endswith(('.xyz', '.txt', '.pts'))])

if "hf_result" not in st.session_state: st.session_state["hf_result"] = None
if "hf_file" not in st.session_state: st.session_state["hf_file"] = None

# --- Header ---
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown('<p class="presentation-title">3D Reconstruction Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B; font-size:1.1rem; font-weight:400;">การเติมเต็มพื้นผิวและซ่อมแซมรูโหว่ด้วยอัลกอริทึม Bezier Cross-Hatching</p>', unsafe_allow_html=True)

with col_t2:
    selected_file = st.selectbox("เลือกข้อมูลที่ต้องการนำเสนอ", db_files, label_visibility="collapsed")
    file_path = os.path.join(DATASET_FOLDER, selected_file)
    
    @st.cache_data
    def load_data(fp): return np.array(load_points_from_file(fp))
    
    if st.session_state["hf_file"] != selected_file:
        st.session_state["points_raw"] = load_data(file_path)
        st.session_state["hf_file"] = selected_file
        st.session_state["hf_result"] = None
    
    points_raw = st.session_state["points_raw"]

st.markdown("<br>", unsafe_allow_html=True)

tab_intro, tab_analysis, tab_result = st.tabs([
    "📍 01. บทนำและข้อมูลดิบ", 
    "⚙️ 02. ขั้นตอนการประมวลผล", 
    "💎 03. ผลลัพธ์และการฟื้นฟูพื้นผิว"
])

# --- TAB 1: INTRODUCTION ---
with tab_intro:
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.markdown('<p class="step-header">ภาพรวมของโครงการ</p>', unsafe_allow_html=True)
        st.write("""
        การซ่อมแซมพื้นผิว 3 มิติ (3D Surface Repair) เป็นส่วนสำคัญในงานอุตสาหกรรมสมัยใหม่ 
        ช่วยให้เราสามารถกู้คืนโมเดลที่เสียหายจากการสแกน หรือเพิ่มความละเอียดของโมเดลได้
        
        **หัวข้อหลักในการนำเสนอ:**
        - **Data Cleaning:** การลบจุดรบกวน (Noise)
        - **Geometric Alignment:** การจัดวางโมเดลตามแกนหลัก (PCA)
        - **Surface Generation:** การสร้างพื้นผิวใหม่ด้วย Cross-Hatching
        """)
        
        st.markdown('<p class="step-header">ตั้งค่าการแสดงผล</p>', unsafe_allow_html=True)
        p_size = st.slider("ขนาดจุด (Point Size)", 1.0, 5.0, 2.0)
        p_color = st.selectbox("โทนสี (Color Palette)", ["Blues", "Greens", "Viridis", "Cividis"])

    with col_r:
        fig1 = go.Figure(data=[go.Scatter3d(
            x=points_raw[:, 0], y=points_raw[:, 1], z=points_raw[:, 2],
            mode='markers',
            marker=dict(size=p_size, color=points_raw[:, 2], colorscale=p_color, opacity=0.6),
            name="Raw Input"
        )])
        fig1.update_layout(
            scene=dict(
                aspectmode='data', 
                xaxis=dict(gridcolor='#E2E8F0', backgroundcolor='white'),
                yaxis=dict(gridcolor='#E2E8F0', backgroundcolor='white'),
                zaxis=dict(gridcolor='#E2E8F0', backgroundcolor='white')
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white",
            height=600
        )
        st.plotly_chart(fig1, use_container_width=True)

# --- TAB 2: METHODOLOGY ---
with tab_analysis:
    col_m, col_p = st.columns([1, 2])
    
    with col_m:
        st.markdown('<p class="step-header">Methodology</p>', unsafe_allow_html=True)
        st.write("ขั้นตอนการวิเคราะห์จุดที่ขาดหายไป")
        
        st.markdown("""
        <div class="methodology-card">
        <b>1. Statistical Removal (SOR)</b><br>
        กรองจุดที่ลอยอยู่อย่างอิสระซึ่งไม่ใช่ส่วนของพื้นผิวจริง
        </div>
        <div class="methodology-card">
        <b>2. Axis Alignment (PCA)</b><br>
        คำนวณ Variance เพื่อหาแกนที่วัตถุแผ่ขยายมากที่สุด
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ ปรับแต่งพารามิเตอร์")
        s_th = st.text_input("Slice Thickness", placeholder="Auto (แนะนำ)", key="s_th_soft")
        g_th = st.text_input("Gap Threshold", placeholder="Auto (แนะนำ)", key="g_th_soft")
        
        if st.button("🔍 เริ่มการประมวลผล (Start Detection)", use_container_width=True):
            with st.spinner("กำลังวิเคราะห์โครงสร้างโมเดล..."):
                s_val = float(s_th) if s_th.strip() else None
                g_val = float(g_th) if g_th.strip() else None
                st.session_state["hf_result"] = process_point_cloud(
                    points_raw, slice_thickness=s_val, gap_threshold=g_val, verbose=False
                )

    with col_p:
        result = st.session_state.get("hf_result")
        if result:
            st.markdown('<p class="step-header">ผลการตรวจหาจุดโหว่ (Hole Mapping)</p>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("จุดที่ผ่านการกรอง", f"{len(result['original_inlier_points']):,}")
            m2.metric("จำนวนรูโหว่ที่พบ", f"{len(result['gaps_axis1']) + len(result['gaps_axis2'])}")
            m3.metric("ความหนาแน่นเฉลี่ย", f"{result['avg_point_spacing']:.4f}")
            
            orig = result['original_inlier_points']
            bound = result['combined_boundary_pts']
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter3d(
                x=orig[:, 0], y=orig[:, 1], z=orig[:, 2],
                mode='markers', marker=dict(size=1.5, color='#94A3B8', opacity=0.2), name="Base Surface"
            ))
            if len(bound) > 0:
                fig2.add_trace(go.Scatter3d(
                    x=bound[:, 0], y=bound[:, 1], z=bound[:, 2],
                    mode='markers', marker=dict(size=4, color='#EF4444', opacity=0.8, line=dict(width=1, color='white')),
                    name="Hole Boundary"
                ))
            fig2.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="white", height=600)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("กรุณากด 'Start Detection' เพื่อวิเคราะห์ข้อมูล")

# --- TAB 3: FINAL RECONSTRUCTION ---
with tab_result:
    result = st.session_state.get("hf_result")
    if result:
        col_f1, col_f2 = st.columns([1, 2])
        
        with col_f1:
            st.markdown('<p class="step-header">การฟื้นฟูพื้นผิวแบบละเอียด</p>', unsafe_allow_html=True)
            
            # --- Cross-Hatching Controls ---
            st.subheader("🔄 Cross-Hatching Logic")
            st.write("เลือกแสดงผลการสานเส้น (Cross-Hatch) แยกตามแกนหลัก")
            
            show_axis1 = st.checkbox(f"แสดงเส้นเชื่อมแกน {result['axis_1']} (Primary)", value=True)
            show_axis2 = st.checkbox(f"แสดงเส้นเชื่อมแกน {result['axis_2']} (Secondary)", value=True)
            
            # --- Animation Control ---
            st.markdown("---")
            st.subheader("🎞️ Simulation Progress")
            st.write("เลื่อนเพื่อแสดงขั้นตอนการเชื่อมจุดแบบ Smooth")
            progress = st.slider("ขั้นตอนการก่อร่าง (Reconstruction Step)", 0, 100, 100)
            
            st.markdown("---")
            st.markdown("### 📥 ส่งออกข้อมูล")
            out_buf = io.StringIO()
            np.savetxt(out_buf, result['merged_points'], fmt='%.8f %.8f %.8f')
            st.download_button("Download Reconstructed XYZ", data=out_buf.getvalue(), 
                             file_name=f"{selected_file.split('.')[0]}_final.xyz", use_container_width=True)

        with col_f2:
            orig = result['original_inlier_points']
            
            # Filter fill points based on axes and progress
            fill_list = []
            if show_axis1: fill_list.append(result['bezier_pts_axis1'])
            if show_axis2: fill_list.append(result['bezier_pts_axis2'])
            
            if fill_list:
                full_fill = np.vstack(fill_list)
                # Simple Animation: show points up to progress %
                n_show = int(len(full_fill) * (progress / 100.0))
                active_fill = full_fill[:n_show]
            else:
                active_fill = np.empty((0,3))
            
            fig3 = go.Figure()
            # Base Original
            fig3.add_trace(go.Scatter3d(
                x=orig[:, 0], y=orig[:, 1], z=orig[:, 2],
                mode='markers', marker=dict(size=1.5, color='#CBD5E1', opacity=0.3), name="Original Model"
            ))
            # Animated Reconstruction
            if len(active_fill) > 0:
                fig3.add_trace(go.Scatter3d(
                    x=active_fill[:, 0], y=active_fill[:, 1], z=active_fill[:, 2],
                    mode='markers', marker=dict(size=2.5, color='#3B82F6', opacity=0.9), name="Reconstructed"
                ))
            
            fig3.update_layout(
                scene=dict(aspectmode='data', xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False)),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="white",
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            if progress < 100:
                st.caption(f"แสดงผลการ Reconstruction: {progress}% (เลื่อน Slider ด้านซ้ายเพื่อดูการเชื่อมจุดแบบต่อเนื่อง)")
    else:
        st.warning("กรุณาทำขั้นตอนการวิเคราะห์ใน Tab 02 ให้เสร็จสิ้นก่อน")
