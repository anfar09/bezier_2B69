import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import io

from slicer import load_points_from_file, get_bounds
from hole_filler import process_point_cloud

st.set_page_config(layout="wide")
st.title("🔬 3D Point Cloud + Hole Detection & Filling")

# ============================================================
# Initialise session-state keys on first run
# ============================================================
if "hf_result" not in st.session_state:
    st.session_state["hf_result"] = None
if "hf_file" not in st.session_state:
    st.session_state["hf_file"]   = None

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["3D View", "Hole Detecter", "Hole Filling"])

# ============================================================
# TAB 1 – Original 3D viewer
# ============================================================
with tab1:
    DATASET_FOLDER = "Dataset"
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(DATASET_FOLDER)
             if f.endswith(('.txt', '.xyz', '.csv', '.pts'))]

    if not files:
        st.error("ใส่ไฟล์ใน Dataset ก่อน")
    else:
        file = st.selectbox("เลือกไฟล์", files)
        path = os.path.join(DATASET_FOLDER, file)

        @st.cache_data
        def load(fp):
            return load_points_from_file(fp)

        points = np.array(load(path))
        bounds  = get_bounds(points)

        col1, col2 = st.columns([3, 1])

        with col2:
            size  = st.slider("Point size", 1, 10, 2)
            axis  = st.selectbox("Axis", ["X", "Y", "Z"])
            fixed = st.checkbox("Fixed Mode")

            r = bounds[axis.lower()][1] - bounds[axis.lower()][0]
            r = max(r, 1e-6)

            w = st.number_input("Slice width", min_value=r/100,
                                value=max(r/5, r/100), step=r/100)

            n = int(np.ceil(r/w))

            if fixed:
                idx_slice = st.slider("Slice", 0, max(n-1, 0), 0)

        with col1:
            fig = go.Figure()

            axis_map = {"X": 0, "Y": 1, "Z": 2}
            idx = axis_map[axis]

            xmin, xmax = bounds['x']
            ymin, ymax = bounds['y']
            zmin, zmax = bounds['z']

            def draw_box(cur, nxt, color='red'):
                if axis == "X":
                    xs, ys, zs = [cur, nxt], [ymin, ymax], [zmin, zmax]
                elif axis == "Y":
                    xs, ys, zs = [xmin, xmax], [cur, nxt], [zmin, zmax]
                else:
                    xs, ys, zs = [xmin, xmax], [ymin, ymax], [cur, nxt]

                edges = [
                    ([xs[0], xs[1]], [ys[0], ys[0]], [zs[0], zs[0]]),
                    ([xs[0], xs[1]], [ys[1], ys[1]], [zs[0], zs[0]]),
                ]
                for ex, ey, ez in edges:
                    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines',
                                              line=dict(color=color, width=4),
                                              showlegend=False))

            if fixed:
                cur = bounds[axis.lower()][0] + idx_slice * w
                nxt = min(cur + w, bounds[axis.lower()][1])

                mask = (points[:, idx] >= cur) & (points[:, idx] < nxt)

                pts_in  = points[mask]
                pts_out = points[~mask]

                fig.add_trace(go.Scatter3d(
                    x=pts_out[:, 0], y=pts_out[:, 1], z=pts_out[:, 2],
                    mode='markers', marker=dict(size=size, color='gray', opacity=0.05)
                ))
                fig.add_trace(go.Scatter3d(
                    x=pts_in[:, 0], y=pts_in[:, 1], z=pts_in[:, 2],
                    mode='markers', marker=dict(size=size+1, color='red')
                ))
                draw_box(cur, nxt, 'yellow')
            else:
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(size=size, color=points[:, 2], colorscale='Viridis')
                ))

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2 – Hole Detecter (Configuration & Boundary Visualization)
# ============================================================
with tab2:
    st.subheader("📁 Select Point Cloud")
    DB_FOLDER = "Dataset"
    os.makedirs(DB_FOLDER, exist_ok=True)

    db_files = sorted([
        f for f in os.listdir(DB_FOLDER)
        if f.endswith(('.xyz', '.txt', '.pts'))
    ])

    if not db_files:
        st.error(f"No `.xyz` files found in the `{DB_FOLDER}/` folder.")
        st.stop()

    selected_file = st.selectbox("Choose a file from Dataset/", db_files, key="hf_file_select")
    file_path = os.path.join(DB_FOLDER, selected_file)

    @st.cache_data
    def load_points_for_hf(fp):
        return np.array(load_points_from_file(fp))

    points_raw = load_points_for_hf(file_path)
    st.info(f"**{len(points_raw)}** raw points loaded.")

    st.subheader("⚙️ Hyperparameters")
    col_slice, col_gap = st.columns(2)
    with col_slice:
        st.markdown("**slice_thickness** (auto if empty)")
        slice_text = st.text_input("slice_thickness", value="", placeholder="auto", key="st_in")
        slice_thickness = float(slice_text) if slice_text.strip() else None
    with col_gap:
        st.markdown("**gap_threshold** (auto if empty)")
        gap_text = st.text_input("gap_threshold", value="", placeholder="auto", key="gt_in")
        gap_threshold = float(gap_text) if gap_text.strip() else None

    col_np, col_nk = st.columns(2)
    with col_np:
        num_points = st.number_input("num_points_per_gap", min_value=3, max_value=100, value=20, key="np_in")
    with col_nk:
        neighbor_k = st.slider("neighbor_k", min_value=2, max_value=30, value=10, key="nk_in")

    run = st.button("🚀 Run Detection", type="primary")

    if run:
        with st.spinner("Detecting holes..."):
            result = process_point_cloud(
                points_raw,
                slice_thickness=slice_thickness,
                gap_threshold=gap_threshold,
                num_points_per_gap=num_points,
                neighbor_k=neighbor_k,
                verbose=False,
            )
        st.session_state["hf_result"] = result
        st.session_state["hf_file"] = selected_file

    result = st.session_state.get("hf_result")
    if result is not None:
        original_pts = result['original_inlier_points']
        boundary_pts = result['combined_boundary_pts']
        
        st.subheader("🌀 Hole Boundary Visualization")
        fig = go.Figure()
        if len(original_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=original_pts[:, 0], y=original_pts[:, 1], z=original_pts[:, 2],
                mode='markers',
                marker=dict(size=2, color='#4A90D9', opacity=0.3, line=dict(width=0)),
                name='Inliers',
            ))
        if len(boundary_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=boundary_pts[:, 0], y=boundary_pts[:, 1], z=boundary_pts[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', opacity=1.0, line=dict(width=1, color='white')),
                name=f'Hole Boundaries ({len(boundary_pts)})',
            ))
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3 – Hole Filling (Results & Export)
# ============================================================
with tab3:
    result = st.session_state.get("hf_result")
    file_name = st.session_state.get("hf_file", "cloud")

    if result is not None:
        original_pts    = result['original_inlier_points']
        bezier_pts_1    = result['bezier_pts_axis1']
        bezier_pts_2    = result['bezier_pts_axis2']
        combined_bezier = result['combined_bezier_pts']
        merged          = result['merged_points']
        axis_1_name     = result['axis_1']
        axis_2_name     = result['axis_2']
        num_gaps        = len(result['gaps_axis1']) + len(result['gaps_axis2'])

        st.subheader("📊 Results")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Inliers", f"{len(original_pts)}")
        cb.metric("Gap pairs", f"{num_gaps}")
        cc.metric("Fill points", f"{result['num_filled']}")
        cd.metric("Final points", f"{len(merged)}")

        st.subheader("🌀 Filling Visualization")
        axis1_label = f"{axis_1_name}-Axis Fill"
        axis2_label = f"{axis_2_name}-Axis Fill"
        mesh_label  = "Full Cross-Hatched"

        viz_mode = st.radio("Fill Mode", options=[axis1_label, axis2_label, mesh_label], horizontal=True)

        active_bezier = combined_bezier
        if viz_mode == axis1_label: active_bezier = bezier_pts_1
        elif viz_mode == axis2_label: active_bezier = bezier_pts_2

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=original_pts[:, 0], y=original_pts[:, 1], z=original_pts[:, 2],
            mode='markers', marker=dict(size=2, color='#4A90D9', opacity=0.3), name='Inliers'
        ))
        if len(active_bezier) > 0:
            fig.add_trace(go.Scatter3d(
                x=active_bezier[:, 0], y=active_bezier[:, 1], z=active_bezier[:, 2],
                mode='markers', marker=dict(size=3, color='#39FF14', opacity=1.0), name='Bezier Fill'
            ))
        fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💾 Export")
        out_buf = io.StringIO()
        np.savetxt(out_buf, merged, fmt='%.8f %.8f %.8f')
        st.download_button(
            "Download merged .xyz",
            data=out_buf.getvalue(),
            file_name=file_name.rsplit('.', 1)[0] + '_filled.xyz',
            mime='text/plain',
        )
    else:
        st.info("Please run detection in the **Hole Detecter** tab first.")
