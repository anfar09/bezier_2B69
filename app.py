import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
import io

from slicer import load_points_from_file, get_bounds, cuboid_slices, detect_holes_in_slices
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
tab1, tab2, tab3 = st.tabs(["3D View", "Analysis", "Hole Filling"])

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
# TAB 2 – Original Analysis
# ============================================================
with tab2:
    DATASET_FOLDER2 = "Dataset"
    os.makedirs(DATASET_FOLDER2, exist_ok=True)

    files2 = [f for f in os.listdir(DATASET_FOLDER2)
              if f.endswith(('.txt', '.xyz', '.csv', '.pts'))]

    if not files2:
        st.error("ใส่ไฟล์ใน Dataset ก่อน")
    else:
        file2  = st.selectbox("เลือกไฟล์", files2, key="file2")
        path2  = os.path.join(DATASET_FOLDER2, file2)

        @st.cache_data
        def load2(fp):
            return load_points_from_file(fp)

        points2  = np.array(load2(path2))
        bounds2  = get_bounds(points2)

        axis = st.selectbox("Axis", ["X", "Y", "Z"], key="axis2")

        r = bounds2[axis.lower()][1] - bounds2[axis.lower()][0]
        r = max(r, 1e-6)

        w = st.number_input("Slice width", min_value=r/100,
                            value=max(r/5, r/100), step=r/100, key="w2")

        if st.button("Detect Hole", key="detect_hole_btn"):
            axis_map = {"X": 1, "Y": 2, "Z": 3}
            slices   = cuboid_slices(points2.tolist(), axis_map[axis], w)
            results, mu = detect_holes_in_slices(slices, axis)

            st.write("Mean gap μ =", mu)

            df = pd.DataFrame(results)
            st.dataframe(df)

            for r in results:
                if r["has_hole"]:
                    st.error(f"Slice {r['slice']} → HOLE 🔴")
                else:
                    st.success(f"Slice {r['slice']} → OK 🟢")

            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "result.csv")


# ============================================================
# TAB 3 – Hole Filling (backend: hole_filler.py)
# ============================================================
with tab3:
    # ------------------------------------------------------------------
    # Step 1: Local Dataset/ folder – no file uploader
    # ------------------------------------------------------------------
    st.subheader("📁 Select Point Cloud")
    DB_FOLDER = "Dataset"
    os.makedirs(DB_FOLDER, exist_ok=True)

    db_files = sorted([
        f for f in os.listdir(DB_FOLDER)
        if f.endswith(('.xyz', '.txt', '.pts'))
    ])

    if not db_files:
        st.error(f"No `.xyz` files found in the `{DB_FOLDER}/` folder. "
                 "Please add point cloud files there.")
        st.stop()

    selected_file = st.selectbox("Choose a file from Dataset/", db_files)
    file_path = os.path.join(DB_FOLDER, selected_file)

    # Parse once using the same helper the rest of the app uses
    @st.cache_data
    def load_points_for_hf(fp):
        return np.array(load_points_from_file(fp))

    points_raw = load_points_for_hf(file_path)

    st.info(f"**{len(points_raw)}** raw points loaded from `{file_path}`")

    # ------------------------------------------------------------------
    # Step 2: Hyperparameter UI
    # ------------------------------------------------------------------
    st.subheader("⚙️ Hyperparameters")

    col_slice, col_gap = st.columns(2)
    with col_slice:
        st.markdown("**slice_thickness** (leave empty = auto)")
        slice_text = st.text_input("slice_thickness", value="",
                                    placeholder="auto", label_visibility="collapsed")
        slice_thickness = float(slice_text) if slice_text.strip() else None

    with col_gap:
        st.markdown("**gap_threshold** (leave empty = auto)")
        gap_text = st.text_input("gap_threshold", value="",
                                 placeholder="auto", label_visibility="collapsed")
        gap_threshold = float(gap_text) if gap_text.strip() else None

    col_np, col_nk = st.columns(2)
    with col_np:
        num_points = st.number_input(
            "num_points_per_gap", min_value=3, max_value=100, value=20)

    with col_nk:
        neighbor_k = st.slider(
            "neighbor_k", min_value=2, max_value=30, value=10,
            help="Number of nearest neighbours used to estimate surface tangent at each "
                 "gap boundary. Higher values give smoother curves but are slower.")

    run = st.button("🚀 Run Hole Filling", type="primary")

    # ------------------------------------------------------------------
    # Step 3: Run backend on button click → cache in session_state
    # ------------------------------------------------------------------
    if run:
        with st.spinner("Running hole-filling pipeline..."):
            result = process_point_cloud(
                points_raw,
                slice_thickness=slice_thickness,
                gap_threshold=gap_threshold,
                num_points_per_gap=num_points,
                neighbor_k=neighbor_k,
                verbose=False,
            )
        # Store once; viz toggle will read from here without re-running
        st.session_state["hf_result"]    = result
        st.session_state["hf_file"]     = selected_file

    # ------------------------------------------------------------------
    # Step 4: Unpack all 6 values from the backend result dict
    # ------------------------------------------------------------------
    result    = st.session_state.get("hf_result")
    file_name = st.session_state.get("hf_file", "cloud")

    if result is not None:
        # Unpack all 6 values from the backend result
        original_pts    = result['original_inlier_points']
        bezier_pts_1    = result['bezier_pts_axis1']
        bezier_pts_2    = result['bezier_pts_axis2']
        combined_bezier = result['combined_bezier_pts']
        merged          = result['merged_points']
        axis_1_name     = result['axis_1']   # e.g. 'X', 'Y', or 'Z'
        axis_2_name     = result['axis_2']   # e.g. 'X', 'Y', or 'Z'
        num_gaps        = len(result['gaps_axis1']) + len(result['gaps_axis2'])

        # ---- Stats ----
        st.subheader("📊 Results")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Raw points",        f"{len(points_raw)}")
        cb.metric("SOR inliers",       f"{len(original_pts)}")
        cc.metric("Gap pairs found",   f"{num_gaps}")
        cd.metric("Fill points",       f"{result['num_filled']}")

        ce, cf = st.columns(2)
        with ce:
            st.write(f"**Primary axis:**   `{axis_1_name}`")
            st.write(f"**Secondary axis:** `{axis_2_name}`")
            st.write(f"**Thickness axis:** `{result['axis_3']}`")
        with cf:
            st.write(f"**slice_thickness:**   {result['slice_thickness']:.6f}")
            st.write(f"**gap_threshold:**      {result['gap_threshold']:.6f}")
            st.write(f"**avg_point_spacing:** {result['avg_point_spacing']:.6f}")

        cg, ch = st.columns(2)
        with cg:
            st.write(f"**{axis_1_name}-axis fill points:** {len(bezier_pts_1)}")
        with ch:
            st.write(f"**{axis_2_name}-axis fill points:** {len(bezier_pts_2)}")

        # ------------------------------------------------------------------
        # Step 5: Dynamic Visualization Mode toggle
        # ------------------------------------------------------------------
        st.subheader("🌀 3D Visualization")

        axis1_label = f"{axis_1_name}-Axis Fill Only"
        axis2_label = f"{axis_2_name}-Axis Fill Only"
        mesh_label  = "Full Cross-Hatched (Mesh)"

        viz_mode = st.radio(
            "Visualization Mode",
            options=[axis1_label, axis2_label, mesh_label],
            format_func=lambda x: x,
            horizontal=True,
            help=(
                f"**{axis1_label}** — Bezier fill lines running along the "
                f"`{axis_1_name}` (primary/highest-variance) axis only.\n"
                f"**{axis2_label}** — Bezier fill lines running along the "
                f"`{axis_2_name}` (secondary) axis only.\n"
                f"**{mesh_label}** — Both sets of fill lines interleaved, "
                "forming a cross-hatched NURBS-like mesh surface."
            ),
        )

        # Select which Bezier points to show
        if viz_mode == axis1_label:
            active_bezier = bezier_pts_1
            bezier_label  = f"{axis_1_name}-Axis Fill ({len(bezier_pts_1)})"
        elif viz_mode == axis2_label:
            active_bezier = bezier_pts_2
            bezier_label  = f"{axis_2_name}-Axis Fill ({len(bezier_pts_2)})"
        else:
            active_bezier = combined_bezier
            bezier_label  = f"Cross-Hatched ({len(combined_bezier)})"

        # Build Plotly figure
        fig = go.Figure()

        # Original inliers – light blue, semi-transparent
        if len(original_pts) > 0:
            fig.add_trace(go.Scatter3d(
                x=original_pts[:, 0],
                y=original_pts[:, 1],
                z=original_pts[:, 2],
                mode='markers',
                marker=dict(size=2, color='#4A90D9', opacity=0.3, line=dict(width=0)),
                name=f'Original Inliers ({len(original_pts)})',
            ))

        # Selected Bezier fill – neon green, fully opaque
        if len(active_bezier) > 0:
            fig.add_trace(go.Scatter3d(
                x=active_bezier[:, 0],
                y=active_bezier[:, 1],
                z=active_bezier[:, 2],
                mode='markers',
                marker=dict(size=3, color='#39FF14', opacity=1.0, line=dict(width=0)),
                name=bezier_label,
            ))

        fig.update_layout(
            title=dict(
                text=f"Hole Fill — {viz_mode}  [{file_name}]",
                x=0.5, font=dict(size=15),
            ),
            legend=dict(x=0.01, y=0.99,
                        bgcolor='rgba(0,0,0,0.5)',
                        font=dict(color='white')),
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # Side-by-side per-axis comparison charts
        # ------------------------------------------------------------------
        if len(bezier_pts_1) > 0 or len(bezier_pts_2) > 0:
            st.subheader("🔀 Axis Comparison")

            left_col, right_col = st.columns(2)
            for col, pts, ax_name in [
                (left_col,  bezier_pts_1, axis_1_name),
                (right_col, bezier_pts_2, axis_2_name),
            ]:
                with col:
                    sub = go.Figure()
                    if len(original_pts) > 0:
                        sub.add_trace(go.Scatter3d(
                            x=original_pts[:, 0], y=original_pts[:, 1], z=original_pts[:, 2],
                            mode='markers',
                            marker=dict(size=2, color='#4A90D9', opacity=0.3, line=dict(width=0)),
                            name='Inliers',
                        ))
                    if len(pts) > 0:
                        sub.add_trace(go.Scatter3d(
                            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                            mode='markers',
                            marker=dict(size=3, color='#FFD700', opacity=1.0, line=dict(width=0)),
                            name=f'{ax_name}-Axis Fill',
                        ))
                    sub.update_layout(
                        title=dict(text=f"{ax_name}-Axis Fill", x=0.5, font=dict(size=13)),
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                   aspectmode='data'),
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=400,
                    )
                    st.plotly_chart(sub, use_container_width=True)

        # ------------------------------------------------------------------
        # Export
        # ------------------------------------------------------------------
        st.subheader("💾 Export")
        out_buf = io.StringIO()
        np.savetxt(out_buf, merged, fmt='%.8f %.8f %.8f')
        st.download_button(
            "Download merged .xyz",
            data=out_buf.getvalue(),
            file_name=file_name.rsplit('.', 1)[0] + '_filled.xyz',
            mime='text/plain',
        )
    elif run:
        st.error("No result returned. Check the console for errors.")
    else:
        st.info("Adjust hyperparameters above and click **🚀 Run Hole Filling** to begin.")
