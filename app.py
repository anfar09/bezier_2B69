import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd

from slicer import load_points_from_file, get_bounds, cuboid_slices, detect_holes_in_slices

st.set_page_config(layout="wide")
st.title("🔬 3D Point Cloud + Hole Detection")

# ---------- Load ----------
DATASET_FOLDER = "Dataset"
os.makedirs(DATASET_FOLDER, exist_ok=True)

files = [f for f in os.listdir(DATASET_FOLDER)
         if f.endswith(('.txt','.xyz','.csv','.pts'))]

if not files:
    st.error("ใส่ไฟล์ใน Dataset ก่อน")
    st.stop()

file = st.selectbox("เลือกไฟล์", files)
path = os.path.join(DATASET_FOLDER, file)

@st.cache_data
def load(fp):
    return load_points_from_file(fp)

points = np.array(load(path))
bounds = get_bounds(points)

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["3D View", "Analysis"])

# ---------- TAB 1 ----------
with tab1:
    col1, col2 = st.columns([3,1])

    with col2:
        size = st.slider("Point size", 1, 10, 2)
        axis = st.selectbox("Axis", ["X","Y","Z"])
        fixed = st.checkbox("Fixed Mode")

        r = bounds[axis.lower()][1] - bounds[axis.lower()][0]
        r = max(r, 1e-6)

        w = st.number_input("Slice width", min_value=r/100,
                            value=max(r/5, r/100), step=r/100)

        n = int(np.ceil(r/w))

        if fixed:
            idx_slice = st.slider("Slice", 0, max(n-1,0), 0)

    with col1:
        fig = go.Figure()

        axis_map = {"X":0,"Y":1,"Z":2}
        idx = axis_map[axis]

        xmin,xmax = bounds['x']
        ymin,ymax = bounds['y']
        zmin,zmax = bounds['z']

        def draw_box(cur,nxt,color='red'):
            if axis=="X":
                xs,ys,zs=[cur,nxt],[ymin,ymax],[zmin,zmax]
            elif axis=="Y":
                xs,ys,zs=[xmin,xmax],[cur,nxt],[zmin,zmax]
            else:
                xs,ys,zs=[xmin,xmax],[ymin,ymax],[cur,nxt]

            edges = [
                ([xs[0],xs[1]],[ys[0],ys[0]],[zs[0],zs[0]]),
                ([xs[0],xs[1]],[ys[1],ys[1]],[zs[0],zs[0]])
            ]

            for ex,ey,ez in edges:
                fig.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',
                                          line=dict(color=color,width=4),
                                          showlegend=False))

        if fixed:
            cur = bounds[axis.lower()][0] + idx_slice*w
            nxt = min(cur+w, bounds[axis.lower()][1])

            mask = (points[:,idx]>=cur)&(points[:,idx]<nxt)

            pts_in = points[mask]
            pts_out = points[~mask]

            fig.add_trace(go.Scatter3d(
                x=pts_out[:,0],y=pts_out[:,1],z=pts_out[:,2],
                mode='markers', marker=dict(size=size,color='gray',opacity=0.05)
            ))

            fig.add_trace(go.Scatter3d(
                x=pts_in[:,0],y=pts_in[:,1],z=pts_in[:,2],
                mode='markers', marker=dict(size=size+1,color='red')
            ))

            draw_box(cur,nxt,'yellow')

        else:
            fig.add_trace(go.Scatter3d(
                x=points[:,0],y=points[:,1],z=points[:,2],
                mode='markers', marker=dict(size=size,color=points[:,2],colorscale='Viridis')
            ))

        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    axis = st.selectbox("Axis", ["X","Y","Z"], key="axis2")

    r = bounds[axis.lower()][1] - bounds[axis.lower()][0]
    r = max(r,1e-6)

    w = st.number_input("Slice width", min_value=r/100,
                        value=max(r/5,r/100), step=r/100, key="w2")

    if st.button("Detect Hole"):
        axis_map = {"X":1,"Y":2,"Z":3}
        slices = cuboid_slices(points.tolist(), axis_map[axis], w)

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