import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.write("# Wine Dataset Explorer 📊")
st.markdown(
    "Explore the [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) "
    "used to train the classification model."
)


@st.cache_data
def get_wine_dataframe():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    df["class_name"] = df["target"].map(
        {i: name for i, name in enumerate(wine.target_names)}
    )
    return df, wine.feature_names


df, feature_names = get_wine_dataframe()

# --- Dataset overview ---
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Samples", df.shape[0])
col2.metric("Features", df.shape[1] - 2)
col3.metric("Classes", df["target"].nunique())

with st.expander("View raw data"):
    st.dataframe(df, use_container_width=True)

with st.expander("Descriptive statistics"):
    st.dataframe(df.describe(), use_container_width=True)

st.divider()

# --- Class distribution ---
st.subheader("Class Distribution")
class_counts = df["class_name"].value_counts().reset_index()
class_counts.columns = ["Class", "Count"]
st.bar_chart(class_counts.set_index("Class"))

st.divider()

# --- Feature distribution ---
st.subheader("Feature Distribution by Class")
selected_feature = st.selectbox("Select a feature to visualize:", feature_names)

chart_data = df[["class_name", selected_feature]].copy()
pivot = chart_data.pivot_table(
    index=chart_data.index,
    columns="class_name",
    values=selected_feature,
)
st.line_chart(pivot)

st.divider()

# --- Feature correlation ---
st.subheader("Feature Correlation Heatmap")
st.markdown("Pairwise Pearson correlation between all 13 features.")
corr = df[list(feature_names)].corr()
st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)

st.divider()

# --- Scatter plot ---
st.subheader("2-D Scatter Plot")
col_a, col_b = st.columns(2)
feat_x = col_a.selectbox("X-axis feature:", feature_names, index=0)
feat_y = col_b.selectbox("Y-axis feature:", feature_names, index=6)

scatter_df = df[[feat_x, feat_y, "class_name"]].copy()
scatter_df.columns = [feat_x, feat_y, "class"]

st.scatter_chart(
    scatter_df,
    x=feat_x,
    y=feat_y,
    color="class",
    use_container_width=True,
)
