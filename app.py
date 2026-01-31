import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from collections import Counter
from PIL import Image

st.set_page_config(
    page_title="MNIST Digit Clustering",
    layout="wide"
)
st.markdown(
    """
    <style>
    .main {
        overflow: visible;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🧠 MNIST Digit Clustering with t-SNE & K-Means")

st.markdown("""
This interactive app demonstrates **unsupervised learning** on handwritten digits  
using **t-SNE for dimensionality reduction** and **K-Means clustering**.
""")

st.sidebar.header("⚙️ Controls")

k = st.sidebar.slider(
    "Number of clusters (k)",
    min_value=3,
    max_value=15,
    value=10
)

perplexity = st.sidebar.slider(
    "t-SNE Perplexity",
    min_value=5,
    max_value=50,
    value=30
)

random_state = st.sidebar.number_input(
    "Random State",
    value=42
)

@st.cache_data
def load_data():
    digits = load_digits()
    return digits.data, digits.target, digits.images

X, y_true, images = load_data()

st.subheader("📊 Dataset Info")
st.write(f"Samples: **{X.shape[0]}**, Features: **{X.shape[1]}**")

st.subheader("🔽 Dimensionality Reduction (t-SNE)")

with st.spinner("Running t-SNE..."):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random"
    )
    X_2d = tsne.fit_transform(X)

with st.spinner("Running K-Means clustering..."):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_2d)

cluster_to_digit = {}
y_pred = np.zeros_like(cluster_labels)

for cluster_id in range(k):
    mask = cluster_labels == cluster_id
    if np.sum(mask) == 0:
        continue
    most_common = Counter(y_true[mask]).most_common(1)[0][0]
    cluster_to_digit[cluster_id] = most_common
    y_pred[mask] = most_common

accuracy = accuracy_score(y_true, y_pred)
sil_score = silhouette_score(X_2d, cluster_labels)

st.subheader("📈 Evaluation Metrics")
col1, col2 = st.columns(2)
col1.metric("Post-Mapped Accuracy", f"{accuracy:.4f}")
col2.metric("Silhouette Score", f"{sil_score:.4f}")

st.subheader("🧩 2D Cluster Visualization")

plot_df = {
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "cluster": cluster_labels.astype(str),
    "true_digit": y_true.astype(str),
    "predicted_digit": y_pred.astype(str)
}

fig = px.scatter(
    plot_df,
    x="x",
    y="y",
    color="cluster",
    hover_data=["true_digit", "predicted_digit"],
    title="t-SNE Projection of MNIST Digits (Colored by Cluster)",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("📌 Cluster Analysis")

stats = []
for cluster_id in range(k):
    mask = cluster_labels == cluster_id
    count = np.sum(mask)
    common_digit = cluster_to_digit.get(cluster_id, "N/A")
    stats.append({
        "Cluster": cluster_id,
        "Samples": count,
        "Most Common Digit": common_digit
    })

st.dataframe(stats)

st.subheader("❌ Misclassified Digits")

mis_idx = np.where(y_true != y_pred)[0]

st.write(f"Total misclassified samples: **{len(mis_idx)}**")

show_n = st.slider("Number of misclassified digits to show", 1, 20, 5)

cols = st.columns(show_n)

for i, idx in enumerate(mis_idx[:show_n]):
    with cols[i]:
        st.image(
            images[idx] / 16.0,  
            caption=f"True: {y_true[idx]} | Pred: {y_pred[idx]}",
            width=80
        )


st.markdown("""
### ✅ Key Takeaways
- t-SNE effectively reveals structure in high-dimensional image data  
- K-Means can group visually similar digits even without labels  
- Some digits (like **4, 9, 3, 5**) overlap due to similar shapes  

This project demonstrates **classical ML + visualization + analysis**,  
which are core skills for backend + AI engineering.
""")
