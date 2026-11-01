# Streamlit dashboard: California Housing map explorer and price predictor.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

SEED = 42
CLIP_COLS = ["AveRooms", "AveBedrms", "AveOccup", "Population"]

st.set_page_config(page_title="California Housing Dashboard", page_icon=":house:",
                   layout="wide")
st.markdown("<style>.block-container{padding-top:1.1rem;}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return fetch_california_housing(as_frame=True).frame


@st.cache_resource
def fit_price_model():
    # the notebook split measured generalization; a deployed model is refit on all rows
    df = load_data()
    return LinearRegression().fit(df.drop(columns=["MedHouseVal"]), df["MedHouseVal"])


@st.cache_resource
def fit_clusters(k):
    df = load_data()
    features = df.drop(columns=["MedHouseVal"]).copy()
    bounds = features[CLIP_COLS].quantile([0.01, 0.99])
    features[CLIP_COLS] = features[CLIP_COLS].clip(bounds.loc[0.01], bounds.loc[0.99], axis=1)
    pca_space = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(features))
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(pca_space).labels_


st.title("California Housing Dashboard")
page = st.sidebar.radio("Section", ["Geographic map", "Price predictor"])

df = load_data()
st.sidebar.caption(f"{len(df):,} block groups, 1990 census (scikit-learn)")

if page == "Geographic map":
    k = st.sidebar.slider("Number of clusters", 2, 8, 4)
    color_by = st.sidebar.radio("Color by", ["Cluster", "Median value"])
    labels = fit_clusters(k) if color_by == "Cluster" else None

    map_col, info_col = st.columns([2, 1], gap="medium")

    with map_col:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        if color_by == "Cluster":
            points = ax.scatter(df["Longitude"], df["Latitude"], c=labels,
                                s=3, alpha=0.5, cmap="rainbow")
            fig.colorbar(points, ax=ax, label="Cluster", ticks=range(k),
                         boundaries=np.arange(k + 1) - 0.5)
        else:
            points = ax.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
                                s=3, alpha=0.5, cmap="viridis")
            fig.colorbar(points, ax=ax, label="MedHouseVal ($100k)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"California block groups ({color_by.lower()})")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")

    with info_col:
        if color_by == "Cluster":
            st.subheader(f"Cluster profiles (k = {k})")
            profile = df.assign(cluster=labels).groupby("cluster").agg(
                Blocks=("MedHouseVal", "size"),
                Income=("MedInc", "mean"),
                Value=("MedHouseVal", "mean"),
                Latitude=("Latitude", "mean"))
            st.dataframe(profile.round(2), width="stretch")
            st.caption("Income in $10k, value in $100k. Clusters: K-Means in 2D PCA space; "
                       "AveRooms, AveBedrms, AveOccup and Population are clipped at their "
                       "1st/99th percentiles first (as in the notebook).")
        else:
            st.subheader("Median house value")
            st.dataframe(df["MedHouseVal"].describe().round(2).to_frame("Value ($100k)"),
                         width="stretch")
            st.caption("The variable the map colors by; the 5.0 cap censors the top of the range.")
else:
    input_col, result_col = st.columns([1.15, 1], gap="medium")

    with input_col:
        st.write("Adjust the block-group characteristics to predict the median house value.")
        sliders = [
            ("MedInc", "Median income ($10k)", 0.5, 15.0, 4.0, 0.1),
            ("HouseAge", "House age (years)", 1.0, 52.0, 20.0, 1.0),
            ("AveRooms", "Average rooms", 1.0, 10.0, 5.0, 0.1),
            ("AveBedrms", "Average bedrooms", 0.5, 2.5, 1.0, 0.05),
            ("Population", "Population", 100.0, 6000.0, 1500.0, 100.0),
            ("AveOccup", "Average occupancy", 1.0, 6.0, 3.0, 0.1),
            ("Latitude", "Latitude", 32.5, 42.0, 36.0, 0.1),
            ("Longitude", "Longitude", -124.4, -114.3, -119.5, 0.1),
        ]
        col_a, col_b = st.columns(2)
        inputs = {}
        for i, (feature, label, low, high, default, step) in enumerate(sliders):
            target = col_a if i < 4 else col_b
            inputs[feature] = target.slider(label, low, high, default, step=step)

    with result_col:
        prediction = fit_price_model().predict(pd.DataFrame([inputs]))[0]
        median = df["MedHouseVal"].median()
        st.metric("Predicted median house value", f"${prediction * 100_000:,.0f}",
                  delta=f"{(prediction - median) * 100_000:+,.0f} vs dataset median",
                  delta_color="off")
        st.caption("Linear regression on all 20,640 rows; held-out R² was about 0.58 in the notebook.")

        fig, ax = plt.subplots(figsize=(5.4, 2.4))
        ax.hist(df["MedHouseVal"], bins=50, color="steelblue", edgecolor="white")
        ax.axvline(prediction, color="crimson", lw=2, label="Prediction")
        ax.legend()
        ax.set_xlabel("MedHouseVal ($100k)")
        ax.set_title("Prediction against the dataset distribution")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
