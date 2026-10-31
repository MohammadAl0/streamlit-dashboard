# Streamlit dashboard: California Housing map explorer and price predictor.
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


@st.cache_data
def load_data():
    return fetch_california_housing(as_frame=True).frame


@st.cache_resource
def fit_price_model():
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

if page == "Geographic map":
    k = st.sidebar.slider("Number of clusters", 2, 8, 4)
    color_by = st.sidebar.radio("Color by", ["Cluster", "Median value"])
    colors = fit_clusters(k) if color_by == "Cluster" else df["MedHouseVal"]
    fig, ax = plt.subplots(figsize=(6, 7))
    points = ax.scatter(df["Longitude"], df["Latitude"], c=colors, s=3, alpha=0.5,
                        cmap="rainbow" if color_by == "Cluster" else "viridis")
    if color_by == "Median value":
        fig.colorbar(points, ax=ax, label="MedHouseVal ($100k)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"California block groups ({color_by.lower()})")
    st.pyplot(fig)
else:
    st.write("Adjust the block-group characteristics to predict the median house value.")
    ranges = {"MedInc": (0.5, 15.0, 4.0), "HouseAge": (1.0, 52.0, 20.0),
              "AveRooms": (1.0, 10.0, 5.0), "AveBedrms": (0.5, 2.5, 1.0),
              "Population": (100.0, 6000.0, 1500.0), "AveOccup": (1.0, 6.0, 3.0),
              "Latitude": (32.5, 42.0, 36.0), "Longitude": (-124.4, -114.3, -119.5)}
    inputs = {feature: st.slider(feature, low, high, default)
              for feature, (low, high, default) in ranges.items()}
    prediction = fit_price_model().predict(pd.DataFrame([inputs]))[0]
    st.metric("Predicted median house value", f"${prediction * 100_000:,.0f}")
