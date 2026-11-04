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


# Cache the data and fitted models so the sliders stay responsive.
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
    labels = fit_clusters(k) if color_by == "Cluster" else None

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    if color_by == "Cluster":
        points = ax.scatter(df["Longitude"], df["Latitude"], c=labels, s=3, alpha=0.5,
                            cmap="rainbow")
        fig.colorbar(points, ax=ax, label="Cluster")
    else:
        points = ax.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"], s=3,
                            alpha=0.5, cmap="viridis")
        fig.colorbar(points, ax=ax, label="MedHouseVal ($100k)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"California block groups ({color_by.lower()})")
    st.pyplot(fig)

    if color_by == "Cluster":
        profile = df.assign(cluster=labels).groupby("cluster").agg(
            Blocks=("MedHouseVal", "size"),
            Income=("MedInc", "mean"),
            Value=("MedHouseVal", "mean"))
        st.dataframe(profile.round(2))
else:
    st.write("Adjust the block-group characteristics to predict the median house value.")
    ranges = {
        "MedInc": ("Median income ($10k)", 0.5, 15.0, 4.0, 0.1),
        "HouseAge": ("House age (years)", 1.0, 52.0, 20.0, 1.0),
        "AveRooms": ("Average rooms", 1.0, 10.0, 5.0, 0.1),
        "AveBedrms": ("Average bedrooms", 0.5, 2.5, 1.0, 0.05),
        "Population": ("Population", 100.0, 6000.0, 1500.0, 100.0),
        "AveOccup": ("Average occupancy", 1.0, 6.0, 3.0, 0.1),
        "Latitude": ("Latitude", 32.5, 42.0, 36.0, 0.1),
        "Longitude": ("Longitude", -124.4, -114.3, -119.5, 0.1),
    }
    inputs = {}
    for feature, (label, low, high, default, step) in ranges.items():
        inputs[feature] = st.slider(label, low, high, default, step)

    prediction = fit_price_model().predict(pd.DataFrame([inputs]))[0]
    st.metric("Predicted median house value", f"${prediction * 100_000:,.0f}")
