# California Housing Dashboard

Streamlit app for the AML601 California Housing project. Two sections: a map
explorer that colors block groups by K-Means cluster (adjustable k) or by
median value, and a price predictor over the eight features. The clustering
matches the notebook: percentile clipping, standardizing, 2D PCA, K-Means with
seed 42.

## Run locally

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

Push this folder to a GitHub repo, then on [share.streamlit.io](https://share.streamlit.io)
create a new app pointing at `streamlit_app.py`. The dataset is downloaded from
scikit-learn at first load, so no data files are needed.

## Docker

```
docker build -t california-housing .
docker run -p 8501:8501 california-housing
```
