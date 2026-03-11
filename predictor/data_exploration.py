from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
GEOJSON_PATH = BASE_DIR / "dummy-data" / "rwanda_districts.geojson"

def _normalize(v):
    return str(v).strip().lower().replace("-", " ")

def _get_centroid(coords):
    # Simplified flattening to extract all lat/lons from potentially nested list
    def flatten(l):
        for i in l:
            if isinstance(i, list) and not isinstance(i[0], (int, float)):
                yield from flatten(i)
            else:
                yield i
    points = list(flatten(coords))
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return sum(lats)/len(lats), sum(lons)/len(lons)

def district_map_html(df: pd.DataFrame) -> str:
    # 1. Prepare data
    counts = df.groupby("district").size().reset_index(name="count")
    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    
    # 2. Extract centers for labels
    centers = {}
    for f in geojson["features"]:
        name = f["properties"].get("shapeName") or f.get("id")
        if name:
            centers[_normalize(name)] = _get_centroid(f["geometry"]["coordinates"])

    # 3. Create map (Rwanda Focused)
    fig = px.choropleth_mapbox(
        counts, geojson=geojson, locations="district",
        featureidkey="properties.shapeName", color="count",
        color_continuous_scale="Blues", opacity=0.7,
        mapbox_style="carto-positron", zoom=7.5,
        center={"lat": -1.94, "lon": 29.87}
    )

    # 4. Add persistent labels with client counts
    label_lats, label_lons, label_texts = [], [], []
    for _, row in counts.iterrows():
        key = _normalize(row["district"])
        if key in centers:
            lat, lon = centers[key]
            label_lats.append(lat)
            label_lons.append(lon)
            label_texts.append(f"{row['district']}<br>{int(row['count'])} clients")

    fig.add_trace(go.Scattermapbox(
        lat=label_lats, lon=label_lons, text=label_texts,
        mode="text", textfont={"size": 11, "color": "#1e293b"},
        hoverinfo="skip", showlegend=False
    ))

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title_text="<b>Rwanda Districts: Client Distribution</b>",
        height=600, coloraxis_colorbar={"title": "Clients", "thickness": 15}
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})

def world_map_html(df: pd.DataFrame) -> str:
    # Same data preparation as district map
    counts = df.groupby("district").size().reset_index(name="count")
    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    
    centers = {}
    for f in geojson["features"]:
        name = f["properties"].get("shapeName") or f.get("id")
        if name:
            centers[_normalize(name)] = _get_centroid(f["geometry"]["coordinates"])

    # Create map with World view
    fig = px.choropleth_mapbox(
        counts, geojson=geojson, locations="district",
        featureidkey="properties.shapeName", color="count",
        color_continuous_scale="Blues", opacity=0.8,
        mapbox_style="carto-positron", zoom=7.5,
        center={"lat": -1.94, "lon": 29.88}
    )

    # Add labels (same as before)
    label_lats, label_lons, label_texts = [], [], []
    for _, row in counts.iterrows():
        key = _normalize(row["district"])
        if key in centers:
            lat, lon = centers[key]
            label_lats.append(lat)
            label_lons.append(lon)
            label_texts.append(f"{row['district']}<br>{int(row['count'])} clients")

    fig.add_trace(go.Scattermapbox(
        lat=label_lats, lon=label_lons, text=label_texts,
        mode="text", textfont={"size": 10, "color": "#1e293b"},
        hoverinfo="skip", showlegend=False
    ))

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        title_text="<b>Global Overview: Regional Hub Analytics</b>",
        height=700, coloraxis_colorbar={"title": "Clients", "thickness": 15}
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def dataset_exploration(df: pd.DataFrame) -> str:
    return df.head(12).to_html(classes="table table-bordered table-striped table-sm", index=False)

def data_exploration(df: pd.DataFrame) -> str:
    return df.describe(include="all").to_html(classes="table table-bordered table-striped table-sm")
