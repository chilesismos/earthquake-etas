# ============================================================
# PRONÓSTICO SÍSMICO PROBABILÍSTICO — ETAS V2
# CHILE
#
# Versión .py para GitHub / GitHub Actions
#
# + USGS / EMSC
# + CONTROL DE CALIDAD
# + Mc / b-value
# + ETAS TEMPORAL
# + VALIDACIÓN TEMPORAL
# + FORECAST ESPACIAL
# + RANKING TOP 5
# + LÁMINA RRSS
# + TÍTULO DEL MAPA EN FILA INDEPENDIENTE
# + PUBLICACIÓN OPCIONAL EN X
# + CREDENCIALES HARDCODED DENTRO DEL ARCHIVO
#
# IMPORTANTE:
# Modelo estadístico experimental.
# NO constituye alerta sísmica ni predicción determinista.
# ============================================================

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Necesario en GitHub Actions / servidores sin pantalla.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from numba import njit

from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

import tweepy

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

CONFIG = {
    "start_date": "2015-01-01",

    # Timestamp completo UTC para no perder los eventos del día actual.
    "end_date": pd.Timestamp.now(tz="UTC"),

    "min_latitude": -56.0,
    "max_latitude": -17.0,

    "min_longitude": -78.0,
    "max_longitude": -65.0,

    "download_min_magnitude": 3.0,

    "chunk_months": 1,
    "max_retries": 3,
    "sleep_seconds": 0.25,

    "download_usgs": True,
    "download_emsc": True,

    # El ajuste ETAS principal usa USGS si está disponible.
    "etas_source": "USGS",

    "training_years": 8,
    "validation_days": 180,

    "magnitude_bin": 0.1,
    "mc_gof_threshold": 90.0,

    "max_fit_events": 4000,
    "optimizer_restarts": 5,

    "grid_resolution_deg": 0.10,

    "background_sigma_cells": 2.0,

    "spatial_q": 1.5,
    "spatial_d_min_km": 5.0,
    "spatial_d_max_km": 100.0,

    "trigger_memory_days": 365.25 * 5,

    "forecast_start": None,

    "forecast_horizons_days": {
        "24h": 1.0,
        "7d": 7.0,
        "30d": 30.0
    },

    "forecast_magnitudes": [
        4.0,
        5.0,
        6.0,
        7.0
    ],

    # Ruta relativa al repositorio / runner de GitHub Actions.
    "output_dir": "output"
}


# ============================================================
# 2. CONFIGURACIÓN DE LA LÁMINA
# ============================================================

POSTER_CONFIG = {
    "main_horizon": "7d",
    "main_magnitude": 5.0,

    "top_zones": 5,
    "minimum_zone_separation_km": 150.0,

    "recent_days": 30,

    "figsize": (12, 12),
    "dpi": 300,

    "title": "PRONÓSTICO SÍSMICO PROBABILÍSTICO",
    "subtitle": "Modelo ETAS V2 · Chile",

    "background_color": "#F3F6F8",
    "panel_color": "#FFFFFF",
    "header_color": "#123B5D",
    "header_text_color": "#FFFFFF",
    "border_color": "#C1CDD5",
    "text_color": "#172633",
    "muted_text_color": "#657681"
}


# ============================================================
# 3. CONFIGURACIÓN X
# ============================================================

TWITTER_CONFIG = {
    "enabled": True,
    "attach_image": True,
    "include_disclaimer": True,
    "fail_silently": True
}


# ============================================================
# 4. CREDENCIALES X HARDCODED
#
# PEGA AQUÍ TUS CREDENCIALES REALES ENTRE COMILLAS.
# Se mantienen dentro del .py, tal como solicitaste.
#
# IMPORTANTE:
# No publiques este archivo en un repositorio público.
# Si las credenciales que usabas ya fueron compartidas,
# regénéralas antes de poner este archivo en producción.
# ============================================================

X_CREDENTIALS = {
    "use_hardcoded_credentials": True,

    "X_API_KEY":
        "vJ6yw8JZowpL2XrKoPDHjV6hD",

    "X_API_SECRET":
        "TtmZgWqwgNVa1FDFC60bvEaK9OipW2wfiIXXSGwNmKdwM55UU0",

    "X_ACCESS_TOKEN":
        "2561368769-CUeq4EyGJ19ehkH5utqn0aXfhNrUQApIZPybCD6",

    "X_ACCESS_TOKEN_SECRET":
        "jYw1qL8ML0J9JQbaNR2IPpMAGOInJ9qHcvyC8XZJS0oDZ"
}


# ============================================================
# 5. DIRECTORIOS
# ============================================================

PROJECT_DIR = Path(CONFIG["output_dir"])
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
FORECAST_DIR = PROJECT_DIR / "forecast"
FIGURE_DIR = PROJECT_DIR / "figures"

for directory in [
    PROJECT_DIR,
    DATA_DIR,
    MODEL_DIR,
    FORECAST_DIR,
    FIGURE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. CLIENTES FDSN
# ============================================================

USGS_CLIENT = Client("USGS")

EMSC_CLIENT = Client(
    "https://www.seismicportal.eu",
    service_mappings={
        "event": "https://www.seismicportal.eu/fdsnws/event/1"
    }
)


# ============================================================
# 7. UTILIDADES
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def ensure_utc_timestamp(value):
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def create_time_chunks(start_date, end_date, months=1):
    start = ensure_utc_timestamp(start_date)
    end = ensure_utc_timestamp(end_date)

    chunks = []
    current = start

    while current < end:
        next_date = min(
            current + pd.DateOffset(months=months),
            end
        )

        chunks.append((current, next_date))
        current = next_date

    return chunks


def catalog_to_dataframe(catalog, source):
    records = []

    for event in catalog:
        try:
            origin = event.preferred_origin()

            if origin is None:
                origin = event.origins[0]

            magnitude = event.preferred_magnitude()

            if magnitude is None:
                magnitude = event.magnitudes[0]

            event_id = (
                str(event.resource_id)
                if event.resource_id is not None
                else None
            )

            records.append({
                "time": pd.Timestamp(
                    origin.time.datetime,
                    tz="UTC"
                ),
                "latitude": float(origin.latitude),
                "longitude": float(origin.longitude),
                "depth_km": (
                    float(origin.depth / 1000.0)
                    if origin.depth is not None
                    else np.nan
                ),
                "magnitude": float(magnitude.mag),
                "magnitude_type": magnitude.magnitude_type,
                "source": source,
                "event_id": event_id
            })

        except Exception:
            continue

    return pd.DataFrame(records)


# ============================================================
# 8. DESCARGA
# ============================================================

def download_catalog(client, source):
    chunks = create_time_chunks(
        CONFIG["start_date"],
        CONFIG["end_date"],
        CONFIG["chunk_months"]
    )

    frames = []

    print(f"\nDescargando catálogo {source}...")

    for i, (chunk_start, chunk_end) in enumerate(
        chunks,
        start=1
    ):
        success = False

        for attempt in range(CONFIG["max_retries"]):
            try:
                catalog = client.get_events(
                    starttime=UTCDateTime(
                        chunk_start.to_pydatetime()
                    ),
                    endtime=UTCDateTime(
                        chunk_end.to_pydatetime()
                    ),
                    minlatitude=CONFIG["min_latitude"],
                    maxlatitude=CONFIG["max_latitude"],
                    minlongitude=CONFIG["min_longitude"],
                    maxlongitude=CONFIG["max_longitude"],
                    minmagnitude=CONFIG[
                        "download_min_magnitude"
                    ],
                    orderby="time-asc"
                )

                df_chunk = catalog_to_dataframe(
                    catalog,
                    source
                )

                if len(df_chunk) > 0:
                    frames.append(df_chunk)

                success = True
                break

            except Exception as exc:
                if (
                    attempt
                    ==
                    CONFIG["max_retries"] - 1
                ):
                    print(
                        "Error:",
                        source,
                        chunk_start.date(),
                        chunk_end.date(),
                        exc
                    )
                else:
                    time.sleep(2.0 * (attempt + 1))

        if success and (
            i % 12 == 0
            or
            i == len(chunks)
        ):
            print(
                f"{source}: "
                f"{i}/{len(chunks)} bloques"
            )

        time.sleep(CONFIG["sleep_seconds"])

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )


# ============================================================
# 9. CONTROL DE CALIDAD
# ============================================================

def quality_control(df):
    if df.empty:
        return df

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    for col in [
        "latitude",
        "longitude",
        "depth_km",
        "magnitude"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "time",
            "latitude",
            "longitude",
            "magnitude"
        ]
    )

    df = df[
        df["latitude"].between(
            CONFIG["min_latitude"],
            CONFIG["max_latitude"]
        )
    ]

    df = df[
        df["longitude"].between(
            CONFIG["min_longitude"],
            CONFIG["max_longitude"]
        )
    ]

    df = df[
        df["magnitude"]
        >=
        CONFIG["download_min_magnitude"]
    ]

    return (
        df
        .sort_values("time")
        .reset_index(drop=True)
    )


# ============================================================
# 10. DEDUPLICACIÓN
# ============================================================

def deduplicate_catalog(df):
    if df.empty:
        return df

    priority = {
        "USGS": 0,
        "EMSC": 1,
        "ISC": 2
    }

    data = (
        df.copy()
        .sort_values("time")
        .reset_index(drop=True)
    )

    data["_priority"] = (
        data["source"]
        .map(priority)
        .fillna(99)
    )

    keep = np.ones(
        len(data),
        dtype=bool
    )

    times = (
        data["time"]
        .astype("int64")
        .values
        /
        1e9
    )

    for i in range(len(data)):
        if not keep[i]:
            continue

        j = i + 1

        while j < len(data):
            dt = times[j] - times[i]

            if dt > 30:
                break

            dm = abs(
                data.iloc[j]["magnitude"]
                -
                data.iloc[i]["magnitude"]
            )

            if dm <= 0.5:
                distance = haversine_km(
                    data.iloc[i]["latitude"],
                    data.iloc[i]["longitude"],
                    data.iloc[j]["latitude"],
                    data.iloc[j]["longitude"]
                )

                if distance <= 50:
                    if (
                        data.iloc[j]["_priority"]
                        <
                        data.iloc[i]["_priority"]
                    ):
                        keep[i] = False
                        break
                    else:
                        keep[j] = False

            j += 1

    return (
        data.loc[keep]
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


# ============================================================
# 11. Mc / B-VALUE
# ============================================================

def estimate_mc_maximum_curvature(
    magnitudes,
    bin_width=0.1
):
    mags = np.asarray(
        magnitudes,
        dtype=float
    )

    mags = mags[np.isfinite(mags)]

    if len(mags) < 20:
        return np.nan

    minimum = (
        np.floor(mags.min() / bin_width)
        *
        bin_width
    )

    maximum = (
        np.ceil(mags.max() / bin_width)
        *
        bin_width
    )

    bins = np.arange(
        minimum,
        maximum + 2 * bin_width,
        bin_width
    )

    hist, edges = np.histogram(
        mags,
        bins=bins
    )

    idx = np.argmax(hist)
    mc = edges[idx]

    return float(np.round(mc, 2))


def estimate_b_value(
    magnitudes,
    mc,
    delta_m=0.1
):
    mags = np.asarray(
        magnitudes,
        dtype=float
    )

    mags = mags[mags >= mc]

    if len(mags) < 20:
        return np.nan

    mean_mag = np.mean(mags)

    denom = (
        mean_mag
        -
        (
            mc
            -
            delta_m / 2.0
        )
    )

    if denom <= 0:
        return np.nan

    b = np.log10(np.e) / denom

    return float(b)


# ============================================================
# 12. ETAS TEMPORAL
# ============================================================

@njit
def etas_negative_loglikelihood(
    params,
    times,
    magnitudes,
    mc,
    duration_days
):
    mu = params[0]
    K = params[1]
    alpha = params[2]
    c = params[3]
    p = params[4]

    n = len(times)

    if (
        mu <= 0
        or
        K <= 0
        or
        c <= 0
        or
        p <= 1
    ):
        return 1e100

    loglik = 0.0

    for i in range(n):
        lam = mu

        for j in range(i):
            dt = times[i] - times[j]

            if dt > 0:
                lam += (
                    K
                    *
                    np.exp(
                        alpha
                        *
                        (
                            magnitudes[j]
                            -
                            mc
                        )
                    )
                    *
                    (
                        dt + c
                    )
                    ** (-p)
                )

        if (
            lam <= 0
            or
            not np.isfinite(lam)
        ):
            return 1e100

        loglik += np.log(lam)

    integral = mu * duration_days

    for j in range(n):
        upper = duration_days - times[j]

        if upper <= 0:
            continue

        productivity = (
            K
            *
            np.exp(
                alpha
                *
                (
                    magnitudes[j]
                    -
                    mc
                )
            )
        )

        temporal_int = (
            (
                (upper + c)
                ** (1.0 - p)
                -
                c
                ** (1.0 - p)
            )
            /
            (1.0 - p)
        )

        integral += (
            productivity
            *
            temporal_int
        )

    return -loglik + integral


def fit_etas_temporal(df, mc):
    data = (
        df[
            df["magnitude"]
            >=
            mc
        ]
        .sort_values("time")
        .copy()
    )

    if len(data) < 30:
        raise RuntimeError(
            "Muy pocos eventos para ajustar ETAS."
        )

    if (
        len(data)
        >
        CONFIG["max_fit_events"]
    ):
        data = data.iloc[
            -CONFIG["max_fit_events"]:
        ].copy()

    t0 = data["time"].iloc[0]

    times = (
        (
            data["time"]
            -
            t0
        )
        .dt.total_seconds()
        .values
        /
        86400.0
    )

    magnitudes = (
        data["magnitude"]
        .values
        .astype(float)
    )

    duration_days = max(
        float(times[-1]),
        1e-6
    )

    bounds = [
        (1e-7, 100.0),
        (1e-8, 10.0),
        (0.0, 5.0),
        (1e-5, 5.0),
        (1.001, 3.0)
    ]

    starts = [
        [0.05, 0.01, 0.8, 0.005, 1.10],
        [0.10, 0.02, 1.0, 0.010, 1.20],
        [0.30, 0.03, 1.2, 0.020, 1.30],
        [0.50, 0.05, 1.5, 0.050, 1.10],
        [1.00, 0.01, 0.6, 0.100, 1.40]
    ]

    results = []

    for x0 in starts[
        :CONFIG["optimizer_restarts"]
    ]:
        result = minimize(
            etas_negative_loglikelihood,
            x0=x0,
            args=(
                times,
                magnitudes,
                mc,
                duration_days
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 1000
            }
        )

        if np.isfinite(result.fun):
            results.append(result)

    if not results:
        raise RuntimeError(
            "No fue posible ajustar ETAS."
        )

    successful = [
        result
        for result in results
        if result.success
    ]

    if successful:
        best = min(
            successful,
            key=lambda x:
                x.fun
        )
    else:
        best = min(
            results,
            key=lambda x:
                x.fun
        )

    params = {
        "mu": float(best.x[0]),
        "K": float(best.x[1]),
        "alpha": float(best.x[2]),
        "c": float(best.x[3]),
        "p": float(best.x[4]),
        "negative_loglikelihood": float(best.fun),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "fit_events": int(len(data))
    }

    return params, data


# ============================================================
# 13. VALIDACIÓN TEMPORAL
# ============================================================

def temporal_etas_intensity_at_times(
    validation_times,
    history_times,
    history_mags,
    params,
    mc
):
    mu = params["mu"]
    K = params["K"]
    alpha = params["alpha"]
    c = params["c"]
    p = params["p"]

    intensities = []

    all_times = list(history_times)
    all_mags = list(history_mags)

    for vt, vm in validation_times:
        lam = mu

        for ht, hm in zip(
            all_times,
            all_mags
        ):
            dt = vt - ht

            if dt > 0:
                lam += (
                    K
                    *
                    np.exp(
                        alpha
                        *
                        (
                            hm - mc
                        )
                    )
                    *
                    (
                        dt + c
                    )
                    ** (-p)
                )

        intensities.append(lam)

        all_times.append(vt)
        all_mags.append(vm)

    return np.asarray(intensities)


def validate_model(
    catalog,
    forecast_start
):
    validation_start = (
        forecast_start
        -
        pd.Timedelta(
            days=CONFIG[
                "validation_days"
            ]
        )
    )

    training_start = (
        validation_start
        -
        pd.DateOffset(
            years=CONFIG[
                "training_years"
            ]
        )
    )

    train = catalog[
        (
            catalog["time"]
            >=
            training_start
        )
        &
        (
            catalog["time"]
            <
            validation_start
        )
    ].copy()

    validation = catalog[
        (
            catalog["time"]
            >=
            validation_start
        )
        &
        (
            catalog["time"]
            <
            forecast_start
        )
    ].copy()

    mc_train = (
        estimate_mc_maximum_curvature(
            train["magnitude"].values,
            CONFIG["magnitude_bin"]
        )
    )

    if not np.isfinite(mc_train):
        return {
            "status":
                "insufficient_data"
        }

    try:
        params_train, _ = (
            fit_etas_temporal(
                train,
                mc_train
            )
        )

    except Exception as exc:
        return {
            "status": "fit_failed",
            "error": str(exc)
        }

    train_mc = train[
        train["magnitude"]
        >=
        mc_train
    ].copy()

    validation_mc = validation[
        validation["magnitude"]
        >=
        mc_train
    ].copy()

    if len(validation_mc) == 0:
        return {
            "status": "no_validation_events",
            "mc": mc_train
        }

    reference = train_mc["time"].iloc[0]

    history_times = (
        (
            train_mc["time"]
            -
            reference
        )
        .dt.total_seconds()
        .values
        /
        86400.0
    )

    history_mags = (
        train_mc["magnitude"]
        .values
    )

    validation_sequence = []

    for _, row in (
        validation_mc
        .sort_values("time")
        .iterrows()
    ):
        t = (
            (
                row["time"]
                -
                reference
            )
            .total_seconds()
            /
            86400.0
        )

        validation_sequence.append(
            (
                t,
                row["magnitude"]
            )
        )

    intensities = (
        temporal_etas_intensity_at_times(
            validation_sequence,
            history_times,
            history_mags,
            params_train,
            mc_train
        )
    )

    training_duration = max(
        (
            train_mc["time"].max()
            -
            train_mc["time"].min()
        )
        .total_seconds()
        /
        86400.0,
        1e-6
    )

    poisson_rate = (
        len(train_mc)
        /
        training_duration
    )

    poisson_rate = max(
        poisson_rate,
        1e-12
    )

    IG_TOTAL = float(
        np.sum(
            np.log(
                np.maximum(
                    intensities,
                    1e-12
                )
                /
                poisson_rate
            )
        )
    )

    IG_PER_EVENT = (
        IG_TOTAL
        /
        len(validation_mc)
    )

    return {
        "status": "ok",
        "mc": float(mc_train),
        "training_events": int(len(train_mc)),
        "validation_events": int(len(validation_mc)),
        "IG_TOTAL": float(IG_TOTAL),
        "IG_PER_EVENT": float(IG_PER_EVENT),
        "poisson_rate": float(poisson_rate),
        "etas_params": params_train
    }


# ============================================================
# 14. GRID ESPACIAL
# ============================================================

def build_spatial_grid():
    resolution = CONFIG[
        "grid_resolution_deg"
    ]

    latitudes = np.arange(
        CONFIG["min_latitude"],
        CONFIG["max_latitude"]
        +
        resolution,
        resolution
    )

    longitudes = np.arange(
        CONFIG["min_longitude"],
        CONFIG["max_longitude"]
        +
        resolution,
        resolution
    )

    LON_GRID, LAT_GRID = (
        np.meshgrid(
            longitudes,
            latitudes
        )
    )

    return (
        latitudes,
        longitudes,
        LAT_GRID,
        LON_GRID
    )


def approximate_cell_area_km2(
    lat_grid,
    resolution_deg
):
    lat_km = (
        111.32
        *
        resolution_deg
    )

    lon_km = (
        111.32
        *
        np.cos(
            np.radians(
                lat_grid
            )
        )
        *
        resolution_deg
    )

    return lat_km * lon_km


def estimate_spatial_scale_km(df):
    coords = np.column_stack([
        df["latitude"].values,
        df["longitude"].values
    ])

    if len(coords) < 2:
        return CONFIG[
            "spatial_d_min_km"
        ]

    mean_lat = np.radians(
        np.mean(
            coords[:, 0]
        )
    )

    x = (
        coords[:, 1]
        *
        111.32
        *
        np.cos(mean_lat)
    )

    y = (
        coords[:, 0]
        *
        111.32
    )

    xy = np.column_stack([x, y])

    tree = cKDTree(xy)

    distances, _ = tree.query(
        xy,
        k=2
    )

    nearest = distances[:, 1]

    d = np.nanmedian(nearest)

    return float(
        np.clip(
            d,
            CONFIG[
                "spatial_d_min_km"
            ],
            CONFIG[
                "spatial_d_max_km"
            ]
        )
    )


def build_background_spatial_pdf(
    catalog,
    lat_grid,
    lon_grid
):
    latitudes = lat_grid[:, 0]
    longitudes = lon_grid[0, :]

    histogram = np.zeros(
        lat_grid.shape,
        dtype=float
    )

    lat_idx = np.searchsorted(
        latitudes,
        catalog["latitude"].values
    )

    lon_idx = np.searchsorted(
        longitudes,
        catalog["longitude"].values
    )

    lat_idx = np.clip(
        lat_idx,
        0,
        len(latitudes) - 1
    )

    lon_idx = np.clip(
        lon_idx,
        0,
        len(longitudes) - 1
    )

    for i, j in zip(
        lat_idx,
        lon_idx
    ):
        histogram[i, j] += 1.0

    smoothed = gaussian_filter(
        histogram,
        sigma=CONFIG[
            "background_sigma_cells"
        ]
    )

    total = smoothed.sum()

    if total <= 0:
        smoothed[:] = 1.0
        total = smoothed.sum()

    return smoothed / total


def event_grid_distance_km(
    event_lat,
    event_lon,
    lat_grid,
    lon_grid
):
    mean_lat = np.radians(
        event_lat
    )

    dx = (
        (
            lon_grid
            -
            event_lon
        )
        *
        111.32
        *
        np.cos(mean_lat)
    )

    dy = (
        (
            lat_grid
            -
            event_lat
        )
        *
        111.32
    )

    return np.sqrt(
        dx ** 2
        +
        dy ** 2
    )


def spatial_kernel_density(
    distance_km,
    d_km,
    q
):
    return (
        (q - 1.0)
        /
        (
            np.pi
            *
            d_km ** 2
        )
        *
        (
            1.0
            +
            (
                distance_km
                /
                d_km
            )
            ** 2
        )
        ** (-q)
    )


def future_omori_integral(
    age_days,
    horizon_days,
    c,
    p
):
    lower = age_days + c
    upper = (
        age_days
        +
        horizon_days
        +
        c
    )

    return (
        upper
        ** (1.0 - p)
        -
        lower
        ** (1.0 - p)
    ) / (1.0 - p)


# ============================================================
# 15. FORECAST ESPACIAL
# ============================================================

def calculate_forecast_grid(
    catalog,
    forecast_start,
    params,
    mc,
    b_value,
    horizon_days,
    magnitude_threshold,
    lat_grid,
    lon_grid,
    cell_area,
    background_pdf,
    spatial_d_km
):
    mu = params["mu"]
    K = params["K"]
    alpha = params["alpha"]
    c = params["c"]
    p = params["p"]

    expected_mc = (
        mu
        *
        horizon_days
        *
        background_pdf
    )

    memory_start = (
        forecast_start
        -
        pd.Timedelta(
            days=CONFIG[
                "trigger_memory_days"
            ]
        )
    )

    triggering_catalog = catalog[
        (
            catalog["time"]
            >=
            memory_start
        )
        &
        (
            catalog["time"]
            <
            forecast_start
        )
        &
        (
            catalog["magnitude"]
            >=
            mc
        )
    ].copy()

    q = CONFIG["spatial_q"]

    for _, event in (
        triggering_catalog.iterrows()
    ):
        age_days = (
            forecast_start
            -
            event["time"]
        ).total_seconds() / 86400.0

        if age_days < 0:
            continue

        productivity = (
            K
            *
            np.exp(
                alpha
                *
                (
                    event["magnitude"]
                    -
                    mc
                )
            )
        )

        temporal_mass = (
            productivity
            *
            future_omori_integral(
                age_days,
                horizon_days,
                c,
                p
            )
        )

        if (
            temporal_mass <= 0
            or
            not np.isfinite(
                temporal_mass
            )
        ):
            continue

        distances = (
            event_grid_distance_km(
                event["latitude"],
                event["longitude"],
                lat_grid,
                lon_grid
            )
        )

        spatial_density = (
            spatial_kernel_density(
                distances,
                spatial_d_km,
                q
            )
        )

        spatial_mass = (
            spatial_density
            *
            cell_area
        )

        total_mass = spatial_mass.sum()

        if total_mass > 0:
            spatial_mass /= total_mass

        expected_mc += (
            temporal_mass
            *
            spatial_mass
        )

    if magnitude_threshold > mc:
        magnitude_factor = (
            10.0
            **
            (
                -b_value
                *
                (
                    magnitude_threshold
                    -
                    mc
                )
            )
        )
    else:
        magnitude_factor = 1.0

    expected_threshold = (
        expected_mc
        *
        magnitude_factor
    )

    probability = (
        1.0
        -
        np.exp(
            -expected_threshold
        )
    )

    domain_lambda = float(
        expected_threshold.sum()
    )

    domain_probability = float(
        1.0
        -
        np.exp(
            -domain_lambda
        )
    )

    return {
        "expected": expected_threshold,
        "probability": probability,
        "domain_lambda": domain_lambda,
        "domain_probability": domain_probability,
        "triggering_events": int(
            len(triggering_catalog)
        )
    }


# ============================================================
# 16. TOP ZONAS
# ============================================================

def select_top_geographic_zones(
    probability_grid,
    lat_grid,
    lon_grid,
    number_zones=5,
    minimum_separation_km=150.0
):
    flat_prob = probability_grid.ravel()

    order = np.argsort(
        flat_prob
    )[::-1]

    selected = []

    rows, cols = (
        probability_grid.shape
    )

    for flat_index in order:
        row, col = np.unravel_index(
            flat_index,
            (rows, cols)
        )

        lat = float(
            lat_grid[row, col]
        )

        lon = float(
            lon_grid[row, col]
        )

        prob = float(
            probability_grid[row, col]
        )

        sufficiently_far = True

        for zone in selected:
            distance = haversine_km(
                lat,
                lon,
                zone["latitude"],
                zone["longitude"]
            )

            if (
                distance
                <
                minimum_separation_km
            ):
                sufficiently_far = False
                break

        if sufficiently_far:
            selected.append({
                "row": int(row),
                "column": int(col),
                "latitude": lat,
                "longitude": lon,
                "probability": prob,
                "probability_percent": prob * 100.0
            })

        if len(selected) >= number_zones:
            break

    return pd.DataFrame(selected)


def expected_magnitude_band_for_cell(
    row,
    column,
    horizon,
    forecast_results
):
    lambda_4 = (
        forecast_results[
            horizon
        ][4.0]["expected"][
            row,
            column
        ]
    )

    lambda_5 = (
        forecast_results[
            horizon
        ][5.0]["expected"][
            row,
            column
        ]
    )

    lambda_6 = (
        forecast_results[
            horizon
        ][6.0]["expected"][
            row,
            column
        ]
    )

    lambda_7 = (
        forecast_results[
            horizon
        ][7.0]["expected"][
            row,
            column
        ]
    )

    bands = {
        "4.0–4.9": max(
            lambda_4 - lambda_5,
            0
        ),
        "5.0–5.9": max(
            lambda_5 - lambda_6,
            0
        ),
        "6.0–6.9": max(
            lambda_6 - lambda_7,
            0
        ),
        "≥7.0": max(
            lambda_7,
            0
        )
    }

    best_band = max(
        bands,
        key=bands.get
    )

    return (
        best_band,
        float(
            bands[best_band]
        )
    )


# ============================================================
# 17. LÁMINA RRSS
# ============================================================

def build_poster(
    forecast_start,
    etas_catalog,
    forecast_results,
    top_zones,
    lat_grid,
    lon_grid,
    mc,
    b_value,
    d_km
):
    main_horizon = POSTER_CONFIG[
        "main_horizon"
    ]

    main_magnitude = POSTER_CONFIG[
        "main_magnitude"
    ]

    main_forecast = (
        forecast_results[
            main_horizon
        ][
            main_magnitude
        ]
    )

    fig = plt.figure(
        figsize=POSTER_CONFIG[
            "figsize"
        ],
        constrained_layout=False
    )

    fig.patch.set_facecolor(
        POSTER_CONFIG[
            "background_color"
        ]
    )

    gs = GridSpec(
        28,
        28,
        figure=fig,
        left=.035,
        right=.965,
        top=.975,
        bottom=.035,
        hspace=1.10,
        wspace=1.00
    )

    # HEADER
    ax_header = fig.add_subplot(
        gs[0:4, 0:28]
    )

    ax_header.axis("off")

    header_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle=
            "round,pad=0.012,rounding_size=0.025",
        transform=ax_header.transAxes,
        linewidth=0,
        facecolor=
            POSTER_CONFIG[
                "header_color"
            ]
    )

    ax_header.add_patch(header_box)

    ax_header.text(
        0.035,
        0.72,
        POSTER_CONFIG[
            "title"
        ],
        transform=ax_header.transAxes,
        fontsize=18,
        fontweight="bold",
        color="white",
        va="center"
    )

    ax_header.text(
        0.035,
        0.43,
        POSTER_CONFIG[
            "subtitle"
        ],
        transform=ax_header.transAxes,
        fontsize=10,
        color="white",
        va="center"
    )

    update_text = (
        f"Actualización: "
        f"{forecast_start.strftime('%Y-%m-%d %H:%M UTC')}"
        f"   ·   Horizonte principal: 7 días"
        f"   ·   Umbral: M≥5"
    )

    ax_header.text(
        0.035,
        0.19,
        update_text,
        transform=ax_header.transAxes,
        fontsize=7.2,
        color="white",
        alpha=0.90,
        va="center"
    )

    # TÍTULO MAPA EN FILA PROPIA
    ax_map_title = fig.add_subplot(
        gs[4:5, 0:17]
    )

    ax_map_title.axis("off")

    ax_map_title.text(
        0.5,
        0.42,
        "PROBABILIDAD ETAS (PRÓXIMOS 7 DÍAS)",
        transform=ax_map_title.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="#172633",
        ha="center",
        va="center"
    )

    # MAPA
    ax_map = fig.add_subplot(
        gs[5:24, 0:17],
        projection=ccrs.PlateCarree()
    )

    ax_map.set_facecolor(
        "#EAF1F5"
    )

    ax_map.set_extent(
        [
            CONFIG["min_longitude"],
            CONFIG["max_longitude"],
            CONFIG["min_latitude"],
            CONFIG["max_latitude"]
        ],
        crs=ccrs.PlateCarree()
    )

    ax_map.add_feature(
        cfeature.LAND,
        facecolor="#ECE8DF",
        zorder=0
    )

    ax_map.add_feature(
        cfeature.OCEAN,
        facecolor="#DCEAF2",
        zorder=0
    )

    ax_map.add_feature(
        cfeature.COASTLINE,
        linewidth=0.6,
        edgecolor="#536875",
        zorder=3
    )

    ax_map.add_feature(
        cfeature.BORDERS,
        linewidth=0.4,
        linestyle=":",
        edgecolor="#748792",
        zorder=3
    )

    probability_map = (
        main_forecast[
            "probability"
        ]
    )

    positive_values = (
        probability_map[
            probability_map > 0
        ]
    )

    if len(positive_values) > 0:
        vmin = max(
            np.nanpercentile(
                positive_values,
                5
            ),
            1e-8
        )
    else:
        vmin = 1e-8

    vmax = max(
        np.nanmax(
            probability_map
        ),
        vmin * 10
    )

    mesh = ax_map.pcolormesh(
        lon_grid,
        lat_grid,
        np.maximum(
            probability_map,
            1e-12
        ),
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),
        shading="auto",
        zorder=1
    )

    # EVENTOS RECIENTES
    recent_start = (
        forecast_start
        -
        pd.Timedelta(
            days=POSTER_CONFIG[
                "recent_days"
            ]
        )
    )

    recent_events = etas_catalog[
        (
            etas_catalog["time"]
            >=
            recent_start
        )
        &
        (
            etas_catalog["time"]
            <
            forecast_start
        )
    ].copy()

    if len(recent_events) > 0:
        sizes = (
            np.maximum(
                recent_events[
                    "magnitude"
                ].values
                -
                2.5,
                0.3
            )
            ** 2
            *
            8
        )

        ax_map.scatter(
            recent_events[
                "longitude"
            ],
            recent_events[
                "latitude"
            ],
            s=sizes,
            facecolors="none",
            edgecolors="#37474F",
            linewidths=0.45,
            alpha=0.50,
            transform=ccrs.PlateCarree(),
            zorder=4
        )

    # TOP 5
    for _, zone in top_zones.iterrows():
        ax_map.scatter(
            zone["longitude"],
            zone["latitude"],
            marker="*",
            s=175,
            facecolor="white",
            edgecolor="#243B4A",
            linewidth=0.9,
            transform=ccrs.PlateCarree(),
            zorder=8
        )

        ax_map.annotate(
            str(
                int(
                    zone["rank"]
                )
            ),
            xy=(
                zone["longitude"],
                zone["latitude"]
            ),
            xytext=(11, 7),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#172633",
            bbox=dict(
                boxstyle=
                    "circle,pad=0.25",
                facecolor="white",
                edgecolor="#435A68",
                linewidth=0.7
            ),
            zorder=10
        )

    gl = ax_map.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="#82939C",
        alpha=0.40,
        linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {
        "size": 7
    }

    gl.ylabel_style = {
        "size": 7
    }

    cbar = fig.colorbar(
        mesh,
        ax=ax_map,
        orientation="horizontal",
        fraction=0.045,
        pad=0.055,
        aspect=35
    )

    cbar.ax.tick_params(
        labelsize=6.5
    )

    cbar.set_label(
        "Probabilidad local de ≥1 evento M≥5",
        fontsize=7.5,
        color="#172633"
    )

    # RANKING
    ax_rank = fig.add_subplot(
        gs[4:11, 17:28]
    )

    ax_rank.axis("off")

    rank_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle=
            "round,pad=0.012,rounding_size=0.025",
        transform=ax_rank.transAxes,
        facecolor="#FFFFFF",
        edgecolor="#C1CDD5",
        linewidth=0.7
    )

    ax_rank.add_patch(rank_box)

    ax_rank.text(
        0.05,
        0.90,
        "RANKING ZONAS MÁXIMA PROBABILIDAD",
        transform=ax_rank.transAxes,
        fontsize=8.3,
        fontweight="bold",
        color="#172633",
        va="center"
    )

    headers = [
        "#",
        "Lat.",
        "Lon.",
        "Mag.\nesperada",
        "Prob.\nlocal"
    ]

    x_positions = [
        0.07,
        0.22,
        0.39,
        0.64,
        0.88
    ]

    for x, header in zip(
        x_positions,
        headers
    ):
        ax_rank.text(
            x,
            0.72,
            header,
            transform=ax_rank.transAxes,
            fontsize=6.5,
            fontweight="bold",
            color="#657681",
            ha="center",
            va="center",
            linespacing=1.0
        )

    row_y = [
        0.57,
        0.45,
        0.33,
        0.21,
        0.09
    ]

    for idx, (_, zone) in enumerate(
        top_zones.iterrows()
    ):
        if idx >= len(row_y):
            break

        y = row_y[idx]

        row_color = (
            "#F5F8FA"
            if idx % 2 == 0
            else
            "#FFFFFF"
        )

        ax_rank.add_patch(
            FancyBboxPatch(
                (
                    0.035,
                    y - 0.043
                ),
                0.93,
                0.086,
                boxstyle=
                    "round,pad=0.005,rounding_size=0.01",
                transform=ax_rank.transAxes,
                facecolor=row_color,
                edgecolor="none"
            )
        )

        values = [
            str(int(zone["rank"])),
            f"{zone['latitude']:.2f}",
            f"{zone['longitude']:.2f}",
            zone["expected_magnitude_band"],
            f"{zone['probability_percent']:.2f}%"
        ]

        for x, value in zip(
            x_positions,
            values
        ):
            ax_rank.text(
                x,
                y,
                value,
                transform=ax_rank.transAxes,
                fontsize=6.5,
                color="#172633",
                ha="center",
                va="center",
                fontweight=(
                    "bold"
                    if x == 0.07
                    else
                    "normal"
                )
            )

    # RESUMEN
    ax_summary = fig.add_subplot(
        gs[11:19, 17:28]
    )

    ax_summary.axis("off")

    summary_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle=
            "round,pad=0.012,rounding_size=0.025",
        transform=ax_summary.transAxes,
        facecolor="#FFFFFF",
        edgecolor="#C1CDD5",
        linewidth=0.7
    )

    ax_summary.add_patch(
        summary_box
    )

    ax_summary.text(
        0.06,
        0.90,
        "RESUMEN DEL PRONÓSTICO",
        transform=ax_summary.transAxes,
        fontsize=8.7,
        fontweight="bold",
        color="#172633"
    )

    p24 = (
        forecast_results[
            "24h"
        ][5.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p7 = (
        forecast_results[
            "7d"
        ][5.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p30 = (
        forecast_results[
            "30d"
        ][5.0][
            "domain_probability"
        ]
        *
        100.0
    )

    summary_lines = [
        (
            "Magnitud de completitud",
            f"Mc = {mc:.2f}"
        ),
        (
            "b-value",
            f"{b_value:.2f}"
        ),
        (
            "Escala espacial",
            f"{d_km:.1f} km"
        ),
        (
            "P dominio · 24 h",
            f"{p24:.1f}%"
        ),
        (
            "P dominio · 7 días",
            f"{p7:.1f}%"
        ),
        (
            "P dominio · 30 días",
            f"{p30:.1f}%"
        )
    ]

    ys = np.linspace(
        0.73,
        0.14,
        len(summary_lines)
    )

    for y, (label, value) in zip(
        ys,
        summary_lines
    ):
        ax_summary.text(
            0.07,
            y,
            label,
            transform=ax_summary.transAxes,
            fontsize=7.4,
            color="#657681",
            va="center"
        )

        ax_summary.text(
            0.92,
            y,
            value,
            transform=ax_summary.transAxes,
            fontsize=7.8,
            fontweight="bold",
            color="#172633",
            ha="right",
            va="center"
        )

    # BARRAS
    ax_bar = fig.add_subplot(
        gs[19:24, 17:28]
    )

    ax_bar.set_facecolor(
        "#FFFFFF"
    )

    bar_values = [
        p24,
        p7,
        p30
    ]

    bar_labels = [
        "24 h",
        "7 días",
        "30 días"
    ]

    bar_colors = [
        "#8CB9D8",
        "#3B7EA1",
        "#163E59"
    ]

    bars = ax_bar.bar(
        bar_labels,
        bar_values,
        color=bar_colors,
        width=0.58
    )

    ax_bar.set_title(
        "PROBABILIDAD EN EL DOMINIO · M≥5",
        fontsize=8.0,
        fontweight="bold",
        color="#172633",
        pad=9
    )

    ax_bar.set_ylim(
        0,
        max(
            100,
            max(bar_values)
            *
            1.18
        )
    )

    ax_bar.set_ylabel(
        "%",
        fontsize=7
    )

    ax_bar.tick_params(
        axis="both",
        labelsize=7
    )

    ax_bar.spines[
        "top"
    ].set_visible(False)

    ax_bar.spines[
        "right"
    ].set_visible(False)

    ax_bar.spines[
        "left"
    ].set_color(
        "#C1CDD5"
    )

    ax_bar.spines[
        "bottom"
    ].set_color(
        "#C1CDD5"
    )

    for bar, value in zip(
        bars,
        bar_values
    ):
        ax_bar.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            bar.get_height()
            +
            max(
                1.0,
                max(bar_values)
                *
                0.025
            ),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#172633"
        )

    # FOOTER
    ax_footer = fig.add_subplot(
        gs[24:28, 0:28]
    )

    ax_footer.axis("off")

    footer_box = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle=
            "round,pad=0.012,rounding_size=0.025",
        transform=ax_footer.transAxes,
        facecolor="#FFFFFF",
        edgecolor="#C1CDD5",
        linewidth=0.7
    )

    ax_footer.add_patch(
        footer_box
    )

    footer_text = (
        "Prob. local = probabilidad de ≥1 evento en una celda del modelo durante "
        "el horizonte indicado. P dominio = probabilidad de ≥1 evento en toda el área analizada.\n\n"
        "Las estrellas identifican las celdas con mayor probabilidad local, "
        "separadas geográficamente para evitar concentrar el ranking en una misma zona.\n\n"
        "Modelo estadístico experimental ETAS. No constituye alerta sísmica ni "
        "predicción determinista. Fuentes sísmicas: USGS / EMSC."
    )

    ax_footer.text(
        0.03,
        0.52,
        footer_text,
        transform=ax_footer.transAxes,
        fontsize=6.4,
        color="#657681",
        va="center",
        linespacing=1.35,
        wrap=True
    )

    poster_png = (
        FIGURE_DIR
        /
        "ETAS_V2_RRSS_CUADRADO.png"
    )

    poster_pdf = (
        FIGURE_DIR
        /
        "ETAS_V2_RRSS_CUADRADO.pdf"
    )

    fig.savefig(
        poster_png,
        dpi=POSTER_CONFIG[
            "dpi"
        ],
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    fig.savefig(
        poster_pdf,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    return (
        poster_png,
        poster_pdf,
        p24,
        p7,
        p30
    )


# ============================================================
# 18. TEXTO PARA X
# ============================================================

def build_x_post_text(
    forecast_results,
    top_zones
):
    p4_24 = (
        forecast_results[
            "24h"
        ][4.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p4_7 = (
        forecast_results[
            "7d"
        ][4.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p4_30 = (
        forecast_results[
            "30d"
        ][4.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p5_24 = (
        forecast_results[
            "24h"
        ][5.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p5_7 = (
        forecast_results[
            "7d"
        ][5.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p5_30 = (
        forecast_results[
            "30d"
        ][5.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p6_24 = (
        forecast_results[
            "24h"
        ][6.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p6_7 = (
        forecast_results[
            "7d"
        ][6.0][
            "domain_probability"
        ]
        *
        100.0
    )

    p6_30 = (
        forecast_results[
            "30d"
        ][6.0][
            "domain_probability"
        ]
        *
        100.0
    )

    top = top_zones.iloc[0]

    top_lat = float(
        top["latitude"]
    )

    top_lon = float(
        top["longitude"]
    )

    top_prob = float(
        top[
            "probability_percent"
        ]
    )

    text = (
        "Actualización ETAS Chile 🇨🇱\n"
        f"M≥4 24h:{p4_24:.1f}% 7d:{p4_7:.1f}% 30d:{p4_30:.1f}%\n"
        f"M≥5 24h:{p5_24:.1f}% 7d:{p5_7:.1f}% 30d:{p5_30:.1f}%\n"
        f"M≥6 24h:{p6_24:.1f}% 7d:{p6_7:.1f}% 30d:{p6_30:.1f}%\n"
        f"Máx M≥5/7d: {top_lat:.2f},{top_lon:.2f} ({top_prob:.2f}%)"
    )

    if TWITTER_CONFIG[
        "include_disclaimer"
    ]:
        text += (
            "\nModelo estadístico experimental; "
            "no es alerta ni predicción determinista."
        )

    return text


# ============================================================
# 19. CREDENCIALES / PUBLICACIÓN X
# ============================================================

def load_x_credentials():
    required = [
        "X_API_KEY",
        "X_API_SECRET",
        "X_ACCESS_TOKEN",
        "X_ACCESS_TOKEN_SECRET"
    ]

    if not X_CREDENTIALS[
        "use_hardcoded_credentials"
    ]:
        raise RuntimeError(
            "Esta versión para GitHub está configurada "
            "para usar credenciales hardcoded."
        )

    credentials = {}

    for name in required:
        value = X_CREDENTIALS.get(
            name,
            ""
        )

        if (
            value is None
            or
            str(value).strip() == ""
            or
            str(value).startswith(
                "PEGA_AQUI"
            )
        ):
            raise RuntimeError(
                f"Debes completar '{name}' "
                "en X_CREDENTIALS."
            )

        credentials[name] = value

    return credentials


def publish_to_x(
    text,
    image_path
):
    credentials = (
        load_x_credentials()
    )

    auth = tweepy.OAuth1UserHandler(
        credentials["X_API_KEY"],
        credentials["X_API_SECRET"],
        credentials["X_ACCESS_TOKEN"],
        credentials[
            "X_ACCESS_TOKEN_SECRET"
        ]
    )

    api_v1 = tweepy.API(auth)

    client_v2 = tweepy.Client(
        consumer_key=credentials[
            "X_API_KEY"
        ],
        consumer_secret=credentials[
            "X_API_SECRET"
        ],
        access_token=credentials[
            "X_ACCESS_TOKEN"
        ],
        access_token_secret=credentials[
            "X_ACCESS_TOKEN_SECRET"
        ]
    )

    media_ids = None

    if TWITTER_CONFIG[
        "attach_image"
    ]:
        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"No existe la imagen: "
                f"{image_path}"
            )

        print(
            "\nSubiendo lámina a X..."
        )

        media = api_v1.media_upload(
            filename=str(
                image_path
            )
        )

        media_ids = [
            media.media_id
        ]

    print(
        "Creando publicación en X..."
    )

    if media_ids:
        response = (
            client_v2.create_tweet(
                text=text,
                media_ids=media_ids
            )
        )
    else:
        response = (
            client_v2.create_tweet(
                text=text
            )
        )

    return response


def x_publication_quality_check(
    poster_png,
    top_zones,
    mc,
    b_value,
    etas_params,
    forecast_results
):
    problems = []

    if not Path(
        poster_png
    ).exists():
        problems.append(
            "No existe la lámina PNG."
        )

    if (
        top_zones is None
        or
        len(top_zones) == 0
    ):
        problems.append(
            "No existe el ranking de zonas."
        )

    if not np.isfinite(mc):
        problems.append(
            "Mc no es válido."
        )

    if not np.isfinite(
        b_value
    ):
        problems.append(
            "b-value no es válido."
        )

    if (
        "optimizer_success"
        in
        etas_params
        and
        not etas_params[
            "optimizer_success"
        ]
    ):
        problems.append(
            "El ajuste ETAS no reportó convergencia."
        )

    for horizon in [
        "24h",
        "7d",
        "30d"
    ]:
        for magnitude in [
            4.0,
            5.0,
            6.0
        ]:
            try:
                p = (
                    forecast_results[
                        horizon
                    ][
                        magnitude
                    ][
                        "domain_probability"
                    ]
                )

                if not np.isfinite(p):
                    problems.append(
                        f"Probabilidad "
                        f"{horizon} / "
                        f"M≥{int(magnitude)} "
                        f"inválida."
                    )

            except Exception:
                problems.append(
                    f"No existe forecast "
                    f"{horizon} / "
                    f"M≥{int(magnitude)}."
                )

    return problems


# ============================================================
# 20. MAIN
# ============================================================

def main():
    print(
        "=" * 72
    )

    print(
        "ETAS V2 CHILE — EJECUCIÓN AUTOMÁTICA"
    )

    print(
        "=" * 72
    )

    catalog_frames = []

    if CONFIG[
        "download_usgs"
    ]:
        usgs_catalog = (
            download_catalog(
                USGS_CLIENT,
                "USGS"
            )
        )

        usgs_catalog = (
            quality_control(
                usgs_catalog
            )
        )

        print(
            "USGS:",
            len(usgs_catalog),
            "eventos"
        )

        catalog_frames.append(
            usgs_catalog
        )

    if CONFIG[
        "download_emsc"
    ]:
        emsc_catalog = (
            download_catalog(
                EMSC_CLIENT,
                "EMSC"
            )
        )

        emsc_catalog = (
            quality_control(
                emsc_catalog
            )
        )

        print(
            "EMSC:",
            len(emsc_catalog),
            "eventos"
        )

        catalog_frames.append(
            emsc_catalog
        )

    if not catalog_frames:
        raise RuntimeError(
            "No se descargó ningún catálogo."
        )

    catalog_all = pd.concat(
        catalog_frames,
        ignore_index=True
    )

    catalog_all = (
        quality_control(
            catalog_all
        )
    )

    catalog_dedup = (
        deduplicate_catalog(
            catalog_all
        )
    )

    print(
        "\nEventos después de deduplicación:",
        len(catalog_dedup)
    )

    if (
        CONFIG["etas_source"]
        in
        catalog_dedup[
            "source"
        ].unique()
    ):
        etas_catalog = (
            catalog_dedup[
                catalog_dedup[
                    "source"
                ]
                ==
                CONFIG[
                    "etas_source"
                ]
            ]
            .copy()
        )
    else:
        etas_catalog = (
            catalog_dedup.copy()
        )

    etas_catalog = (
        etas_catalog
        .sort_values("time")
        .reset_index(drop=True)
    )

    if etas_catalog.empty:
        raise RuntimeError(
            "El catálogo ETAS quedó vacío."
        )

    if (
        CONFIG[
            "forecast_start"
        ]
        is None
    ):
        forecast_start = (
            etas_catalog[
                "time"
            ].max()
        )
    else:
        forecast_start = (
            ensure_utc_timestamp(
                CONFIG[
                    "forecast_start"
                ]
            )
        )

    print(
        "\nInicio forecast:",
        forecast_start
    )

    training_start = (
        forecast_start
        -
        pd.DateOffset(
            years=CONFIG[
                "training_years"
            ]
        )
    )

    final_train = etas_catalog[
        (
            etas_catalog[
                "time"
            ]
            >=
            training_start
        )
        &
        (
            etas_catalog[
                "time"
            ]
            <
            forecast_start
        )
    ].copy()

    print(
        "Ventana entrenamiento:",
        training_start,
        "→",
        forecast_start
    )

    print(
        "Eventos entrenamiento:",
        len(final_train)
    )

    mc = (
        estimate_mc_maximum_curvature(
            final_train[
                "magnitude"
            ].values,
            CONFIG[
                "magnitude_bin"
            ]
        )
    )

    if not np.isfinite(mc):
        raise RuntimeError(
            "No se pudo estimar Mc."
        )

    b_value = (
        estimate_b_value(
            final_train[
                "magnitude"
            ].values,
            mc,
            CONFIG[
                "magnitude_bin"
            ]
        )
    )

    if not np.isfinite(
        b_value
    ):
        raise RuntimeError(
            "No se pudo estimar b-value."
        )

    print(
        "\nMc:",
        round(mc, 2)
    )

    print(
        "b-value:",
        round(
            b_value,
            3
        )
    )

    validation_results = (
        validate_model(
            etas_catalog,
            forecast_start
        )
    )

    print(
        "\nValidación temporal:"
    )

    print(
        json.dumps(
            validation_results,
            indent=2,
            default=str
        )
    )

    etas_params, fit_catalog = (
        fit_etas_temporal(
            final_train,
            mc
        )
    )

    print(
        "\nParámetros ETAS:"
    )

    print(
        json.dumps(
            etas_params,
            indent=2
        )
    )

    (
        grid_latitudes,
        grid_longitudes,
        lat_grid,
        lon_grid
    ) = build_spatial_grid()

    cell_area = (
        approximate_cell_area_km2(
            lat_grid,
            CONFIG[
                "grid_resolution_deg"
            ]
        )
    )

    final_train_mc = (
        final_train[
            final_train[
                "magnitude"
            ]
            >=
            mc
        ]
    )

    background_pdf = (
        build_background_spatial_pdf(
            final_train_mc,
            lat_grid,
            lon_grid
        )
    )

    d_km = (
        estimate_spatial_scale_km(
            final_train_mc
        )
    )

    print(
        "\nEscala espacial d:",
        round(
            d_km,
            2
        ),
        "km"
    )

    forecast_results = {}

    for (
        horizon_name,
        horizon_days
    ) in (
        CONFIG[
            "forecast_horizons_days"
        ].items()
    ):
        forecast_results[
            horizon_name
        ] = {}

        for magnitude_threshold in (
            CONFIG[
                "forecast_magnitudes"
            ]
        ):
            forecast_results[
                horizon_name
            ][
                magnitude_threshold
            ] = (
                calculate_forecast_grid(
                    etas_catalog,
                    forecast_start,
                    etas_params,
                    mc,
                    b_value,
                    horizon_days,
                    magnitude_threshold,
                    lat_grid,
                    lon_grid,
                    cell_area,
                    background_pdf,
                    d_km
                )
            )

    summary_rows = []

    for (
        horizon_name,
        horizon_days
    ) in (
        CONFIG[
            "forecast_horizons_days"
        ].items()
    ):
        for magnitude_threshold in (
            CONFIG[
                "forecast_magnitudes"
            ]
        ):
            result = (
                forecast_results[
                    horizon_name
                ][
                    magnitude_threshold
                ]
            )

            summary_rows.append({
                "horizon": horizon_name,
                "days": horizon_days,
                "magnitude_threshold":
                    magnitude_threshold,
                "expected_events":
                    result[
                        "domain_lambda"
                    ],
                "domain_probability":
                    result[
                        "domain_probability"
                    ],
                "domain_probability_percent":
                    100.0
                    *
                    result[
                        "domain_probability"
                    ]
            })

    forecast_summary = pd.DataFrame(
        summary_rows
    )

    print(
        "\nRESUMEN FORECAST"
    )

    print(
        forecast_summary.to_string(
            index=False
        )
    )

    main_horizon = POSTER_CONFIG[
        "main_horizon"
    ]

    main_magnitude = POSTER_CONFIG[
        "main_magnitude"
    ]

    main_forecast = (
        forecast_results[
            main_horizon
        ][
            main_magnitude
        ]
    )

    top_zones = (
        select_top_geographic_zones(
            main_forecast[
                "probability"
            ],
            lat_grid,
            lon_grid,
            number_zones=
                POSTER_CONFIG[
                    "top_zones"
                ],
            minimum_separation_km=
                POSTER_CONFIG[
                    "minimum_zone_separation_km"
                ]
        )
    )

    magnitude_bands = []

    for _, zone in (
        top_zones.iterrows()
    ):
        band, _ = (
            expected_magnitude_band_for_cell(
                int(
                    zone["row"]
                ),
                int(
                    zone["column"]
                ),
                main_horizon,
                forecast_results
            )
        )

        magnitude_bands.append(
            band
        )

    top_zones[
        "expected_magnitude_band"
    ] = magnitude_bands

    top_zones[
        "rank"
    ] = np.arange(
        1,
        len(top_zones) + 1
    )

    print(
        "\nTOP ZONAS"
    )

    print(
        top_zones[[
            "rank",
            "latitude",
            "longitude",
            "expected_magnitude_band",
            "probability_percent"
        ]].to_string(
            index=False
        )
    )

    # Guardar resultados
    catalog_dedup.to_parquet(
        DATA_DIR
        /
        "catalog_deduplicated.parquet",
        index=False
    )

    forecast_summary.to_csv(
        FORECAST_DIR
        /
        "forecast_summary.csv",
        index=False
    )

    top_zones.to_csv(
        FORECAST_DIR
        /
        "top_zones.csv",
        index=False
    )

    with open(
        MODEL_DIR
        /
        "etas_parameters.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "forecast_start":
                    str(
                        forecast_start
                    ),
                "mc": mc,
                "b_value": b_value,
                "spatial_d_km": d_km,
                "params": etas_params,
                "validation":
                    validation_results
            },
            f,
            indent=2,
            default=str,
            ensure_ascii=False
        )

    # Lámina
    (
        poster_png,
        poster_pdf,
        p24,
        p7,
        p30
    ) = build_poster(
        forecast_start,
        etas_catalog,
        forecast_results,
        top_zones,
        lat_grid,
        lon_grid,
        mc,
        b_value,
        d_km
    )

    print(
        "\nLámina PNG:",
        poster_png
    )

    print(
        "Lámina PDF:",
        poster_pdf
    )

    # Texto X
    x_post_text = (
        build_x_post_text(
            forecast_results,
            top_zones
        )
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "PREVIEW DE PUBLICACIÓN EN X"
    )

    print(
        "=" * 72
    )

    print(
        "\n"
        +
        x_post_text
    )

    print(
        "\nCaracteres:",
        len(x_post_text)
    )

    print(
        "=" * 72
    )

    publication_problems = (
        x_publication_quality_check(
            poster_png,
            top_zones,
            mc,
            b_value,
            etas_params,
            forecast_results
        )
    )

    if publication_problems:
        print(
            "\n⚠ PUBLICACIÓN BLOQUEADA"
        )

        for problem in (
            publication_problems
        ):
            print(
                "•",
                problem
            )

    else:
        print(
            "\n✓ Control previo superado."
        )

        if TWITTER_CONFIG[
            "enabled"
        ]:
            try:
                x_response = (
                    publish_to_x(
                        text=x_post_text,
                        image_path=poster_png
                    )
                )

                print(
                    "\n✓ PUBLICACIÓN REALIZADA EN X"
                )

                try:
                    print(
                        "ID:",
                        x_response.data[
                            "id"
                        ]
                    )
                except Exception:
                    pass

                print(
                    "\nRespuesta:"
                )

                print(
                    x_response
                )

            except Exception as exc:
                print(
                    "\n❌ ERROR AL PUBLICAR EN X"
                )

                print(
                    repr(exc)
                )

                if not TWITTER_CONFIG[
                    "fail_silently"
                ]:
                    raise

        else:
            print(
                "\nMODO PREVIEW"
            )

            print(
                "No se publicó nada en X."
            )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "PROCESO COMPLETADO"
    )

    print(
        "=" * 72
    )

    print(
        "\nInicio forecast:",
        forecast_start
    )

    print(
        "Mc:",
        round(
            mc,
            2
        )
    )

    print(
        "b-value:",
        round(
            b_value,
            3
        )
    )

    print(
        f"P(M≥5) 24 h: "
        f"{p24:.2f}%"
    )

    print(
        f"P(M≥5) 7 días: "
        f"{p7:.2f}%"
    )

    print(
        f"P(M≥5) 30 días: "
        f"{p30:.2f}%"
    )

    print(
        "\nModelo estadístico experimental."
    )

    print(
        "No constituye alerta sísmica "
        "ni predicción determinista."
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
