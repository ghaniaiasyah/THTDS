from pathlib import Path
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================
# CONFIG
# ==========================
APP_NAME = "Smart ETA Monitor"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚚",
    layout="wide",
)

# ==========================
# UI CSS
# ==========================
st.markdown("""
<style>
.block-container { padding-top: 2.6rem; padding-bottom: 2rem; max-width: 1200px; }
            /* Prevent title cropping */
h1, h2, h3 { margin-top: 0.25rem !important; padding-top: 0.15rem !important; }

h1, h2, h3 { letter-spacing: -0.02em; }
p, li { line-height: 1.55; }

/* KPI cards (HTML only) */
.kpi-card {
  padding: 16px 16px;
  margin-bottom: 12px;          /* penting: biar antar KPI ga dempet */
  border-radius: 16px;
  border: 1px solid rgba(49, 51, 63, 0.12);
  background: rgba(255,255,255,0.65);
}
.kpi-title { font-size: 0.85rem; opacity: 0.75; margin-bottom: 6px; }
.kpi-value { font-size: 1.6rem; font-weight: 750; margin: 0; }
.kpi-sub { font-size: 0.85rem; opacity: 0.7; margin-top: 6px; }

/* Badges */
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 700;
  border: 1px solid rgba(49, 51, 63, 0.12);
}
.badge-green { background: rgba(0, 200, 83, 0.10); }
.badge-red { background: rgba(255, 82, 82, 0.10); }
.badge-blue { background: rgba(33, 150, 243, 0.10); }

/* SAFE styling for Streamlit bordered containers (no padding/shadow tweaks) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: 18px !important;
  overflow: hidden;
  margin-bottom: 14px;          /* penting: biar card tidak “ketiban” */
}

section[data-testid="stSidebar"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================
# CONSTANTS (from your notebook)
# ==========================
MAE_TEST_REF = 6.36
RMSE_TEST_REF = 9.21
R2_TEST_REF = 0.811

LATE_THRESHOLD_DEFAULT = 70.0

ARTIFACT_MODEL = Path("artifacts/eta_ridge_pipeline.pkl")
ARTIFACT_EVAL = Path("artifacts/eval_test.csv")

TARGET = "Delivery_Time_min"
RAW_FEATURES = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

# ==========================
# UI HELPERS
# ==========================
def kpi_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

@contextmanager
def card():
    """
    Proper Streamlit 'card' wrapper.
    IMPORTANT: This wraps real Streamlit components (no empty HTML boxes).
    """
    try:
        with st.container(border=True):
            yield
    except TypeError:
        # fallback if border=True not supported (shouldn't happen on 1.37)
        with st.container():
            yield

# ==========================
# LOADERS
# ==========================
@st.cache_resource
def load_pipeline():
    if not ARTIFACT_MODEL.exists():
        return None
    return joblib.load(ARTIFACT_MODEL)

@st.cache_data
def load_eval_df():
    if not ARTIFACT_EVAL.exists():
        return None
    return pd.read_csv(ARTIFACT_EVAL)

def ensure_columns(df: pd.DataFrame, required_cols: list[str]):
    missing = [c for c in required_cols if c not in df.columns]
    return (len(missing) == 0, missing)

# ==========================
# CORE
# ==========================
def predict_eta(pipe, input_row: dict) -> float:
    df = pd.DataFrame([input_row])
    return float(pipe.predict(df)[0])

def make_eval_table(pipe, eval_df: pd.DataFrame) -> pd.DataFrame:
    X = eval_df[RAW_FEATURES].copy()
    y = eval_df[TARGET].astype(float).values
    y_pred = pipe.predict(X).astype(float)

    out = X.copy()
    out[TARGET] = y
    out["y_pred"] = y_pred
    out["residual"] = y - y_pred
    out["abs_error"] = np.abs(out["residual"])
    return out

def compute_metrics(df_res: pd.DataFrame):
    y = df_res[TARGET].values
    yp = df_res["y_pred"].values
    mae = mean_absolute_error(y, yp)
    rmse = mean_squared_error(y, yp, squared=False)
    r2 = r2_score(y, yp)
    return mae, rmse, r2

# ==========================
# PLOTS
# ==========================
def plot_actual_vs_pred(df_res: pd.DataFrame):
    y = df_res[TARGET].values
    yp = df_res["y_pred"].values
    fig, ax = plt.subplots()
    ax.scatter(y, yp, alpha=0.6)
    lo = float(min(y.min(), yp.min()))
    hi = float(max(y.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_title("Actual vs Predicted (ETA)")
    ax.set_xlabel("Actual (min)")
    ax.set_ylabel("Predicted (min)")
    return fig

def plot_residual_hist(df_res: pd.DataFrame):
    r = df_res["residual"].values
    fig, ax = plt.subplots()
    ax.hist(r, bins=30)
    ax.set_title("Residual Distribution (Actual - Predicted)")
    ax.set_xlabel("Residual (min)")
    ax.set_ylabel("Count")
    return fig

def plot_residual_vs_pred(df_res: pd.DataFrame):
    r = df_res["residual"].values
    yp = df_res["y_pred"].values
    fig, ax = plt.subplots()
    ax.scatter(yp, r, alpha=0.6)
    ax.axhline(0)
    ax.set_title("Residual vs Predicted (Heteroscedasticity check)")
    ax.set_xlabel("Predicted (min)")
    ax.set_ylabel("Residual (min)")
    return fig

def segment_mae_tables(df_res: pd.DataFrame):
    """
    Return tables MAE by:
    - distance_bin: Near / Mid / Far (lebih mudah dibaca daripada interval qcut)
    - traffic: Low/Medium/High/Unknown
    - pred_decile: D1..D10
    """
    df = df_res.copy()

    # Distance bins -> label Near/Mid/Far
    df["distance_bin"] = pd.qcut(
        df["Distance_km"], q=3, labels=["Near", "Mid", "Far"], duplicates="drop"
    )
    by_dist = (
        df.groupby("distance_bin", dropna=False)["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "MAE"})
    )
    by_dist["MAE"] = by_dist["MAE"].round(2)

    # Traffic bins (handle None -> Unknown)
    df["Traffic_Level"] = df["Traffic_Level"].fillna("Unknown")
    by_traffic = (
        df.groupby("Traffic_Level", dropna=False)["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "MAE"})
    )
    by_traffic["MAE"] = by_traffic["MAE"].round(2)

    # Pred decile
    df["pred_decile"] = pd.qcut(
        df["y_pred"],
        q=10,
        labels=[f"D{i}" for i in range(1, 11)],
        duplicates="drop",
    )
    by_decile = (
        df.groupby("pred_decile", dropna=False)["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "MAE"})
    )
    by_decile["MAE"] = by_decile["MAE"].round(2)

    return by_dist, by_traffic, by_decile


def plot_bar_mae(df: pd.DataFrame, x_col: str, title: str):
    """
    Simple bar chart MAE by category.
    """
    fig, ax = plt.subplots()
    ax.bar(df[x_col].astype(str), df["MAE"].astype(float))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("MAE (min)")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    return fig


def takeaways_overview(mae: float, rmse: float, r2: float) -> str:
    return f"""
- Model cukup akurat pada eval set: **MAE {mae:.2f} min**, **RMSE {rmse:.2f}**, **R² {r2:.3f}**.
- Error cenderung membesar pada **ETA tinggi** (lihat residual menyebar saat prediksi tinggi).
- Model cocok untuk **prioritization** (lift Top bucket lebih tinggi dari baseline → pantau order berisiko duluan).
"""


def takeaways_residual() -> str:
    return """
**Asumsi Regresi (dikaitkan dengan residual study):**
- **Linearity**: cukup oke (actual vs predicted mengikuti diagonal).
- **Independence**: tidak tampak pola berulang kuat dari plot (idealnya dicek juga pakai urutan waktu bila ada).
- **Homoscedasticity**: *tidak sepenuhnya terpenuhi* → residual makin menyebar pada prediksi tinggi (**heteros attaching** ringan).
- **Normality**: histogram residual mendekati normal, ada ekor/outlier.
"""


def takeaways_segment(by_dist: pd.DataFrame, by_traffic: pd.DataFrame, by_decile: pd.DataFrame) -> str:
    # ambil worst category untuk storytelling
    worst_dist = by_dist.sort_values("MAE", ascending=False).iloc[0].to_dict()
    worst_traf = by_traffic.sort_values("MAE", ascending=False).iloc[0].to_dict()
    worst_dec = by_decile.sort_values("MAE", ascending=False).iloc[0].to_dict()

    return f"""
- **Distance**: error tertinggi di **{worst_dist['distance_bin']}** (MAE ≈ {worst_dist['MAE']:.2f}).
- **Traffic**: error tertinggi di **{worst_traf['Traffic_Level']}** (MAE ≈ {worst_traf['MAE']:.2f}).
- **Predicted decile**: error tertinggi di **{worst_dec['pred_decile']}** (MAE ≈ {worst_dec['MAE']:.2f}) → kasus ekstrem lebih sulit.
"""


def takeaways_lift(base_rate: float) -> str:
    return f"""
- **Base late rate** (actual > threshold) = **{base_rate*100:.1f}%**.
- Jika tim hanya bisa memonitor sebagian order, ranking berdasarkan **predicted ETA** membantu fokus ke order yang lebih berisiko.
- Lift tinggi di bucket awal → strategi **prioritization** efektif dibanding monitoring acak.
"""


def lift_table(df_res: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df_res.copy()
    df["is_late"] = (df[TARGET] > threshold).astype(int)

    df = df.sort_values("y_pred", ascending=False).reset_index(drop=True)
    df["bucket"] = pd.qcut(df.index + 1, q=10, labels=[f"Top {i*10}%" for i in range(1, 11)])

    base_rate = df["is_late"].mean()
    tbl = df.groupby("bucket")["is_late"].agg(["mean", "sum", "count"]).reset_index()
    tbl = tbl.rename(columns={"mean": "late_rate", "sum": "late_count"})
    tbl["base_rate"] = base_rate
    tbl["lift"] = np.where(base_rate > 0, tbl["late_rate"] / base_rate, np.nan)
    return tbl

def plot_lift(tbl: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(tbl["bucket"], tbl["lift"], marker="o")
    ax.set_title("Lift by Decile (ranked by predicted ETA)")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Lift")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return fig

# ==========================
# EXPLAINABILITY
# ==========================
def try_get_coeff_table(pipe):
    try:
        model = None
        for _, step in pipe.named_steps.items():
            if hasattr(step, "coef_"):
                model = step
                break
        if model is None:
            return None

        coefs = np.array(model.coef_).ravel()

        feat_names = None
        for _, step in pipe.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                try:
                    feat_names = step.get_feature_names_out()
                    break
                except Exception:
                    pass

        if feat_names is None:
            feat_names = [f"f{i}" for i in range(len(coefs))]
        else:
            feat_names = list(feat_names)

        if len(feat_names) != len(coefs):
            feat_names = [f"f{i}" for i in range(len(coefs))]

        df = pd.DataFrame({"feature": feat_names, "coef": coefs})
        df["abs_coef"] = df["coef"].abs()
        df = df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
        return df
    except Exception:
        return None

def plot_top_coefficients(coef_df: pd.DataFrame, top_n: int = 20):
    df = coef_df.copy()
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=True).tail(top_n)
    fig, ax = plt.subplots()
    ax.barh(df["feature"], df["coef"])
    ax.axvline(0)
    ax.set_title(f"Top {top_n} Ridge Coefficients (Signed)")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    return fig

# ==========================
# NAV
# ==========================
st.sidebar.title(f"{APP_NAME} 🚚")
st.sidebar.markdown("Prediksi ETA + evaluasi model (residual, segment error, lift/gain).")

if "page" not in st.session_state:
    st.session_state.page = "Beranda"

def go_to_predict():
    st.session_state.page = "Mulai Prediksi"

page = st.sidebar.radio(
    "Navigasi",
    ["Beranda", "Tentang Sistem", "Teknologi & Proses", "Mulai Prediksi", "Insight Model"],
    key="page",
)

pipe = load_pipeline()
eval_df = load_eval_df()

# ==========================
# PAGE: BERANDA
# ==========================
if page == "Beranda":
    st.markdown(f"# {APP_NAME} 🚚")
    st.markdown(
        "Dashboard untuk **prediksi ETA**, **diagnostik residual**, dan **prioritisasi late-risk** "
        "pada operasional last-mile delivery."
    )
    st.markdown("---")

    top = st.columns([1.3, 1, 1, 1])

    with top[0]:
        with card():
            st.markdown("### What you can do")
            st.markdown(
                """
                - Prediksi **ETA (menit)** dari input raw features  
                - Cek **model quality**: actual vs predicted, residual study  
                - Lihat **error by segment** (distance / traffic / decile)  
                - Gunakan **late-risk lift** untuk prioritisasi
                """
            )
            st.button("🚀 Mulai Prediksi", on_click=go_to_predict)

    with top[1]:
        kpi_card("MAE (Test – ref)", f"{MAE_TEST_REF:.2f}", "Lebih kecil lebih baik")
    with top[2]:
        kpi_card("RMSE (Test – ref)", f"{RMSE_TEST_REF:.2f}", "Menekan error besar")
    with top[3]:
        kpi_card("R² (Test – ref)", f"{R2_TEST_REF:.3f}", "Explained variance")

    st.markdown("")
    b1, b2 = st.columns([1.3, 1])

    with b1:
        with card():
            st.markdown("### Output & business rule")
            st.markdown(
                f"""
                **Predicted ETA** → estimasi menit.  
                **Late-risk** (opsional) → `pred_eta > {LATE_THRESHOLD_DEFAULT:.0f}` menit.
                """
            )

    with b2:
        with card():
            st.markdown("### Quick navigation")
            st.markdown(
                """
                - **Mulai Prediksi** → input & what-if  
                - **Insight Model** → evaluasi, residual, segment, lift, explainability  
                """
            )

# ==========================
# PAGE: TENTANG SISTEM
# ==========================
elif page == "Tentang Sistem":
    st.markdown(f"## ℹ️ Tentang {APP_NAME}")

    with card():
        st.markdown(
            """
            Dashboard ini dibuat untuk membantu pengambilan keputusan operasional,
            khususnya proses last-mile delivery.
            """
        )
        st.markdown("### Tujuan")
        st.markdown(
            """
            - Prediksi **ETA (menit)** untuk setiap order  
            - Sinyal **late-risk** untuk prioritas monitoring  
            - Laporan kualitas model yang mudah dipahami (plot + segment + lift)
            """
        )

    st.markdown("")
    with card():
        st.markdown("### Fitur yang digunakan (raw input)")
        st.markdown(
            """
            - `Distance_km`
            - `Preparation_Time_min`
            - `Courier_Experience_yrs`
            - `Weather`
            - `Traffic_Level`
            - `Time_of_Day`
            - `Vehicle_Type`
            """
        )

# ==========================
# PAGE: TEKNOLOGI & PROSES
# ==========================
elif page == "Teknologi & Proses":
    st.markdown("## ⚙️ Teknologi & Proses")

    with card():
        st.markdown("### Model & pipeline")
        st.markdown(
            """
            - Model: **Ridge Regression**
            - Pipeline inference end-to-end:
              - Missing categorical → **Unknown**
              - OHE: Weather/Time/Vehicle (handle_unknown aman)
              - Ordinal: Traffic (Unknown < Low < Medium < High)
              - Numeric: median + StandardScaler
            """
        )

    st.markdown("")
    cols = st.columns(4)

    with cols[0]:
        with card():
            st.markdown("#### 1️⃣ Input Raw")
            st.markdown("- Distance\n- Prep time\n- Experience\n- Weather/Traffic/Time/Vehicle")

    with cols[1]:
        with card():
            st.markdown("#### 2️⃣ Preprocess")
            st.markdown("- Impute Unknown/median\n- Encode (OHE/Ordinal)\n- Scale numeric")

    with cols[2]:
        with card():
            st.markdown("#### 3️⃣ Model")
            st.markdown("- Ridge Regression\n- Output: ETA (min)")

    with cols[3]:
        with card():
            st.markdown("#### 4️⃣ Action")
            st.markdown("- ETA\n- Late-risk flag\n- Prioritization by rank")

    st.markdown("---")
    with card():
        st.markdown("### Catatan kualitas")
        st.markdown(
            """
            - Residual study membantu cek pola error (heteroscedasticity ringan pada ETA tinggi).
            - Segment error membantu menemukan area sulit (distance jauh, traffic high/unknown).
            """
        )

# ==========================
# PAGE: MULAI PREDIKSI
# ==========================
elif page == "Mulai Prediksi":
    st.markdown("## 🚀 Mulai Prediksi Delivery ETA")

    if pipe is None:
        st.error("Model pipeline belum ditemukan. Pastikan `artifacts/eta_ridge_pipeline.pkl` ada.")
        st.stop()

    with card():
        st.markdown("Isi parameter order untuk mendapatkan **prediksi ETA (menit)** dan sinyal **late-risk**.")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                distance = st.number_input("Distance_km", min_value=0.0, value=5.0, step=0.1)
                prep = st.number_input("Preparation_Time_min", min_value=0.0, value=15.0, step=1.0)
                exp = st.number_input("Courier_Experience_yrs", min_value=0.0, value=2.0, step=0.5)

            with col2:
                weather = st.selectbox("Weather", ["Sunny", "Rainy", "Stormy", "Foggy", "Unknown"], index=0)
                traffic = st.selectbox("Traffic_Level", ["Low", "Medium", "High", "Unknown"], index=1)
                tod = st.selectbox("Time_of_Day", ["Morning", "Afternoon", "Evening", "Night", "Unknown"], index=1)
                vehicle = st.selectbox("Vehicle_Type", ["Bike", "Car", "Scooter", "Van", "Unknown"], index=0)

            threshold = st.number_input("Late threshold (minutes)", min_value=0.0, value=float(LATE_THRESHOLD_DEFAULT), step=1.0)
            submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    if submitted:
        input_data = {
            "Distance_km": distance,
            "Preparation_Time_min": prep,
            "Courier_Experience_yrs": exp,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": tod,
            "Vehicle_Type": vehicle,
        }

        pred = predict_eta(pipe, input_data)
        is_late = pred > threshold

        st.markdown("---")
        st.markdown("### Hasil Prediksi")

        left, right = st.columns([1.6, 1])

        with left:
            with card():
                badge = "badge-green"
                label = "🟢 Low late-risk"
                if is_late:
                    badge = "badge-red"
                    label = "🔴 Late-risk"

                st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)
                st.markdown("")
                kpi_card("Predicted ETA", f"{pred:.1f} min", f"Rule: pred_eta > {threshold:.0f} → late-risk")

                st.markdown("##### Action suggestion")
                if is_late:
                    st.write("Prioritaskan monitoring: cek kesiapan kurir, kondisi traffic, dan buffer waktu.")
                else:
                    st.write("Monitoring standar. Jika kondisi berubah (traffic/weather), lakukan what-if cepat.")

        with right:
            with card():
                st.markdown("##### Visual scale")
                scale_max = 120.0
                st.progress(float(np.clip(pred / scale_max, 0.0, 1.0)))
                st.caption("Skala visual (0–120). Bukan probabilitas.")
                st.markdown("##### Input summary")
                st.dataframe(pd.DataFrame([input_data]), use_container_width=True)

    st.markdown("---")
    st.markdown("### What-if (Quick)")
    st.caption("Ubah traffic & weather untuk lihat perubahan ETA dibanding baseline.")

    base = {
        "Distance_km": 5.0,
        "Preparation_Time_min": 15.0,
        "Courier_Experience_yrs": 2.0,
        "Weather": "Sunny",
        "Traffic_Level": "Medium",
        "Time_of_Day": "Afternoon",
        "Vehicle_Type": "Bike",
    }
    base_pred = float(pipe.predict(pd.DataFrame([base]))[0])

    w1, w2, w3 = st.columns([1, 1, 1.2])
    with w1:
        traffic2 = st.selectbox("Traffic what-if", ["Low", "Medium", "High", "Unknown"], index=1)
    with w2:
        weather2 = st.selectbox("Weather what-if", ["Sunny", "Rainy", "Stormy", "Foggy", "Unknown"], index=0)
    with w3:
        base2 = base.copy()
        base2["Traffic_Level"] = traffic2
        base2["Weather"] = weather2
        what_pred = float(pipe.predict(pd.DataFrame([base2]))[0])
        kpi_card("Baseline ETA", f"{base_pred:.1f} min", "Default baseline")
        kpi_card("What-if ETA", f"{what_pred:.1f} min", f"Δ {(what_pred - base_pred):+.1f} min")

# ==========================
# PAGE: INSIGHT MODEL
# ==========================
elif page == "Insight Model":
    st.markdown("## 📊 Insight Model")

    if pipe is None:
        st.error("Model pipeline belum ditemukan. Pastikan `artifacts/eta_ridge_pipeline.pkl` ada.")
        st.stop()

    if eval_df is None:
        st.warning("File evaluasi belum ada. Pastikan `artifacts/eval_test.csv` tersedia.")
        st.stop()

    ok, missing = ensure_columns(eval_df, RAW_FEATURES + [TARGET])
    if not ok:
        st.error(f"Kolom kurang di eval_test.csv: {missing}")
        st.stop()

    df_res = make_eval_table(pipe, eval_df)
    mae, rmse, r2 = compute_metrics(df_res)

    st.markdown("---")
    tabs = st.tabs(["Overview", "Residual Study", "Segment Error", "Lift/Gain", "Explainability", "Download"])

    # ==========================
    # TAB 1: OVERVIEW (1 + 4)
    # ==========================
    with tabs[0]:
        st.markdown("### Model Performance (Eval Set)")

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("MAE (computed)", f"{mae:.2f}", "From eval_test.csv")
        with c2:
            kpi_card("RMSE (computed)", f"{rmse:.2f}", "From eval_test.csv")
        with c3:
            kpi_card("R² (computed)", f"{r2:.3f}", "From eval_test.csv")

        st.caption(
            f"Notebook reference: MAE {MAE_TEST_REF:.2f} | RMSE {RMSE_TEST_REF:.2f} | R² {R2_TEST_REF:.3f}. "
            "Computed bisa berbeda jika eval_test.csv berbeda versi/split."
        )

        with card():
            st.markdown("#### Key Takeaways")
            st.markdown(takeaways_overview(mae, rmse, r2))

        # Visual: actual vs pred + residual hist
        a, b = st.columns(2)
        with a:
            with card():
                st.pyplot(plot_actual_vs_pred(df_res), clear_figure=True)
                with st.expander("Penjelasan singkat: Actual vs Predicted"):
                    st.markdown(
                        """
                        - Titik dekat garis diagonal = prediksi akurat.  
                        - Pada ETA tinggi, titik lebih menyebar → kasus ekstrem lebih sulit diprediksi.
                        """
                    )
        with b:
            with card():
                st.pyplot(plot_residual_hist(df_res), clear_figure=True)
                with st.expander("Penjelasan singkat: Residual Distribution"):
                    st.markdown(
                        """
                        - Residual = **Actual − Predicted**.  
                        - Mayoritas dekat 0 → rata-rata prediksi cukup baik.  
                        - Ekor/outlier → ada sebagian order yang under/over-predicted cukup besar.
                        """
                    )

    # ==========================
    # TAB 2: RESIDUAL STUDY (1 + 2 + 4)
    # ==========================
    with tabs[1]:
        st.markdown("### Residual Study (Diagnostics)")

        with card():
            st.pyplot(plot_residual_vs_pred(df_res), clear_figure=True)
            with st.expander("Penjelasan singkat: Residual vs Predicted"):
                st.markdown(
                    """
                    - Dipakai untuk cek **heteroscedasticity**.  
                    - Jika residual makin menyebar saat predicted makin besar → varians error meningkat di ETA tinggi.
                    """
                )

        with card():
            st.markdown("#### Assumption Check (Regresi) — dikaitkan dengan plot residual")
            st.markdown(takeaways_residual())

        with card():
            st.markdown("#### Insight Singkat")
            st.markdown(
                """
                - Model **cukup linear** untuk mayoritas data.  
                - **Ketidakpastian naik** pada ETA tinggi → rekomendasi operasional:
                  prioritaskan monitoring untuk prediksi ETA tinggi (karena error lebih besar + risk lebih tinggi).
                """
            )

    # ==========================
    # TAB 3: SEGMENT ERROR (1 + 3 + 4)
    # ==========================
    with tabs[2]:
        st.markdown("### Segment Error (MAE)")
        st.caption("Tujuan: menemukan area model paling lemah → bahan improvement atau aturan operasional.")

        by_dist, by_traffic, by_decile = segment_mae_tables(df_res)

        with card():
            st.markdown("#### Insight Singkat (Segment)")
            st.markdown(takeaways_segment(by_dist, by_traffic, by_decile))

        a, b = st.columns(2)

        with a:
            with card():
                st.markdown("#### MAE by Distance (Near/Mid/Far)")
                st.pyplot(plot_bar_mae(by_dist, "distance_bin", "MAE by Distance Bin"), clear_figure=True)
                st.dataframe(by_dist, use_container_width=True)

        with b:
            with card():
                st.markdown("#### MAE by Traffic Level")
                st.pyplot(plot_bar_mae(by_traffic, "Traffic_Level", "MAE by Traffic Level"), clear_figure=True)
                st.dataframe(by_traffic, use_container_width=True)

        with card():
            st.markdown("#### MAE by Predicted Decile")
            st.pyplot(plot_bar_mae(by_decile, "pred_decile", "MAE by Predicted Decile"), clear_figure=True)
            st.dataframe(by_decile, use_container_width=True)
            st.caption("Biasanya decile tinggi punya MAE lebih besar → kasus ekstrem lebih sulit.")

    # ==========================
    # TAB 4: LIFT/GAIN (1 + 4)
    # ==========================
    with tabs[3]:
        st.markdown("### Late-risk Prioritization (Lift/Gain)")

        threshold = st.number_input(
            "Late threshold (minutes)",
            min_value=0.0,
            value=float(LATE_THRESHOLD_DEFAULT),
            step=1.0,
            key="lift_threshold",
        )

        base_rate = (df_res[TARGET] > threshold).mean()
        kpi_card("Base late rate", f"{base_rate*100:.1f}%", "Actual > threshold")

        tbl = lift_table(df_res, threshold)

        with card():
            st.markdown("#### Insight Singkat (Lift/Gain)")
            st.markdown(takeaways_lift(base_rate))

        c1, c2 = st.columns([1.3, 1])
        with c1:
            with card():
                st.markdown("#### Lift table (by ranked predicted ETA)")
                st.dataframe(tbl, use_container_width=True)

        with c2:
            with card():
                st.markdown("#### Lift curve")
                st.pyplot(plot_lift(tbl), clear_figure=True)
                with st.expander("Penjelasan singkat: Lift Curve"):
                    st.markdown(
                        """
                        - Bucket **Top 10%** = order dengan predicted ETA tertinggi.  
                        - Lift tinggi di bucket awal → ranking by predicted ETA efektif untuk prioritas monitoring.
                        """
                    )

    # ==========================
    # TAB 5: EXPLAINABILITY (1 + 4)
    # ==========================
    with tabs[4]:
        st.markdown("### Explainability (Ridge Coefficients — Global)")

        coef_df = try_get_coeff_table(pipe)
        if coef_df is None:
            st.info("Koefisien belum bisa diekstrak dari artifact pipeline ini.")
        else:
            top_n = st.slider("Top N coefficients", min_value=5, max_value=50, value=20, step=5)

            a, b = st.columns([1.2, 1.8])
            with a:
                with card():
                    st.markdown("#### Top coefficients (table)")
                    st.dataframe(coef_df.head(top_n), use_container_width=True)
                    with st.expander("Cara baca koefisien (singkat)"):
                        st.markdown(
                            """
                            - Koefisien **positif** → menaikkan ETA  
                            - Koefisien **negatif** → menurunkan ETA  
                            - Besar-kecilnya koefisien menunjukkan pengaruh relatif setelah preprocessing/encoding
                            """
                        )

            with b:
                with card():
                    st.markdown("#### Top coefficients (chart)")
                    st.pyplot(plot_top_coefficients(coef_df, top_n=top_n), clear_figure=True)

            with card():
                st.markdown("#### Insight Singkat (Explainability)")
                st.markdown(
                    """
                    Driver umum ETA (sesuai hasil modeling kamu):
                    - **Distance_km** dan **Preparation_Time_min** biasanya menaikkan ETA.
                    - **Traffic_Level** tinggi cenderung menaikkan ETA.
                    - **Courier_Experience_yrs** cenderung menurunkan ETA (lebih cepat/efisien).
                    """
                )

    # ==========================
    # TAB 6: DOWNLOAD
    # ==========================
    with tabs[5]:
        st.markdown("### Download")
        with card():
            st.download_button(
                "Download eval_with_predictions.csv",
                data=df_res.to_csv(index=False).encode("utf-8"),
                file_name="eval_with_predictions.csv",
                mime="text/csv",
            )
            st.caption("CSV berisi: y_pred, residual, abs_error untuk analisis lanjut.")
