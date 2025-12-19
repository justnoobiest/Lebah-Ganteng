import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Pastel
px.defaults.color_continuous_scale = px.colors.sequential.YlGn

st.set_page_config(page_title="COVID-19 Global Dashboard", page_icon="🦠", layout="wide")

PASTEL_CSS = """
<style>
:root{
  --bg1:#eaf7ee;
  --bg2:#f6fff8;
  --sidebar:#dff3e6;
  --card:#ffffffcc;
  --border:rgba(31,45,42,0.14);
  --text:#1f2d2a;
  --muted:rgba(31,45,42,0.68);
  --accent:#2e7d5b;
  --accent2:#6fbf9a;
}

html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; color: var(--text); }

[data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 65%, #ffffff 100%);
}

[data-testid="stHeader"]{
  background: rgba(255,255,255,0.0);
}

[data-testid="stSidebar"]{
  background: var(--sidebar);
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] *{
  color: var(--text);
}

div[data-testid="metric-container"]{
  background: var(--card);
  border: 1px solid var(--border);
  padding: 14px 14px;
  border-radius: 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"]{
  color: var(--muted) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color: var(--text) !important;
  font-weight: 800;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"]{
  color: var(--accent) !important;
  font-weight: 700;
}

[data-testid="stExpander"]{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 10px 24px rgba(0,0,0,0.05);
}
[data-testid="stExpander"] summary{
  background: rgba(255,255,255,0.55);
}

.stButton>button, .stDownloadButton>button{
  background: linear-gradient(180deg, rgba(111,191,154,0.65), rgba(46,125,91,0.65));
  border: 1px solid rgba(46,125,91,0.25);
  color: #0e2b20;
  border-radius: 14px;
  font-weight: 700;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  filter: brightness(1.03);
}

hr{
  border: 0;
  height: 1px;
  background: linear-gradient(to right, transparent, var(--border), transparent);
  margin: 1.1rem 0;
}

.small-note{
  color: var(--muted);
  font-size: 0.92rem;
}
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)


def pastelize(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1f2d2a"),
        legend=dict(bgcolor="rgba(255,255,255,0.55)"),
    )
    return fig


@st.cache_data
def load_data():
    day_wise = pd.read_csv("day_wise.csv")
    full_grouped = pd.read_csv("full_grouped.csv")
    country_latest = pd.read_csv("country_wise_latest.csv")
    worldometer = pd.read_csv("worldometer_data.csv")
    usa_county = pd.read_csv("usa_county_wise.csv")
    clean = pd.read_csv("covid_19_clean_complete.csv")

    for df in [day_wise, full_grouped, usa_county, clean]:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    skip_like = {
        "country/region",
        "province/state",
        "admin2",
        "province_state",
        "country_region",
        "combined_key",
        "continent",
        "who region",
        "who_region",
    }

    for df in [day_wise, full_grouped, country_latest, worldometer, usa_county, clean]:
        for col in df.select_dtypes(include="object").columns:
            if col and col.strip().lower() in skip_like:
                continue
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return day_wise, full_grouped, country_latest, worldometer, usa_county, clean


day_wise, full_grouped, country_latest, worldometer, usa_county, clean = load_data()


def format_number(x):
    try:
        if pd.isna(x):
            return "-"
        return f"{int(float(x)):,}"
    except Exception:
        return "-"


def format_pct(x, digits=1):
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return "-"


def format_float(x, digits=2):
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def trend_phrase(frac):
    if frac is None or pd.isna(frac):
        return "perubahannya tidak dapat dihitung (data pembanding tidak tersedia)."
    if frac > 0:
        return f"terjadi peningkatan sekitar {format_pct(frac)} dibanding periode sebelumnya."
    if frac < 0:
        return f"terjadi penurunan sekitar {format_pct(abs(frac))} dibanding periode sebelumnya."
    return "perubahannya relatif stabil dibanding periode sebelumnya."


def normalize_country(s):
    if pd.isna(s):
        return ""
    return (
        str(s)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(".", "")
        .replace(",", "")
    )


def nearest_row_by_date(df, date_ts):
    if df.empty or "Date" not in df.columns:
        return None
    d = df.dropna(subset=["Date"]).copy()
    if d.empty:
        return None
    idx = (d["Date"] - pd.to_datetime(date_ts)).abs().idxmin()
    return d.loc[idx]


def build_story_events(df):
    if df.empty or "Date" not in df.columns:
        return []
    d = df.dropna(subset=["Date"]).sort_values("Date").copy()
    if d.empty:
        return []

    def first_above(col, frac=0.01):
        if col not in d.columns:
            return None
        s = pd.to_numeric(d[col], errors="coerce").fillna(0)
        mx = float(s.max())
        if mx <= 0:
            return None
        thr = mx * float(frac)
        hit = d[s >= thr]
        if hit.empty:
            return None
        return pd.to_datetime(hit.iloc[0]["Date"])

    def peak_day(col, rolling=7):
        if col not in d.columns:
            return None
        s = pd.to_numeric(d[col], errors="coerce").fillna(0)
        if rolling and rolling > 1:
            s2 = s.rolling(rolling, min_periods=max(1, rolling // 2)).mean()
        else:
            s2 = s
        idx = s2.idxmax()
        return pd.to_datetime(d.loc[idx, "Date"])

    events = []
    start_date = pd.to_datetime(d["Date"].min())
    end_date = pd.to_datetime(d["Date"].max())

    events.append({"date": start_date, "title": "Awal terdeteksinya Covid-19", "desc": "Mulai dimulainya observasi."})

    f_cases = first_above("New cases", frac=0.01)
    if f_cases is not None:
        events.append({"date": f_cases, "title": "Peningkatan kasus Covid-19", "desc": "Hari pertama kasus mencapai ambang signifikan."})

    f_deaths = first_above("New deaths", frac=0.01)
    if f_deaths is not None:
        events.append({"date": f_deaths, "title": "Peningkatan kasus kematian akibat Covid-19", "desc": "Hari pertama kasus kematian akibat Covid-19 mencapai ambang signifikan."})

    p_cases = peak_day("New cases", rolling=7) or peak_day("New cases", rolling=0)
    if p_cases is not None:
        events.append({"date": p_cases, "title": "Puncak kasus baru (rata-rata dalam 7-days)", "desc": "Puncak kasus baru dalam rata-rata 7 hari."})

    p_deaths = peak_day("New deaths", rolling=7) or peak_day("New deaths", rolling=0)
    if p_deaths is not None:
        events.append({"date": p_deaths, "title": "Puncak kematian baru (rata-rata dalam 7-days)", "desc": "Puncak kematian baru dalam rata-rata 7 hari."})

    p_active = peak_day("Active", rolling=0)
    if p_active is not None:
        events.append({"date": p_active, "title": "Puncak kasus aktif", "desc": "Hari dengan kasus aktif tertinggi."})

    events.append({"date": end_date, "title": "Akhir periode data", "desc": "Snapshot paling akhir pada dataset."})

    seen = set()
    uniq = []
    for e in sorted(events, key=lambda x: x["date"]):
        key = (pd.to_datetime(e["date"]).date(), e["title"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


@st.cache_data
def get_population_table():
    w = worldometer.copy()
    if w.empty:
        return None, None, pd.DataFrame(columns=["_key", "_pop"])

    w_country_col = None
    for cand in ["Country/Region", "Country", "country", "Country/Region "]:
        if cand in w.columns:
            w_country_col = cand
            break

    pop_col = None
    for cand in ["Population", "population", "Pop", "pop"]:
        if cand in w.columns:
            pop_col = cand
            break

    if w_country_col is None or pop_col is None:
        return w_country_col, pop_col, pd.DataFrame(columns=["_key", "_pop"])

    w["_key"] = w[w_country_col].apply(normalize_country)
    w["_pop"] = pd.to_numeric(w[pop_col], errors="coerce")
    pop_df = w[["_key", "_pop"]].dropna(subset=["_key"]).drop_duplicates("_key")
    return w_country_col, pop_col, pop_df


@st.cache_data
def build_country_metrics_snapshot():
    if country_latest.empty or "Country/Region" not in country_latest.columns:
        return pd.DataFrame()

    df = country_latest.copy()
    df["_key"] = df["Country/Region"].apply(normalize_country)

    for c in ["Confirmed", "Deaths", "Recovered", "Active"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    _, _, pop_df = get_population_table()
    if not pop_df.empty:
        df = df.merge(pop_df, on="_key", how="left")

    if "Confirmed" in df.columns:
        df["CFR"] = df["Deaths"] / df["Confirmed"]
        df["Recovery rate"] = df["Recovered"] / df["Confirmed"]

    if "_pop" in df.columns:
        for c in ["Confirmed", "Deaths", "Recovered", "Active"]:
            if c in df.columns:
                df[f"{c} per 1M"] = (df[c] / df["_pop"]) * 1_000_000

    return df


@st.cache_data
def build_country_14d_hotspots():
    if full_grouped.empty or "Date" not in full_grouped.columns or "Country/Region" not in full_grouped.columns:
        return pd.DataFrame()

    d = full_grouped.dropna(subset=["Date"]).copy()
    if d.empty:
        return pd.DataFrame()

    if "New cases" not in d.columns:
        return pd.DataFrame()

    d["New cases"] = pd.to_numeric(d["New cases"], errors="coerce").fillna(0)
    end_date = pd.to_datetime(d["Date"].max()).normalize()
    start_28 = end_date - pd.Timedelta(days=27)
    mid_14 = end_date - pd.Timedelta(days=14)

    d28 = d[(d["Date"] >= start_28) & (d["Date"] <= end_date)].copy()
    a = d28[d28["Date"] <= mid_14].groupby("Country/Region")["New cases"].sum().rename("Prev 14D")
    b = d28[d28["Date"] > mid_14].groupby("Country/Region")["New cases"].sum().rename("Last 14D")

    out = pd.concat([a, b], axis=1).fillna(0).reset_index()
    out["Abs change"] = out["Last 14D"] - out["Prev 14D"]
    out["Pct change"] = np.where(out["Prev 14D"] > 0, out["Abs change"] / out["Prev 14D"], np.nan)
    out["As of"] = end_date.date()
    return out


def detect_waves(series, dates, min_prom_frac=0.25, min_gap_days=21):
    s = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
    if len(s) < 10:
        return []
    mx = float(np.max(s))
    if mx <= 0:
        return []

    peaks = []
    for i in range(2, len(s) - 2):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > 0:
            peaks.append(i)

    peaks = sorted(peaks, key=lambda i: s[i], reverse=True)
    selected = []
    for i in peaks:
        if s[i] < mx * float(min_prom_frac):
            continue
        ok = True
        for j in selected:
            if abs(i - j) < int(min_gap_days):
                ok = False
                break
        if ok:
            selected.append(i)
        if len(selected) >= 6:
            break

    selected = sorted(selected)
    waves = []
    for i in selected:
        peak_date = pd.to_datetime(dates[i]).date()
        peak_val = float(s[i])

        left = i
        while left > 0 and s[left] > peak_val * 0.20:
            left -= 1
        right = i
        while right < len(s) - 1 and s[right] > peak_val * 0.20:
            right += 1

        waves.append(
            {
                "Start": pd.to_datetime(dates[left]).date(),
                "Peak": peak_date,
                "End": pd.to_datetime(dates[right]).date(),
                "Peak value": peak_val,
                "Duration (days)": int((pd.to_datetime(dates[right]) - pd.to_datetime(dates[left])).days),
            }
        )
    return waves


def lag_correlation(cases, deaths, max_lag=35):
    a = pd.to_numeric(cases, errors="coerce").fillna(0).to_numpy()
    b = pd.to_numeric(deaths, errors="coerce").fillna(0).to_numpy()
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    lags = list(range(0, int(max_lag) + 1))
    vals = []
    for lag in lags:
        if lag == 0:
            x = a
            y = b
        else:
            x = a[:-lag]
            y = b[lag:]
        if len(x) < 10:
            vals.append(np.nan)
            continue
        if np.std(x) == 0 or np.std(y) == 0:
            vals.append(np.nan)
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return pd.DataFrame({"Lag (days)": lags, "Correlation": vals})


def simple_kmeans(X, k=4, iters=30, seed=42):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int), np.empty((0, X.shape[1]))
    k = int(max(1, min(k, n)))
    centers = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(int(iters)):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                centers[j] = X[rng.integers(0, n)]
            else:
                centers[j] = X[idx].mean(axis=0)
    return labels, centers


if "Date" in day_wise.columns and day_wise["Date"].notna().any():
    _all_dates = pd.to_datetime(day_wise["Date"].dropna().unique())
    min_date = pd.to_datetime(_all_dates.min()).date()
    max_date = pd.to_datetime(_all_dates.max()).date()
else:
    min_date = pd.to_datetime("2020-01-01").date()
    max_date = pd.to_datetime("2020-12-31").date()

if "Country/Region" in full_grouped.columns:
    all_countries = sorted([c for c in full_grouped["Country/Region"].dropna().unique()])
else:
    all_countries = []

default_countries = [c for c in ["Indonesia", "US", "Italy", "China", "India"] if c in all_countries]
if not default_countries and all_countries:
    default_countries = all_countries[:3]

@st.cache_data
def load_idn_data():
    df = pd.read_csv("covid_19_indonesia_time_series_all.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    return df

def indonesia_province_view():
    st.title("🇮🇩 Indonesia Province View")
    st.caption("Pilih provinsi untuk melihat tren dan perbandingan antar provinsi.")

    df = load_idn_data()
    prov_df = df[df["Location"] != "Indonesia"].copy()

    provinces = sorted(prov_df["Location"].dropna().unique())
    default_idx = provinces.index("DKI Jakarta") if "DKI Jakarta" in provinces else 0
    prov = st.selectbox("Pilih Provinsi:", provinces, index=default_idx)

    dfp = prov_df[prov_df["Location"] == prov].sort_values("Date")
    last = dfp.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Confirmed", f"{int(last['Total Cases']):,}")
    c2.metric("Total Deaths", f"{int(last['Total Deaths']):,}")
    c3.metric("Total Recovered", f"{int(last['Total Recovered']):,}")
    c4.metric("Total Active", f"{int(last['Total Active Cases']):,}")

    st.subheader(f"Perkembangan kasus di {prov}")

    metric = st.selectbox(
        "Metrik:",
        ["New Cases", "New Deaths", "New Recovered", "New Active Cases",
         "Total Cases", "Total Deaths", "Total Recovered", "Total Active Cases"],
        index=0
    )
    view = st.selectbox("Tampilan:", ["Harian", "Rata-rata 7 hari"], index=1)

    plot_df = dfp[["Date", metric]].copy()
    if view == "Rata-rata 7 hari":
        plot_df[metric] = plot_df[metric].rolling(7).mean()

    fig_line = px.line(plot_df, x="Date", y=metric)
    try:
        st.plotly_chart(fig_line, width="stretch")
    except TypeError:
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("⬆️ 10 provinsi dengan nilai tertinggi (snapshot tanggal)")
    min_d, max_d = prov_df["Date"].min(), prov_df["Date"].max()
    snap = st.slider("Pilih tanggal:", min_value=min_d.date(), max_value=max_d.date(), value=max_d.date())
    snap_df = prov_df[prov_df["Date"].dt.date == snap].copy()

    rank_metric = st.selectbox("Ranking berdasarkan:", ["Total Cases", "New Cases", "Total Deaths", "New Deaths"], index=0)
    top10 = snap_df.nlargest(10, rank_metric)[["Location", rank_metric]].sort_values(rank_metric, ascending=False)

    fig_bar = px.bar(top10, x="Location", y=rank_metric)
    try:
        st.plotly_chart(fig_bar, width="stretch")
    except TypeError:
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📁 Data provinsi (mentah)"):
        st.dataframe(dfp)

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Pilih halaman:",
    (
        "🏠 Overview",
        "📖 Global Timeline Story Mode",
        "🌍 Global Map",
        "📘 Analisis Data Perkembangan COVID-19",
        "📊 Country Dashboard",
        "📈 Country Comparison",
        "🗽 USA View",
    	"🇮🇩 Indonesia Province View",
        "🔥 Insights & Hotspots",
        "⏱️ Timelapse",
        "📑 Data Explorer",
        "ℹ️ About",
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: COVID-19 Corona Virus Report  \n"
    "Sumber: [Kaggle – imdevskp](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Kelompok Lebah Ganteng**  \n"
    "Muhammad Dimas Sudirman – 021002404001  \n"
    "Ari Wahyu Patriangga – 021002404007  \n"
    "Lola Aritasari – 021002404004"
)


if page == "🏠 Overview":
    st.markdown("<h1 style='text-align:center'>🦠 COVID-19 Global Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>Ringkasan perkembangan kasus COVID-19 secara global berdasarkan data time-series.</div>", unsafe_allow_html=True)

    if "Date" in day_wise.columns and day_wise["Date"].notna().any():
        latest_row = day_wise.dropna(subset=["Date"]).sort_values("Date").iloc[-1]
    else:
        st.warning("Kolom Date tidak ditemukan/valid pada day_wise.csv")
        latest_row = pd.Series(dtype="object")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Confirmed", format_number(latest_row.get("Confirmed")), delta=format_number(latest_row.get("New cases")))
    with c2:
        st.metric("Total Deaths", format_number(latest_row.get("Deaths")), delta=format_number(latest_row.get("New deaths")))
    with c3:
        st.metric("Total Recovered", format_number(latest_row.get("Recovered")), delta=format_number(latest_row.get("New recovered")))
    with c4:
        st.metric("Active Cases", format_number(latest_row.get("Active")))

    st.markdown("### 🌐 Komposisi kasus global & negara dengan kasus terbanyak")
    col1, col2 = st.columns(2)

    with col1:
        pie_data = pd.DataFrame(
            {
                "Status": ["Active", "Recovered", "Deaths"],
                "Count": [
                    max(float(latest_row.get("Active", 0) or 0), 0),
                    max(float(latest_row.get("Recovered", 0) or 0), 0),
                    max(float(latest_row.get("Deaths", 0) or 0), 0),
                ],
            }
        )
        fig_pie = px.pie(pie_data, names="Status", values="Count", hole=0.45, title="Komposisi kasus global (snapshot terakhir)")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(pastelize(fig_pie), use_container_width=True)

    with col2:
        if "Country/Region" in country_latest.columns and "Confirmed" in country_latest.columns:
            tmp = country_latest.copy()
            tmp["Confirmed"] = pd.to_numeric(tmp["Confirmed"], errors="coerce").fillna(0)
            top10 = tmp[tmp["Confirmed"] > 0].sort_values("Confirmed", ascending=False).head(10)
            fig_top = px.bar(top10, x="Country/Region", y="Confirmed", title="🔝 10 negara dengan kasus terkonfirmasi tertinggi")
            fig_top.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(pastelize(fig_top), use_container_width=True)
        else:
            st.info("Kolom Country/Region atau Confirmed tidak tersedia pada country_wise_latest.csv")

    st.markdown("### 📈 Tren global dari waktu ke waktu")
    metric_candidates = [m for m in ["Confirmed", "Deaths", "Recovered", "Active"] if m in day_wise.columns]
    metrics_to_plot = st.multiselect(
        "Pilih metrik yang ditampilkan:",
        options=metric_candidates,
        default=[m for m in ["Confirmed", "Deaths", "Recovered"] if m in metric_candidates],
    )

    if metrics_to_plot and "Date" in day_wise.columns:
        long_df = day_wise[["Date"] + metrics_to_plot].dropna(subset=["Date"]).melt(
            id_vars="Date", value_vars=metrics_to_plot, var_name="Metric", value_name="Value"
        )
        fig = px.line(long_df, x="Date", y="Value", color="Metric", title="Perkembangan kasus global")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(pastelize(fig), use_container_width=True)

    if "New cases" in day_wise.columns and "Date" in day_wise.columns:
        st.markdown("### 📊 Kasus baru per hari (global)")
        fig_new = px.bar(day_wise.dropna(subset=["Date"]), x="Date", y="New cases", title="New confirmed cases per day")
        st.plotly_chart(pastelize(fig_new), use_container_width=True)

    st.markdown("### 🔍 Korelasi antar indikator global")
    corr_cols = [c for c in ["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered"] if c in day_wise.columns]
    if len(corr_cols) >= 2:
        corr_mat = day_wise[corr_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr_mat, text_auto=".2f", title="Matriks korelasi indikator global (harian)")
        st.plotly_chart(pastelize(fig_corr), use_container_width=True)


elif page == "📖 Global Timeline Story Mode":
    st.header("📖 Global Timeline Story Mode")
    st.markdown("<div class='small-note'>Mode naratif untuk menelusuri momen penting pada time-series global berdasarkan indikator yang dipilih.</div>", unsafe_allow_html=True)

    if day_wise.empty or "Date" not in day_wise.columns or not day_wise["Date"].notna().any():
        st.warning("day_wise.csv tidak valid / kolom Date tidak ditemukan.")
    else:
        story_df = day_wise.dropna(subset=["Date"]).sort_values("Date").copy()
        story_metrics = [m for m in ["New cases", "New deaths", "New recovered", "Confirmed", "Deaths", "Recovered", "Active"] if m in story_df.columns]
        if not story_metrics:
            st.warning("Tidak ada kolom metrik yang cocok pada day_wise.csv")
        else:
            c1, c2, c3 = st.columns([1.3, 1, 1])
            with c1:
                story_metric = st.selectbox("Metrik:", options=story_metrics, index=0)
            with c2:
                smooth = st.selectbox("Tampilan:", options=["Harian", "Rata-rata 7 hari"], index=1)
            with c3:
                date_range = st.slider("Rentang tanggal:", min_value=min_date, max_value=max_date, value=(min_date, max_date))

            story_df = story_df[story_df["Date"].dt.date.between(date_range[0], date_range[1])].copy()
            story_df[story_metric] = pd.to_numeric(story_df[story_metric], errors="coerce").fillna(0)

            events = build_story_events(story_df)
            if not events:
                st.warning("Tidak bisa membangun story events dari data yang tersedia.")
            else:
                if "story_idx" not in st.session_state:
                    st.session_state.story_idx = 0

                labels = [f"{i+1}. {e['title']} ({pd.to_datetime(e['date']).date()})" for i, e in enumerate(events)]
                selected = st.selectbox("Momen:", options=labels, index=min(st.session_state.story_idx, len(labels) - 1), key="story_event_select")
                new_idx = labels.index(selected)
                if new_idx != st.session_state.story_idx:
                    st.session_state.story_idx = new_idx

                b1, b2, b3 = st.columns([1, 1, 1])
                with b1:
                    if st.button("⬅️ Prev", use_container_width=True):
                        st.session_state.story_idx = max(0, st.session_state.story_idx - 1)
                with b2:
                    if st.button("⟲ Reset", use_container_width=True):
                        st.session_state.story_idx = 0
                with b3:
                    if st.button("Next ➡️", use_container_width=True):
                        st.session_state.story_idx = min(len(events) - 1, st.session_state.story_idx + 1)

                idx = st.session_state.story_idx
                ev = events[idx]
                ev_date = pd.to_datetime(ev["date"])

                plot_df = story_df[["Date", story_metric]].copy().sort_values("Date")
                if smooth == "Rata-rata 7 hari":
                    plot_df["Value"] = plot_df[story_metric].rolling(7, min_periods=3).mean()
                else:
                    plot_df["Value"] = plot_df[story_metric]

                fig = px.line(plot_df, x="Date", y="Value", title=f"{story_metric} ({smooth})")
                fig.update_layout(hovermode="x unified")
                fig.add_vline(x=ev_date, line_width=2)
                fig.add_annotation(
                    x=ev_date,
                    y=float(plot_df["Value"].max()) if plot_df["Value"].notna().any() else 0,
                    text=ev["title"],
                    showarrow=True,
                    arrowhead=2,
                )
                st.plotly_chart(pastelize(fig), use_container_width=True)

                r = nearest_row_by_date(story_df, ev_date)
                if r is not None:
                    st.subheader(f"{ev['title']} — {pd.to_datetime(r['Date']).date()}")
                    st.markdown(f"<div class='small-note'>{ev['desc']}</div>", unsafe_allow_html=True)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Confirmed", format_number(r.get("Confirmed")))
                    with m2:
                        st.metric("Deaths", format_number(r.get("Deaths")))
                    with m3:
                        st.metric("Recovered", format_number(r.get("Recovered")))
                    with m4:
                        st.metric("Active", format_number(r.get("Active")))

                    n1, n2, n3 = st.columns(3)
                    with n1:
                        st.metric("New cases", format_number(r.get("New cases")))
                    with n2:
                        st.metric("New deaths", format_number(r.get("New deaths")))
                    with n3:
                        st.metric("New recovered", format_number(r.get("New recovered")))


elif page == "🌍 Global Map":
    st.header("🌍 Global Map")
    st.markdown("<div class='small-note'>Peta per negara. Warna menunjukkan besaran indikator pada data terbaru.</div>", unsafe_allow_html=True)

    if country_latest.empty:
        st.warning("country_wise_latest.csv kosong.")
    else:
        map_metric_options = [
            "Confirmed",
            "Deaths",
            "Recovered",
            "Active",
            "Deaths / 100 Cases",
            "Recovered / 100 Cases",
            "Deaths / 100 Recovered",
        ]
        map_metric_options = [m for m in map_metric_options if m in country_latest.columns]
        if not map_metric_options:
            st.warning("Tidak ada kolom metrik yang cocok di country_wise_latest.csv")
        else:
            metric = st.selectbox("Pilih indikator:", map_metric_options, index=0)
            map_df = country_latest.copy()
            if "Confirmed" in map_df.columns:
                map_df["Confirmed"] = pd.to_numeric(map_df["Confirmed"], errors="coerce").fillna(0)
                map_df = map_df[map_df["Confirmed"] > 0]

            fig = px.choropleth(
                map_df,
                locations="Country/Region",
                locationmode="country names",
                color=metric,
                hover_name="Country/Region",
                title=f"{metric} – latest snapshot",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(pastelize(fig), use_container_width=True)

            with st.expander("📋 Tabel ringkas"):
                show_cols = [c for c in ["Country/Region", "Confirmed", "Deaths", "Recovered", "Active"] if c in map_df.columns]
                if show_cols:
                    sort_col = "Confirmed" if "Confirmed" in show_cols else show_cols[1]
                    st.dataframe(map_df[show_cols].sort_values(sort_col, ascending=False).reset_index(drop=True))
                else:
                    st.dataframe(map_df.head(50))


elif page == "📘 Analisis Data Perkembangan COVID-19":
    st.title("📘 Analisis Data Perkembangan COVID-19")
    st.markdown("<div class='small-note'>Analisis kasus Covid-19 berbasis tren, gelombang, jeda kasus kematian.</div>", unsafe_allow_html=True)

    if day_wise.empty or "Date" not in day_wise.columns or not day_wise["Date"].notna().any():
        st.warning("day_wise.csv tidak valid / kolom Date tidak ditemukan.")
    else:
        g = day_wise.dropna(subset=["Date"]).sort_values("Date").copy()
        for c in ["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered"]:
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)

        if "New cases" in g.columns:
            g["New cases 7D"] = g["New cases"].rolling(7, min_periods=3).mean()
        if "New deaths" in g.columns:
            g["New deaths 7D"] = g["New deaths"].rolling(7, min_periods=3).mean()

        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            analysis_range = st.slider("Rentang analisis:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        with c2:
            cfr_floor = st.number_input("Minimum confirmed (filter CFR):", min_value=0, value=10000, step=1000)
        with c3:
            show_tables = st.checkbox("Tampilkan tabel detail", value=False)

        g2 = g[g["Date"].dt.date.between(analysis_range[0], analysis_range[1])].copy()
        if g2.empty:
            st.warning("Rentang tanggal tidak menghasilkan data.")
        else:
            latest = g2.iloc[-1]
            prev_7 = g2.tail(14).head(7) if len(g2) >= 14 else g2.head(0)
            last_7 = g2.tail(7)

            last_cases_7 = float(last_7["New cases"].sum()) if "New cases" in g2.columns else np.nan
            prev_cases_7 = float(prev_7["New cases"].sum()) if ("New cases" in g2.columns and not prev_7.empty) else np.nan
            wow_cases = (last_cases_7 - prev_cases_7) / prev_cases_7 if prev_cases_7 and prev_cases_7 > 0 else np.nan

            last_deaths_7 = float(last_7["New deaths"].sum()) if "New deaths" in g2.columns else np.nan
            prev_deaths_7 = float(prev_7["New deaths"].sum()) if ("New deaths" in g2.columns and not prev_7.empty) else np.nan
            wow_deaths = (last_deaths_7 - prev_deaths_7) / prev_deaths_7 if prev_deaths_7 and prev_deaths_7 > 0 else np.nan

            with st.expander("1) Ringkasan Global", expanded=True):
                st.markdown(
                    f"<div class='small-note'>Bagian ini merangkum kondisi pada akhir rentang analisis, lalu membandingkan total kasus/kematian baru pada 7 hari terakhir dengan 7 hari sebelumnya. Dari data terlihat {trend_phrase(wow_cases)} Untuk kematian baru, {trend_phrase(wow_deaths)}</div>",
                    unsafe_allow_html=True,
                )

                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.metric("Confirmed (akhir rentang)", format_number(latest.get("Confirmed")))
                with a2:
                    st.metric("Deaths (akhir rentang)", format_number(latest.get("Deaths")))
                with a3:
                    st.metric("Recovered (akhir rentang)", format_number(latest.get("Recovered")))
                with a4:
                    st.metric("Active (akhir rentang)", format_number(latest.get("Active")))

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total New cases (7 hari)", format_number(last_cases_7), delta=format_pct(wow_cases))
                with b2:
                    st.metric("Rata-rata New cases (7D)", format_number(last_7["New cases 7D"].iloc[-1] if "New cases 7D" in last_7.columns else np.nan))
                with b3:
                    st.metric("Total New deaths (7 hari)", format_number(last_deaths_7), delta=format_pct(wow_deaths))
                with b4:
                    st.metric("Rata-rata New deaths (7D)", format_number(last_7["New deaths 7D"].iloc[-1] if "New deaths 7D" in last_7.columns else np.nan))

                plot_cols = [c for c in ["New cases", "New cases 7D", "New deaths", "New deaths 7D"] if c in g2.columns]
                if plot_cols:
                    long = g2[["Date"] + plot_cols].melt(id_vars="Date", var_name="Metric", value_name="Value")
                    fig = px.line(long, x="Date", y="Value", color="Metric", title="Dinamika global (harian vs 7D avg)")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(pastelize(fig), use_container_width=True)

            with st.expander("2) Identifikasi Gelombang Kasus (berdasarkan 7-Days New Cases)"):
                if "New cases 7D" not in g2.columns or g2["New cases 7D"].isna().all():
                    st.info("Kolom New cases tidak tersedia untuk membangun gelombang.")
                else:
                    waves = detect_waves(g2["New cases 7D"], g2["Date"], min_prom_frac=0.25, min_gap_days=21)
                    if waves:
                        biggest = max(waves, key=lambda w: w["Peak value"])
                        st.markdown(
                            f"<div class='small-note'>Pada rentang ini terdeteksi {len(waves)} puncak dominan, puncak terbesar terjadi sekitar {biggest['Peak']} dengan nilai ~{format_number(biggest['Peak value'])}.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("<div class='small-note'>Tidak ada puncak yang cukup menonjol pada rentang ini (indikasi gelombang tidak kuat).</div>", unsafe_allow_html=True)

                    fig = px.line(g2, x="Date", y="New cases 7D", title="New cases 7D (indikasi gelombang)")
                    for wv in waves:
                        fig.add_vline(x=pd.to_datetime(wv["Peak"]), line_width=2)
                    st.plotly_chart(pastelize(fig), use_container_width=True)

                    if waves:
                        st.dataframe(pd.DataFrame(waves).sort_values("Peak value", ascending=False).reset_index(drop=True))

            with st.expander("3) Kasus vs Kematian: Analisis Jeda Waktu (Lag)"):
                if "New cases 7D" not in g2.columns or "New deaths 7D" not in g2.columns:
                    st.info("Butuh kolom New cases dan New deaths untuk analisis lag.")
                else:
                    lag_max = st.slider("Maksimum lag (hari):", min_value=7, max_value=60, value=35, step=1)
                    ldf = lag_correlation(g2["New cases 7D"], g2["New deaths 7D"], max_lag=lag_max).dropna()
                    if ldf.empty:
                        st.info("Korelasi tidak dapat dihitung pada rentang data ini.")
                    else:
                        best = ldf.loc[ldf["Correlation"].idxmax()]
                        st.markdown(
                            f"<div class='small-note'>Bagian ini mengukur jeda waktu kematian terhadap kasus (menggunakan korelasi pada beberapa lag). Korelasi tertinggi terjadi pada lag sekitar <b>{int(best['Lag (days)'])} hari</b>.</div>",
                            unsafe_allow_html=True,
                        )
                        fig = px.line(ldf, x="Lag (days)", y="Correlation", title="Korelasi New deaths 7D terhadap New cases 7D pada berbagai lag")
                        st.plotly_chart(pastelize(fig), use_container_width=True)

            snap = build_country_metrics_snapshot()

            with st.expander("4) CFR & Recovery Rate: Highlight pada Setiap Negara"):
                if snap.empty or "Country/Region" not in snap.columns:
                    st.info("country_wise_latest.csv tidak cukup untuk analisis ini.")
                else:
                    work = snap.copy()
                    if "Confirmed" in work.columns:
                        work = work[work["Confirmed"].fillna(0) >= float(cfr_floor)]
                    work = work.replace([np.inf, -np.inf], np.nan)

                    show_cfr = work.dropna(subset=["CFR"]).copy() if "CFR" in work.columns else pd.DataFrame()
                    if show_cfr.empty:
                        st.info("Tidak ada data CFR yang memenuhi filter.")
                    else:
                        topn = st.slider("Top N CFR:", min_value=10, max_value=60, value=25, step=1)
                        show = show_cfr.sort_values("CFR", ascending=False).head(topn)
                        top_country = show.iloc[0]["Country/Region"]
                        top_cfr = show.iloc[0]["CFR"]

                        st.markdown(
                            f"<div class='small-note'>CFR (Deaths/Confirmed) menunjukkan proporsi kematian dari kasus yang terkonfirmasi. Pada filter ini, CFR tertinggi ada pada <b>{top_country}</b> (~{format_pct(top_cfr, 2)}).</div>",
                            unsafe_allow_html=True,
                        )

                        fig = px.bar(show, x="Country/Region", y="CFR", title=f"Top {topn} CFR (minimum confirmed {cfr_floor})")
                        fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".2%")
                        st.plotly_chart(pastelize(fig), use_container_width=True)

                        if "Recovery rate" in show.columns:
                            fig2 = px.bar(show.sort_values("Recovery rate", ascending=False), x="Country/Region", y="Recovery rate", title="Recovery rate (Recovered/Confirmed) — negara yang sama")
                            fig2.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".2%")
                            st.plotly_chart(pastelize(fig2), use_container_width=True)

                        if show_tables:
                            st.dataframe(show[["Country/Region", "Confirmed", "Deaths", "Recovered", "Active", "CFR", "Recovery rate"]].reset_index(drop=True))

            with st.expander("5) Per 1 Juta Penduduk"):
                if snap.empty or "_pop" not in snap.columns or snap["_pop"].dropna().empty:
                    st.info("Kolom populasi tidak tersedia pada worldometer_data.csv (atau tidak bisa dicocokkan).")
                else:
                    metric_pc_options = [c for c in ["Confirmed per 1M", "Deaths per 1M", "Recovered per 1M", "Active per 1M"] if c in snap.columns]
                    metric_pc = st.selectbox("Metrik per 1M:", options=metric_pc_options, index=0)

                    topn = st.slider("Top N per 1M:", min_value=10, max_value=80, value=30, step=5)
                    tmp = snap.dropna(subset=[metric_pc]).copy()
                    tmp = tmp[tmp[metric_pc] >= 0]
                    show = tmp.sort_values(metric_pc, ascending=False).head(topn)

                    if not show.empty:
                        st.markdown(
                            f"<div class='small-note'>Bagian ini membantu membandingkan beban kasus antarnegara yang memiliki ukuran populasi berbeda. Untuk metrik ini, peringkat teratas adalah <b>{show.iloc[0]['Country/Region']}</b>.</div>",
                            unsafe_allow_html=True,
                        )

                    fig = px.bar(show, x="Country/Region", y=metric_pc, title=f"Top {topn} — {metric_pc}")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(pastelize(fig), use_container_width=True)

                    if show_tables:
                        cols = ["Country/Region", metric_pc, "_pop"]
                        if "CFR" in show.columns:
                            cols.append("CFR")
                        st.dataframe(show[cols].reset_index(drop=True))

            hot = build_country_14d_hotspots()

            with st.expander("6) Hotspots : 14 Hari Terakhir vs 14 Hari Sebelumnya)"):
                if hot.empty:
                    st.info("Tidak ada data hotspots (butuh full_grouped.csv + kolom New cases).")
                else:
                    max_prev = int(max(1000, hot["Prev 14D"].max()))
                    min_prev = st.slider("Minimum Prev 14D (filter):", min_value=0, max_value=max_prev, value=1000, step=500)

                    hw = hot.copy()
                    hw = hw[hw["Prev 14D"] >= float(min_prev)]
                    hw = hw.replace([np.inf, -np.inf], np.nan).dropna(subset=["Pct change"])
                    topn = st.slider("Top N perubahan:", min_value=10, max_value=60, value=20, step=5)

                    inc = hw.sort_values("Pct change", ascending=False).head(topn)
                    dec = hw.sort_values("Pct change", ascending=True).head(topn)

                    if not inc.empty and not dec.empty:
                        st.markdown(
                            f"<div class='small-note'>Menunjukkan negara yang mengalami perubahan paling tajam. Pada filter ini, lonjakan terbesar dipimpin <b>{inc.iloc[0]['Country/Region']}</b>, sedangkan penurunan terbesar dipimpin <b>{dec.iloc[0]['Country/Region']}</b>.</div>",
                            unsafe_allow_html=True,
                        )

                    t1, t2 = st.columns(2)
                    with t1:
                        fig = px.bar(inc, x="Country/Region", y="Pct change", title=f"Top {topn} lonjakan (Last 14D vs Prev 14D)")
                        fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
                        st.plotly_chart(pastelize(fig), use_container_width=True)
                    with t2:
                        fig = px.bar(dec, x="Country/Region", y="Pct change", title=f"Top {topn} penurunan (Last 14D vs Prev 14D)")
                        fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
                        st.plotly_chart(pastelize(fig), use_container_width=True)

                    if show_tables:
                        st.dataframe(hw.sort_values("Pct change", ascending=False).reset_index(drop=True))

            with st.expander("7) Analisis Negara Pilihan (Tren, Volatilitas, dan Puncak)"):
                if full_grouped.empty or "Country/Region" not in full_grouped.columns or "Date" not in full_grouped.columns:
                    st.info("full_grouped.csv tidak valid untuk analisis negara.")
                else:
                    c1, c2, c3 = st.columns([1.3, 1, 1])
                    with c1:
                        ctry = st.selectbox("Negara:", options=all_countries, index=all_countries.index("Indonesia") if "Indonesia" in all_countries else 0)
                    metric_opts = [m for m in ["New cases", "New deaths", "Confirmed", "Deaths", "Recovered", "Active"] if m in full_grouped.columns]
                    with c2:
                        met = st.selectbox("Metrik:", options=metric_opts, index=0)
                    with c3:
                        logy = st.checkbox("Skala log", value=False)

                    cdf = full_grouped[(full_grouped["Country/Region"] == ctry)].dropna(subset=["Date"]).sort_values("Date").copy()
                    cdf = cdf[cdf["Date"].dt.date.between(analysis_range[0], analysis_range[1])]
                    if cdf.empty:
                        st.info("Tidak ada data untuk negara/rentang ini.")
                    else:
                        cdf[met] = pd.to_numeric(cdf[met], errors="coerce").fillna(0)
                        window = st.slider("Rolling window (hari):", min_value=3, max_value=28, value=7, step=1)
                        cdf["Rolling"] = cdf[met].rolling(window, min_periods=max(2, window // 2)).mean()
                        cdf["Pct"] = cdf[met].pct_change().replace([np.inf, -np.inf], np.nan)

                        peak_idx = int(cdf["Rolling"].idxmax())
                        peak_date = pd.to_datetime(cdf.loc[peak_idx, "Date"]).date()
                        peak_val = float(cdf.loc[peak_idx, "Rolling"])

                        vol = float(np.nanstd(cdf["Pct"])) if cdf["Pct"].notna().any() else np.nan

                        st.markdown(
                            f"<div class='small-note'>Pada rentang ini, puncak <b>{met}</b> (rolling {window} hari) terjadi sekitar <b>{peak_date}</b>. Volatilitas perubahan hariannya ~{format_float(vol*100, 2)}%.</div>",
                            unsafe_allow_html=True,
                        )

                        fig = px.line(cdf, x="Date", y=["Rolling", met], title=f"{ctry} — {met} (harian vs rolling {window} hari)")
                        if logy:
                            fig.update_yaxes(type="log")
                        fig.update_layout(hovermode="x unified")
                        fig.add_vline(x=pd.to_datetime(peak_date), line_width=2)
                        st.plotly_chart(pastelize(fig), use_container_width=True)

                        st.metric("Volatilitas (% change harian, std dev)", f"{format_float(vol*100, 2)}%")

                        if show_tables:
                            st.dataframe(cdf[["Date", met, "Rolling", "Pct"]].reset_index(drop=True))

            with st.expander("8) Perbandingan Antar Negara"):
                if snap.empty:
                    st.info("Snapshot negara tidak tersedia.")
                else:
                    choices = st.multiselect("Pilih negara (maks 6):", options=sorted(snap["Country/Region"].dropna().unique()), default=default_countries)[:6]
                    if not choices:
                        st.info("Pilih minimal satu negara.")
                    else:
                        features = [cand for cand in ["Confirmed per 1M", "Deaths per 1M", "CFR", "Recovery rate"] if cand in snap.columns]
                        if not features:
                            st.info("Tidak ada fitur yang cukup untuk perbandingan.")
                        else:
                            sub = snap[snap["Country/Region"].isin(choices)].copy()
                            for f in features:
                                sub[f] = pd.to_numeric(sub[f], errors="coerce")
                            sub = sub.dropna(subset=features, how="any")
                            if sub.empty:
                                st.info("Data negara terpilih tidak lengkap pada fitur yang dipilih.")
                            else:
                                st.markdown(
                                    "<div class='small-note'>Menampilkan skor 0–1 atas beberapa indikator agar bisa dibandingkan dalam satu tampilan. Semakin besar area, semakin tinggi skor relatif pada indikator yang dipilih.</div>",
                                    unsafe_allow_html=True,
                                )

                                f_min = sub[features].min()
                                f_max = sub[features].max()
                                norm = (sub[features] - f_min) / (f_max - f_min).replace(0, np.nan)
                                norm["Country/Region"] = sub["Country/Region"].values
                                long = norm.melt(id_vars="Country/Region", var_name="Feature", value_name="Score").dropna()

                                fig = px.line_polar(long, r="Score", theta="Feature", color="Country/Region", line_close=True, title="Skor ternormalisasi (0–1)")
                                st.plotly_chart(pastelize(fig), use_container_width=True)

            with st.expander("9) Segmentasi Negara (Cluster Sederhana)"):
                if snap.empty:
                    st.info("Snapshot negara tidak tersedia.")
                else:
                    base = snap.copy()
                    feat = [cand for cand in ["Confirmed per 1M", "Deaths per 1M", "CFR", "Recovery rate"] if cand in base.columns]
                    base = base.dropna(subset=feat).copy() if feat else pd.DataFrame()
                    if base.empty or len(feat) < 2:
                        st.info("Tidak cukup fitur untuk melakukan cluster.")
                    else:
                        k = st.slider("Jumlah cluster (k):", min_value=2, max_value=6, value=4, step=1)

                        base_num = base[feat].replace([np.inf, -np.inf], np.nan).dropna()
                        X = base_num.to_numpy(dtype=float)
                        keys = base_num.index

                        mu = X.mean(axis=0)
                        sd = X.std(axis=0)
                        sd = np.where(sd == 0, 1, sd)
                        Xz = (X - mu) / sd

                        labels, _ = simple_kmeans(Xz, k=k, iters=40, seed=42)
                        base2 = base.loc[keys].copy()
                        base2["Cluster"] = labels.astype(int)

                        cluster_sizes = base2["Cluster"].value_counts().sort_index()
                        st.markdown(
                            f"<div class='small-note'>Cluster mengelompokkan negara dengan pola indikator yang mirip. Dengan k={k}, ukuran cluster: "
                            + ", ".join([f"Cluster {i}: {int(cluster_sizes.get(i,0))}" for i in range(k)])
                            + ".</div>",
                            unsafe_allow_html=True,
                        )

                        x_axis = "Confirmed per 1M" if "Confirmed per 1M" in base2.columns else feat[0]
                        y_axis = "Deaths per 1M" if "Deaths per 1M" in base2.columns else feat[1]

                        fig = px.scatter(
                            base2,
                            x=x_axis,
                            y=y_axis,
                            color=base2["Cluster"].astype(str),
                            hover_name="Country/Region",
                            title="Scatter cluster negara (berbasis snapshot)",
                        )
                        st.plotly_chart(pastelize(fig), use_container_width=True)

                        st.dataframe(base2.groupby("Cluster")[feat].median(numeric_only=True).reset_index())

                        if show_tables:
                            st.dataframe(base2[["Country/Region", "Cluster"] + feat].sort_values(["Cluster", x_axis], ascending=[True, False]).reset_index(drop=True))

            with st.expander("10) Data & Anomali"):
                neg_cases = int((g2["New cases"] < 0).sum()) if "New cases" in g2.columns else 0
                neg_deaths = int((g2["New deaths"] < 0).sum()) if "New deaths" in g2.columns else 0
                dupe = 0
                if not full_grouped.empty and "Country/Region" in full_grouped.columns and "Date" in full_grouped.columns:
                    dupe = int(full_grouped.dropna(subset=["Country/Region", "Date"]).duplicated(["Country/Region", "Date"]).sum())

                st.markdown(
                    f"<div class='small-note'>Cek untuk mendeteksi pola yang sering muncul pada data pelaporan: nilai negatif (revisi/backfill) dan duplikasi. Pada rentang analisis: New cases negatif={neg_cases}, New deaths negatif={neg_deaths}. Duplikasi (Country/Region, Date) di full_grouped={dupe}.</div>",
                    unsafe_allow_html=True,
                )

                msgs = []
                if neg_cases > 0:
                    msgs.append(f"- Ditemukan {neg_cases} nilai negatif pada New cases (rentang analisis).")
                if neg_deaths > 0:
                    msgs.append(f"- Ditemukan {neg_deaths} nilai negatif pada New deaths (rentang analisis).")
                if dupe > 0:
                    msgs.append(f"- Ditemukan {dupe} duplikasi baris pada (Country/Region, Date) di full_grouped.csv.")
                if msgs:
                    st.write("\n".join(msgs))
                else:
                    st.write("Tidak ada anomali dasar terdeteksi pada cek cepat (negatif/duplikasi).")

            with st.expander("Download ringkasan analisis (CSV)"):
                out = build_country_metrics_snapshot()
                if out.empty:
                    st.info("Tidak ada ringkasan untuk diunduh.")
                else:
                    cols = ["Country/Region"]
                    for c in ["Confirmed", "Deaths", "Recovered", "Active", "CFR", "Recovery rate", "Confirmed per 1M", "Deaths per 1M", "_pop"]:
                        if c in out.columns:
                            cols.append(c)
                    out2 = out[cols].copy()
                    csv = out2.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Download summary metrics (CSV)", data=csv, file_name="covid_analysis_summary.csv", mime="text/csv")


elif page == "📊 Country Dashboard":
    st.header("📊 Country Dashboard")
    st.markdown("<div class='small-note'>Pilih negara untuk melihat tren kasus dari waktu ke waktu.</div>", unsafe_allow_html=True)

    if not all_countries:
        st.warning("Kolom Country/Region tidak ditemukan pada full_grouped.csv")
    else:
        c1, c2 = st.columns(2)
        with c1:
            country = st.selectbox(
                "Pilih negara:",
                options=all_countries,
                index=all_countries.index("Indonesia") if "Indonesia" in all_countries else 0,
            )
        with c2:
            log_scale = st.checkbox("Gunakan skala logaritmik?", value=False)

        country_df = full_grouped[full_grouped["Country/Region"] == country].dropna(subset=["Date"]).sort_values("Date")
        if country_df.empty:
            st.warning("Tidak ada data untuk negara ini.")
        else:
            latest = country_df.iloc[-1]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Confirmed", format_number(latest.get("Confirmed")), delta=format_number(latest.get("New cases", 0)))
            with c2:
                st.metric("Total Deaths", format_number(latest.get("Deaths")), delta=format_number(latest.get("New deaths", 0)))
            with c3:
                st.metric("Total Recovered", format_number(latest.get("Recovered")), delta=format_number(latest.get("New recovered", 0)))
            with c4:
                st.metric("Active Cases", format_number(latest.get("Active")))

            col1, col2 = st.columns(2)

            with col1:
                pie_country = pd.DataFrame(
                    {
                        "Status": ["Active", "Recovered", "Deaths"],
                        "Count": [
                            max(float(latest.get("Active", 0) or 0), 0),
                            max(float(latest.get("Recovered", 0) or 0), 0),
                            max(float(latest.get("Deaths", 0) or 0), 0),
                        ],
                    }
                )
                fig_pie_country = px.pie(pie_country, names="Status", values="Count", hole=0.45, title=f"Komposisi kasus di {country}")
                fig_pie_country.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(pastelize(fig_pie_country), use_container_width=True)

            with col2:
                new_cols = [c for c in ["New cases", "New deaths", "New recovered"] if c in country_df.columns]
                if new_cols:
                    long_new = country_df[["Date"] + new_cols].melt(id_vars="Date", value_vars=new_cols, var_name="Jenis", value_name="Jumlah")
                    fig_new_country = px.bar(long_new, x="Date", y="Jumlah", color="Jenis", title=f"Kasus baru harian di {country}")
                    fig_new_country.update_layout(hovermode="x unified")
                    st.plotly_chart(pastelize(fig_new_country), use_container_width=True)

            metrics = [m for m in ["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered"] if m in country_df.columns]
            selected_metrics = st.multiselect(
                "Pilih metrik:",
                options=metrics,
                default=[m for m in ["Confirmed", "Deaths", "Recovered"] if m in metrics],
            )
            if selected_metrics:
                long_c = country_df[["Date"] + selected_metrics].melt(id_vars="Date", value_vars=selected_metrics, var_name="Metric", value_name="Value")
                fig = px.line(long_c, x="Date", y="Value", color="Metric", title=f"Perkembangan kasus di {country}")
                if log_scale:
                    fig.update_yaxes(type="log")
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(pastelize(fig), use_container_width=True)

            with st.expander("📋 Data harian negara"):
                st.dataframe(country_df.reset_index(drop=True))


elif page == "📈 Country Comparison":
    st.header("📈 Country Comparison")
    st.markdown("<div class='small-note'>Bandingkan beberapa negara pada metrik yang sama untuk melihat perbedaan pola dan level kasus Covid-19.</div>", unsafe_allow_html=True)

    if not all_countries:
        st.warning("Kolom Country/Region tidak ditemukan pada full_grouped.csv")
    else:
        countries = st.multiselect("Pilih beberapa negara:", options=all_countries, default=default_countries)
        metric_options = [m for m in ["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths"] if m in full_grouped.columns]
        if not metric_options:
            st.warning("Tidak ada kolom metrik yang cocok pada full_grouped.csv")
        else:
            metric = st.selectbox("Metrik:", options=metric_options, index=0)
            date_range = st.slider("Rentang tanggal:", min_value=min_date, max_value=max_date, value=(min_date, max_date))

            if not countries:
                st.info("Pilih minimal satu negara.")
            else:
                compare_df = full_grouped.dropna(subset=["Date"]).copy()
                compare_df = compare_df[
                    (compare_df["Country/Region"].isin(countries))
                    & (compare_df["Date"].dt.date.between(date_range[0], date_range[1]))
                ]
                if compare_df.empty:
                    st.warning("Tidak ada data untuk kombinasi filter ini.")
                else:
                    fig = px.line(compare_df, x="Date", y=metric, color="Country/Region", title=f"{metric} dari waktu ke waktu")
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(pastelize(fig), use_container_width=True)

                    snap_date = st.date_input("Tanggal snapshot:", value=date_range[1], min_value=date_range[0], max_value=date_range[1])
                    snap_df = compare_df[compare_df["Date"].dt.date == snap_date][["Country/Region", metric]].sort_values(metric, ascending=False)
                    st.dataframe(snap_df.reset_index(drop=True))


elif page == "🗽 USA View":
    st.header("🗽 USA Country / State View")
    st.markdown("<div class='small-note'>Pilih state untuk melihat tren.</div>", unsafe_allow_html=True)

    if "Province_State" not in usa_county.columns:
        st.warning("Kolom Province_State tidak ditemukan pada usa_county_wise.csv")
    else:
        states = sorted(usa_county["Province_State"].dropna().unique())
        state = st.selectbox("Pilih State:", options=states, index=states.index("New York") if "New York" in states else 0)

        state_df = usa_county[usa_county["Province_State"] == state].copy()
        if state_df.empty:
            st.warning("Tidak ada data untuk state ini.")
        else:
            metric_cols = [c for c in ["Confirmed", "Deaths"] if c in state_df.columns]
            if "Date" not in state_df.columns or not metric_cols:
                st.warning("Kolom Date/Confirmed/Deaths tidak lengkap pada usa_county_wise.csv")
            else:
                state_daily = state_df.dropna(subset=["Date"]).groupby("Date")[metric_cols].sum().reset_index().sort_values("Date")
                latest = state_daily.iloc[-1]

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Total Confirmed", format_number(latest.get("Confirmed")))
                with c2:
                    st.metric("Total Deaths", format_number(latest.get("Deaths")))

                fig = px.line(state_daily, x="Date", y=metric_cols, title=f"Perkembangan kasus di {state}")
                st.plotly_chart(pastelize(fig), use_container_width=True)

                if "Admin2" in state_df.columns:
                    latest_state = state_df[state_df["Date"] == state_df["Date"].max()]
                    county_agg = latest_state.groupby("Admin2")[metric_cols].sum().reset_index()
                    if "Confirmed" in county_agg.columns:
                        top_counties = county_agg.sort_values("Confirmed", ascending=False).head(10)
                        fig_county = px.bar(top_counties, x="Admin2", y="Confirmed", hover_data=[c for c in ["Deaths"] if c in top_counties.columns], title=f"🔝 10 county dengan kasus tertinggi di {state}")
                        fig_county.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(pastelize(fig_county), use_container_width=True)

                with st.expander("📋 Data county mentah"):
                    show_cols = [c for c in ["Date", "Admin2", "Province_State", "Confirmed", "Deaths"] if c in usa_county.columns]
                    st.dataframe(usa_county[usa_county["Province_State"] == state][show_cols].sort_values(["Date", "Admin2"]))


elif page == "🇮🇩 Indonesia Province View":
    indonesia_province_view()


elif page == "🔥 Insights & Hotspots":
    st.header("🔥 Insights & Hotspots")
    st.markdown("<div class='small-note'>Ringkasan singkat mengenai berbagai indikator.</div>", unsafe_allow_html=True)

    if country_latest.empty or "Country/Region" not in country_latest.columns:
        st.warning("country_wise_latest.csv tidak valid / kolom Country/Region tidak ditemukan.")
    else:
        base = country_latest.copy()
        base["_key"] = base["Country/Region"].apply(normalize_country)

        _, pop_col, pop_df = get_population_table()
        if pop_col is not None and not pop_df.empty:
            base = base.merge(pop_df, on="_key", how="left")

        snapshot_metrics = [m for m in ["Confirmed", "Deaths", "Recovered", "Active"] if m in base.columns]
        if not snapshot_metrics:
            st.warning("Tidak ada kolom Confirmed/Deaths/Recovered/Active pada country_wise_latest.csv")
        else:
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                metric = st.selectbox("Metrik snapshot:", options=snapshot_metrics, index=0)
            with c2:
                top_n = st.slider("Top N:", min_value=5, max_value=50, value=15, step=1)
            with c3:
                per_capita = st.checkbox("Per 1M (jika ada populasi)", value=False)

            work = base.copy()
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
            per_label = metric

            if per_capita and pop_col is not None and "_pop" in work.columns:
                work[metric + " per 1M"] = (pd.to_numeric(work[metric], errors="coerce") / pd.to_numeric(work["_pop"], errors="coerce")) * 1_000_000
                per_label = metric + " per 1M"

            work["_m"] = pd.to_numeric(work.get(per_label, work.get(metric)), errors="coerce")
            work = work.dropna(subset=["_m"])
            work = work[work["_m"] >= 0]
            top_df = work.sort_values("_m", ascending=False).head(top_n)

            if not top_df.empty:
                st.markdown(
                    f"<div class='small-note'>Ranking ini menunjukkan negara dengan nilai tertinggi untuk metrik yang dipilih. Peringkat teratas saat ini adalah <b>{top_df.iloc[0]['Country/Region']}</b>.</div>",
                    unsafe_allow_html=True,
                )

            fig_rank = px.bar(top_df, x="Country/Region", y="_m", title=f"Top {top_n} — {per_label}")
            fig_rank.update_layout(xaxis_tickangle=-45, yaxis_title=per_label)
            st.plotly_chart(pastelize(fig_rank), use_container_width=True)

            work["Confirmed"] = pd.to_numeric(work.get("Confirmed"), errors="coerce")
            work["Deaths"] = pd.to_numeric(work.get("Deaths"), errors="coerce")

            x_col, y_col = "Confirmed", "Deaths"
            if per_capita and pop_col is not None and "_pop" in work.columns:
                work["Confirmed per 1M"] = (work["Confirmed"] / pd.to_numeric(work["_pop"], errors="coerce")) * 1_000_000
                work["Deaths per 1M"] = (work["Deaths"] / pd.to_numeric(work["_pop"], errors="coerce")) * 1_000_000
                x_col, y_col = "Confirmed per 1M", "Deaths per 1M"

            fig_scatter = px.scatter(
                work.dropna(subset=[x_col, y_col]),
                x=x_col,
                y=y_col,
                hover_name="Country/Region",
                size="Confirmed" if "Confirmed" in work.columns else None,
                title=f"{y_col} vs {x_col}",
            )
            fig_scatter.update_layout(hovermode="closest")
            st.plotly_chart(pastelize(fig_scatter), use_container_width=True)

            with st.expander("📋 Tabel"):
                cols = ["Country/Region"]
                for c in [metric, per_label, "Deaths", "Recovered", "Active"]:
                    if c and c in top_df.columns and c not in cols:
                        cols.append(c)
                if pop_col is not None and "_pop" in top_df.columns:
                    cols.append("_pop")
                st.dataframe(top_df[cols].reset_index(drop=True))


elif page == "⏱️ Timelapse":
    st.header("⏱️ Timelapse")
    st.markdown("<div class='small-note'>Menampilkan peta indikator dari waktu ke waktu secara global.</div>", unsafe_allow_html=True)

    if full_grouped.empty or "Date" not in full_grouped.columns or "Country/Region" not in full_grouped.columns:
        st.warning("full_grouped.csv tidak valid / kolom Date atau Country/Region tidak ditemukan.")
    else:
        tl_metrics = [m for m in ["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered"] if m in full_grouped.columns]
        if not tl_metrics:
            st.warning("Tidak ada kolom metrik yang cocok pada full_grouped.csv")
        else:
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                tl_metric = st.selectbox("Metrik timelapse:", options=tl_metrics, index=0)
            with c2:
                tl_scale = st.selectbox("Skala warna:", options=["Linear", "Log"], index=0)
            with c3:
                speed = st.slider("Kecepatan (ms/frame):", min_value=50, max_value=1200, value=250, step=50)

            tl_range = st.slider("Rentang tanggal:", min_value=min_date, max_value=max_date, value=(min_date, max_date))

            tl_df = full_grouped.dropna(subset=["Date"]).copy()
            tl_df = tl_df[tl_df["Date"].dt.date.between(tl_range[0], tl_range[1])]
            tl_df = tl_df.groupby(["Date", "Country/Region"], as_index=False)[tl_metric].sum()
            tl_df["DateStr"] = tl_df["Date"].dt.strftime("%Y-%m-%d")
            tl_df[tl_metric] = pd.to_numeric(tl_df[tl_metric], errors="coerce").fillna(0)

            color_col = tl_metric
            if tl_scale == "Log":
                tl_df["_color"] = np.log10(tl_df[tl_metric].clip(lower=0) + 1)
                color_col = "_color"

            fig = px.choropleth(
                tl_df,
                locations="Country/Region",
                locationmode="country names",
                color=color_col,
                hover_name="Country/Region",
                animation_frame="DateStr",
                title=f"Timelapse — {tl_metric}",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))

            try:
                if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
                    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = speed
                    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = max(int(speed * 0.6), 0)
            except Exception:
                pass

            st.plotly_chart(pastelize(fig), use_container_width=True)


elif page == "📑 Data Explorer":
    st.header("📑 Data Explorer")
    st.markdown("<div class='small-note'>Preview dataset mentah.</div>", unsafe_allow_html=True)

    dataset_name = st.selectbox(
        "Pilih dataset:",
        options=[
            "day_wise",
            "full_grouped",
            "country_wise_latest",
            "worldometer_data",
            "usa_county_wise",
            "covid_19_clean_complete",
        ],
    )

    if dataset_name == "day_wise":
        df = day_wise.copy()
    elif dataset_name == "full_grouped":
        df = full_grouped.copy()
    elif dataset_name == "country_wise_latest":
        df = country_latest.copy()
    elif dataset_name == "worldometer_data":
        df = worldometer.copy()
    elif dataset_name == "usa_county_wise":
        df = usa_county.copy()
    else:
        df = clean.copy()

    st.write(f"Menampilkan {dataset_name} — {df.shape[0]:,} baris, {df.shape[1]} kolom.")
    st.dataframe(df.head(500))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV (subset ditampilkan)", data=csv, file_name=f"{dataset_name}.csv", mime="text/csv")


elif page == "ℹ️ About":
    st.title("ℹ️ Tentang Kami — COVID-19 Global Dashboard")
    st.markdown("<div class='small-note'>Ringkasan singkat dashboard.</div>", unsafe_allow_html=True)

    st.write(
        "Dashboard interaktif berbasis Streamlit untuk memvisualisasikan perkembangan kasus COVID-19 "
        "secara global."
    )

    st.subheader("Halaman Ini Memuat📚")
    st.write(
        "- Overview\n"
        "- Global Timeline Story Mode\n"
        "- Global Map\n"
        "- Analisis Data Perkembangan COVID-19\n"
        "- Country Dashboard\n"
        "- Country Comparison\n"
        "- USA View\n"
        "- Insights & Hotspots\n"
        "- Timelapse\n"
        "- Data Explorer"
    )

    st.subheader("Sumber Data")
    st.write("Dataset berasal dari Kaggle: COVID-19 Corona Virus Report (imdevskp).")

    st.subheader("Tim")
    st.write(
        "Kelompok Lebah Ganteng:\n"
        "- Muhammad Dimas Sudirman (021002404001)\n"
        "- Ari Wahyu Patriangga (021002404007)\n"
        "- Lola Aritasari (021002404004)"
    )
