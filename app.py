import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------
# Page configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Global Dashboard",
    page_icon="🦠",
    layout="wide",
)

# -----------------------------------------------------------
# Data loading with caching
# -----------------------------------------------------------
@st.cache_data
def load_data():
    # Semua CSV di folder yang sama dengan app.py
    day_wise = pd.read_csv("day_wise.csv")
    full_grouped = pd.read_csv("full_grouped.csv")
    country_latest = pd.read_csv("country_wise_latest.csv")
    worldometer = pd.read_csv("worldometer_data.csv")
    usa_county = pd.read_csv("usa_county_wise.csv")
    clean = pd.read_csv("covid_19_clean_complete.csv")

    # Date -> datetime
    day_wise["Date"] = pd.to_datetime(day_wise["Date"])
    full_grouped["Date"] = pd.to_datetime(full_grouped["Date"])
    clean["Date"] = pd.to_datetime(clean["Date"])
    usa_county["Date"] = pd.to_datetime(usa_county["Date"])

    # Sedikit rapihin tipe data numerik
    for df in [day_wise, full_grouped, country_latest, worldometer, usa_county, clean]:
        for col in df.select_dtypes(include="object").columns:
            # kolom label jangan diotak-atik
            if col.lower() in [
                "country/region",
                "province/state",
                "admin2",
                "province_state",
                "country_region",
                "combined_key",
                "continent",
                "who region",
                "who_region",
            ]:
                continue
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return day_wise, full_grouped, country_latest, worldometer, usa_county, clean


day_wise, full_grouped, country_latest, worldometer, usa_county, clean = load_data()

# Helper global
# all_dates sebagai Timestamp -> ambil tanggal Python (date) untuk slider
all_dates = pd.to_datetime(day_wise["Date"].unique())
min_date = all_dates.min().date()
max_date = all_dates.max().date()

all_countries = sorted(full_grouped["Country/Region"].unique())
default_countries = [
    c for c in ["Indonesia", "US", "Italy", "China", "India"] if c in all_countries
]
if not default_countries:
    default_countries = all_countries[:3]

# -----------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Pilih halaman:",
    (
        "🏠 Overview",
        "🌍 Global Map",
        "📊 Country Dashboard",
        "📈 Country Comparison",
        "🗽 USA View",
        "📑 Data Explorer",
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset**: COVID-19 Corona Virus Report  \nSumber: Kaggle – imdevskp"
)

# 🔽 Tambahan identitas kelompok
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Kelompok Lebah Ganteng**  \n"
    "Muhammad Dimas Sudirman – 021002404001  \n"
    "Ari Wahyu Patriangga – 021002404007  \n"
    "Lola Aritasari – 021002404004"
)

# -----------------------------------------------------------
# Helper formatting
# -----------------------------------------------------------
def format_number(x):
    try:
        return f"{int(x):,}"
    except (ValueError, TypeError):
        return "-"


# ===========================================================
# 1. OVERVIEW
# ===========================================================
if page == "🏠 Overview":
    st.markdown(
        "<h1 style='text-align:center'>🦠 COVID-19 Global Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.write(
        "Ringkasan perkembangan kasus COVID-19 secara global berdasarkan data time-series."
    )

    latest_row = day_wise.sort_values("Date").iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Total Confirmed",
            format_number(latest_row["Confirmed"]),
            delta=format_number(latest_row["New cases"]),
        )
    with c2:
        st.metric(
            "Total Deaths",
            format_number(latest_row["Deaths"]),
            delta=format_number(latest_row["New deaths"]),
        )
    with c3:
        st.metric(
            "Total Recovered",
            format_number(latest_row["Recovered"]),
            delta=format_number(latest_row["New recovered"]),
        )
    with c4:
        st.metric("Active Cases", format_number(latest_row["Active"]))

    # ---------- NEW: pie chart + top countries ----------
    st.markdown("### 🌐 Komposisi kasus global & negara dengan kasus terbanyak")
    col1, col2 = st.columns(2)

    # Pie: Active vs Recovered vs Deaths
    with col1:
        pie_data = pd.DataFrame(
            {
                "Status": ["Active", "Recovered", "Deaths"],
                "Count": [
                    max(latest_row["Active"], 0),
                    max(latest_row["Recovered"], 0),
                    max(latest_row["Deaths"], 0),
                ],
            }
        )
        fig_pie = px.pie(
            pie_data,
            names="Status",
            values="Count",
            hole=0.45,
            title="Komposisi kasus global (snapshot terakhir)",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Bar: top 10 countries by confirmed (country_wise_latest)
    with col2:
        top10 = (
            country_latest[country_latest["Confirmed"] > 0]
            .sort_values("Confirmed", ascending=False)
            .head(10)
        )
        fig_top = px.bar(
            top10,
            x="Country/Region",
            y="Confirmed",
            title="🔝 10 negara dengan kasus terkonfirmasi tertinggi",
        )
        fig_top.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top, use_container_width=True)

    # ---------- Existing line & bar ----------
    st.markdown("### 📈 Tren global dari waktu ke waktu")

    metrics_to_plot = st.multiselect(
        "Pilih metrik yang ditampilkan:",
        options=["Confirmed", "Deaths", "Recovered", "Active"],
        default=["Confirmed", "Deaths", "Recovered"],
    )

    if metrics_to_plot:
        long_df = day_wise[["Date"] + metrics_to_plot].melt(
            id_vars="Date",
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Value",
        )
        fig = px.line(
            long_df,
            x="Date",
            y="Value",
            color="Metric",
            title="Perkembangan kasus global",
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Kasus baru per hari (global)")
    fig_new = px.bar(
        day_wise,
        x="Date",
        y="New cases",
        title="New confirmed cases per day",
    )
    st.plotly_chart(fig_new, use_container_width=True)

    # ---------- NEW: correlation heatmap ----------
    st.markdown("### 🔍 Korelasi antar indikator global")
    corr_cols = [
        col
        for col in [
            "Confirmed",
            "Deaths",
            "Recovered",
            "Active",
            "New cases",
            "New deaths",
            "New recovered",
        ]
        if col in day_wise.columns
    ]
    if len(corr_cols) >= 2:
        corr_mat = day_wise[corr_cols].corr()
        fig_corr = px.imshow(
            corr_mat,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Matriks korelasi indikator global (harian)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# ===========================================================
# 2. GLOBAL MAP
# ===========================================================
elif page == "🌍 Global Map":
    st.header("🌍 Global Map")

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

    metric = st.selectbox("Pilih indikator:", map_metric_options, index=0)

    st.caption(
        "Data menggunakan agregasi **country_wise_latest.csv** (snapshot terbaru)."
    )

    map_df = country_latest.copy()
    map_df = map_df[map_df["Confirmed"] > 0]

    fig = px.choropleth(
        map_df,
        locations="Country/Region",
        locationmode="country names",
        color=metric,
        hover_name="Country/Region",
        color_continuous_scale="Reds",
        title=f"{metric} – latest snapshot",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Tabel ringkas"):
        show_cols = ["Country/Region", "Confirmed", "Deaths", "Recovered", "Active"]
        show_cols = [c for c in show_cols if c in map_df.columns]
        st.dataframe(
            map_df[show_cols]
            .sort_values("Confirmed", ascending=False)
            .reset_index(drop=True)
        )


# ===========================================================
# 3. COUNTRY DASHBOARD
# ===========================================================
elif page == "📊 Country Dashboard":
    st.header("📊 Country Dashboard")

    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox(
            "Pilih negara:",
            options=all_countries,
            index=all_countries.index("Indonesia") if "Indonesia" in all_countries else 0,
        )
    with c2:
        log_scale = st.checkbox("Gunakan skala logaritmik untuk grafik?", value=False)

    country_df = full_grouped[full_grouped["Country/Region"] == country].sort_values(
        "Date"
    )
    if country_df.empty:
        st.warning("Tidak ada data untuk negara ini.")
    else:
        latest = country_df.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Total Confirmed",
                format_number(latest["Confirmed"]),
                delta=format_number(latest.get("New cases", 0)),
            )
        with c2:
            st.metric(
                "Total Deaths",
                format_number(latest["Deaths"]),
                delta=format_number(latest.get("New deaths", 0)),
            )
        with c3:
            st.metric(
                "Total Recovered",
                format_number(latest["Recovered"]),
                delta=format_number(latest.get("New recovered", 0)),
            )
        with c4:
            st.metric("Active Cases", format_number(latest["Active"]))

        # ---------- NEW: pie + daily new ----------
        st.markdown(f"### 🇨🇮 Komposisi kasus & kasus baru di {country}")
        col1, col2 = st.columns(2)

        with col1:
            pie_country = pd.DataFrame(
                {
                    "Status": ["Active", "Recovered", "Deaths"],
                    "Count": [
                        max(latest["Active"], 0),
                        max(latest["Recovered"], 0),
                        max(latest["Deaths"], 0),
                    ],
                }
            )
            fig_pie_country = px.pie(
                pie_country,
                names="Status",
                values="Count",
                hole=0.45,
                title=f"Komposisi kasus di {country} (snapshot terakhir)",
            )
            fig_pie_country.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_country, use_container_width=True)

        with col2:
            new_cols = [
                col
                for col in ["New cases", "New deaths", "New recovered"]
                if col in country_df.columns
            ]
            if new_cols:
                long_new = country_df[["Date"] + new_cols].melt(
                    id_vars="Date",
                    value_vars=new_cols,
                    var_name="Jenis",
                    value_name="Jumlah",
                )
                fig_new_country = px.bar(
                    long_new,
                    x="Date",
                    y="Jumlah",
                    color="Jenis",
                    title=f"Kasus baru harian di {country}",
                )
                fig_new_country.update_layout(hovermode="x unified")
                st.plotly_chart(fig_new_country, use_container_width=True)

        # ---------- Existing time-series ----------
        st.markdown(f"### Tren waktu di {country}")
        metrics = [
            "Confirmed",
            "Deaths",
            "Recovered",
            "Active",
            "New cases",
            "New deaths",
            "New recovered",
        ]
        metrics = [m for m in metrics if m in country_df.columns]

        selected_metrics = st.multiselect(
            "Pilih metrik:",
            options=metrics,
            default=["Confirmed", "Deaths", "Recovered"],
        )

        if selected_metrics:
            long_c = country_df[["Date"] + selected_metrics].melt(
                id_vars="Date",
                value_vars=selected_metrics,
                var_name="Metric",
                value_name="Value",
            )
            fig = px.line(
                long_c,
                x="Date",
                y="Value",
                color="Metric",
                title=f"Perkembangan kasus di {country}",
            )
            if log_scale:
                fig.update_yaxes(type="log")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Data harian negara"):
            st.dataframe(country_df.reset_index(drop=True))


# ===========================================================
# 4. COUNTRY COMPARISON
# ===========================================================
elif page == "📈 Country Comparison":
    st.header("📈 Country Comparison")

    countries = st.multiselect(
        "Pilih beberapa negara:",
        options=all_countries,
        default=default_countries,
    )

    metric = st.selectbox(
        "Metrik:",
        options=["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths"],
        index=0,
    )

    # Slider pakai tipe date (bukan pandas.Timestamp)
    date_range = st.slider(
        "Rentang tanggal:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    if not countries:
        st.info("Pilih minimal satu negara.")
    else:
        # Bandingkan dengan .dt.date karena kolom Date di dataframe adalah datetime
        compare_df = full_grouped[
            (full_grouped["Country/Region"].isin(countries))
            & (full_grouped["Date"].dt.date.between(date_range[0], date_range[1]))
        ]

        if compare_df.empty:
            st.warning("Tidak ada data untuk kombinasi filter ini.")
        else:
            fig = px.line(
                compare_df,
                x="Date",
                y=metric,
                color="Country/Region",
                title=f"{metric} dari waktu ke waktu",
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Snapshot di tanggal tertentu")
            snap_date = st.date_input(
                "Pilih tanggal snapshot:",
                value=date_range[1],
                min_value=date_range[0],
                max_value=date_range[1],
            )

            # Cocokkan tanggal snapshot dengan Date di dataframe
            snap_df = compare_df[compare_df["Date"].dt.date == snap_date]
            snap_df = snap_df[["Country/Region", metric]].sort_values(
                metric, ascending=False
            )
            st.dataframe(snap_df.reset_index(drop=True))


# ===========================================================
# 5. USA VIEW
# ===========================================================
elif page == "🗽 USA View":
    st.header("🗽 USA County / State View")

    states = sorted(usa_county["Province_State"].dropna().unique())
    state = st.selectbox(
        "Pilih State:",
        options=states,
        index=states.index("New York") if "New York" in states else 0,
    )

    state_df = usa_county[usa_county["Province_State"] == state].copy()
    if state_df.empty:
        st.warning("Tidak ada data untuk state ini.")
    else:
        state_daily = (
            state_df.groupby("Date")[["Confirmed", "Deaths"]]
            .sum()
            .reset_index()
            .sort_values("Date")
        )

        c1, c2 = st.columns(2)
        latest = state_daily.iloc[-1]
        with c1:
            st.metric("Total Confirmed", format_number(latest["Confirmed"]))
        with c2:
            st.metric("Total Deaths", format_number(latest["Deaths"]))

        fig = px.line(
            state_daily,
            x="Date",
            y=["Confirmed", "Deaths"],
            title=f"Perkembangan kasus di {state}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- NEW: top 10 county ----------
        st.markdown(f"### Top 10 county dengan kasus tertinggi di {state}")
        if "Admin2" in state_df.columns:
            latest_state = state_df[state_df["Date"] == state_df["Date"].max()]
            county_agg = (
                latest_state.groupby("Admin2")[["Confirmed", "Deaths"]]
                .sum()
                .reset_index()
            )
            top_counties = county_agg.sort_values(
                "Confirmed", ascending=False
            ).head(10)
            fig_county = px.bar(
                top_counties,
                x="Admin2",
                y="Confirmed",
                hover_data=["Deaths"],
                title=f"🔝 10 county dengan kasus tertinggi di {state}",
            )
            fig_county.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_county, use_container_width=True)

        with st.expander("📋 Data county mentah"):
            show_cols = ["Date", "Admin2", "Province_State", "Confirmed", "Deaths"]
            show_cols = [c for c in show_cols if c in usa_county.columns]
            st.dataframe(
                usa_county[usa_county["Province_State"] == state][show_cols].sort_values(
                    ["Date", "Admin2"]
                )
            )


# ===========================================================
# 6. DATA EXPLORER
# ===========================================================
elif page == "📑 Data Explorer":
    st.header("📑 Data Explorer")

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

    st.write(f"Menampilkan **{dataset_name}** – {df.shape[0]:,} baris, {df.shape[1]} kolom.")
    st.dataframe(df.head(500))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download CSV (subset ditampilkan)",
        data=csv,
        file_name=f"{dataset_name}.csv",
        mime="text/csv",
    )
