import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="COVID-19 Global Dashboard", page_icon="🦠", layout="wide")

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

    events.append({"date": start_date, "title": "Awal periode data", "desc": "Mulai observasi time-series."})

    f_cases = first_above("New cases", frac=0.01)
    if f_cases is not None:
        events.append({"date": f_cases, "title": "Kasus baru mulai meningkat", "desc": "Hari pertama New cases mencapai ambang signifikan."})

    f_deaths = first_above("New deaths", frac=0.01)
    if f_deaths is not None:
        events.append({"date": f_deaths, "title": "Kematian baru mulai meningkat", "desc": "Hari pertama New deaths mencapai ambang signifikan."})

    p_cases = peak_day("New cases", rolling=7) or peak_day("New cases", rolling=0)
    if p_cases is not None:
        events.append({"date": p_cases, "title": "Puncak kasus baru (7D avg)", "desc": "Puncak rata-rata 7 hari untuk New cases."})

    p_deaths = peak_day("New deaths", rolling=7) or peak_day("New deaths", rolling=0)
    if p_deaths is not None:
        events.append({"date": p_deaths, "title": "Puncak kematian baru (7D avg)", "desc": "Puncak rata-rata 7 hari untuk New deaths."})

    p_active = peak_day("Active", rolling=0)
    if p_active is not None:
        events.append({"date": p_active, "title": "Puncak kasus aktif", "desc": "Hari dengan Active tertinggi."})

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


if "Date" in day_wise.columns and day_wise["Date"].notna().any():
    all_dates = pd.to_datetime(day_wise["Date"].dropna().unique())
    min_date = pd.to_datetime(all_dates.min()).date()
    max_date = pd.to_datetime(all_dates.max()).date()
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

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Pilih halaman:",
    (
        "🏠 Overview",
        "📖 Global Timeline Story Mode",
        "🌍 Global Map",
        "📊 Country Dashboard",
        "📈 Country Comparison",
        "🗽 USA View",
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
    st.write("Ringkasan perkembangan kasus COVID-19 secara global berdasarkan data time-series.")

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
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        if "Country/Region" in country_latest.columns and "Confirmed" in country_latest.columns:
            tmp = country_latest.copy()
            tmp["Confirmed"] = pd.to_numeric(tmp["Confirmed"], errors="coerce").fillna(0)
            top10 = tmp[tmp["Confirmed"] > 0].sort_values("Confirmed", ascending=False).head(10)
            fig_top = px.bar(top10, x="Country/Region", y="Confirmed", title="🔝 10 negara dengan kasus terkonfirmasi tertinggi")
            fig_top.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)

    if "New cases" in day_wise.columns and "Date" in day_wise.columns:
        st.markdown("### 📊 Kasus baru per hari (global)")
        fig_new = px.bar(day_wise.dropna(subset=["Date"]), x="Date", y="New cases", title="New confirmed cases per day")
        st.plotly_chart(fig_new, use_container_width=True)

    st.markdown("### 🔍 Korelasi antar indikator global")
    corr_cols = [c for c in ["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered"] if c in day_wise.columns]
    if len(corr_cols) >= 2:
        corr_mat = day_wise[corr_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r", title="Matriks korelasi indikator global (harian)")
        st.plotly_chart(fig_corr, use_container_width=True)


elif page == "📖 Global Timeline Story Mode":
    st.header("📖 Global Timeline Story Mode")

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
                st.plotly_chart(fig, use_container_width=True)

                r = nearest_row_by_date(story_df, ev_date)
                if r is not None:
                    st.subheader(f"{ev['title']} — {pd.to_datetime(r['Date']).date()}")
                    st.write(ev["desc"])

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
                color_continuous_scale="Reds",
                title=f"{metric} – latest snapshot",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Tabel ringkas"):
                show_cols = [c for c in ["Country/Region", "Confirmed", "Deaths", "Recovered", "Active"] if c in map_df.columns]
                if show_cols:
                    sort_col = "Confirmed" if "Confirmed" in show_cols else show_cols[1]
                    st.dataframe(map_df[show_cols].sort_values(sort_col, ascending=False).reset_index(drop=True))
                else:
                    st.dataframe(map_df.head(50))


elif page == "📊 Country Dashboard":
    st.header("📊 Country Dashboard")

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
                st.plotly_chart(fig_pie_country, use_container_width=True)

            with col2:
                new_cols = [c for c in ["New cases", "New deaths", "New recovered"] if c in country_df.columns]
                if new_cols:
                    long_new = country_df[["Date"] + new_cols].melt(id_vars="Date", value_vars=new_cols, var_name="Jenis", value_name="Jumlah")
                    fig_new_country = px.bar(long_new, x="Date", y="Jumlah", color="Jenis", title=f"Kasus baru harian di {country}")
                    fig_new_country.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_new_country, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Data harian negara"):
                st.dataframe(country_df.reset_index(drop=True))


elif page == "📈 Country Comparison":
    st.header("📈 Country Comparison")

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
                    st.plotly_chart(fig, use_container_width=True)

                    snap_date = st.date_input("Tanggal snapshot:", value=date_range[1], min_value=date_range[0], max_value=date_range[1])
                    snap_df = compare_df[compare_df["Date"].dt.date == snap_date][["Country/Region", metric]].sort_values(metric, ascending=False)
                    st.dataframe(snap_df.reset_index(drop=True))


elif page == "🗽 USA View":
    st.header("🗽 USA County / State View")

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
                st.plotly_chart(fig, use_container_width=True)

                if "Admin2" in state_df.columns:
                    latest_state = state_df[state_df["Date"] == state_df["Date"].max()]
                    county_agg = latest_state.groupby("Admin2")[metric_cols].sum().reset_index()
                    if "Confirmed" in county_agg.columns:
                        top_counties = county_agg.sort_values("Confirmed", ascending=False).head(10)
                        fig_county = px.bar(top_counties, x="Admin2", y="Confirmed", hover_data=[c for c in ["Deaths"] if c in top_counties.columns], title=f"🔝 10 county dengan kasus tertinggi di {state}")
                        fig_county.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_county, use_container_width=True)

                with st.expander("📋 Data county mentah"):
                    show_cols = [c for c in ["Date", "Admin2", "Province_State", "Confirmed", "Deaths"] if c in usa_county.columns]
                    st.dataframe(usa_county[usa_county["Province_State"] == state][show_cols].sort_values(["Date", "Admin2"]))


elif page == "🔥 Insights & Hotspots":
    st.header("🔥 Insights & Hotspots")

    if country_latest.empty or "Country/Region" not in country_latest.columns:
        st.warning("country_wise_latest.csv tidak valid / kolom Country/Region tidak ditemukan.")
    else:
        base = country_latest.copy()
        base["_key"] = base["Country/Region"].apply(normalize_country)

        w = worldometer.copy()
        pop_col = None
        w_country_col = None

        if not w.empty:
            for cand in ["Country/Region", "Country", "country", "Country/Region "]:
                if cand in w.columns:
                    w_country_col = cand
                    break

            if w_country_col is not None:
                w["_key"] = w[w_country_col].apply(normalize_country)
                for cand in ["Population", "population", "Pop", "pop"]:
                    if cand in w.columns:
                        pop_col = cand
                        break
                if pop_col is not None:
                    w_pop = w[["_key", pop_col]].drop_duplicates("_key")
                    base = base.merge(w_pop, on="_key", how="left")

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

            if per_capita and pop_col is not None and pop_col in work.columns:
                work["_pop"] = pd.to_numeric(work[pop_col], errors="coerce")
                work[metric + " per 1M"] = (work[metric] / work["_pop"]) * 1_000_000
                per_label = metric + " per 1M"

            work["_m"] = pd.to_numeric(work[per_label], errors="coerce") if per_label in work.columns else pd.to_numeric(work[metric], errors="coerce")
            work = work.dropna(subset=["_m"])
            work = work[work["_m"] >= 0]
            top_df = work.sort_values("_m", ascending=False).head(top_n)

            fig_rank = px.bar(top_df, x="Country/Region", y="_m", title=f"Top {top_n} — {per_label}")
            fig_rank.update_layout(xaxis_tickangle=-45, yaxis_title=per_label)
            st.plotly_chart(fig_rank, use_container_width=True)

            work["Confirmed"] = pd.to_numeric(work.get("Confirmed"), errors="coerce")
            work["Deaths"] = pd.to_numeric(work.get("Deaths"), errors="coerce")

            x_col, y_col = "Confirmed", "Deaths"
            if per_capita and pop_col is not None and pop_col in work.columns:
                work["_pop"] = pd.to_numeric(work.get(pop_col), errors="coerce")
                work["Confirmed per 1M"] = (work["Confirmed"] / work["_pop"]) * 1_000_000
                work["Deaths per 1M"] = (work["Deaths"] / work["_pop"]) * 1_000_000
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
            st.plotly_chart(fig_scatter, use_container_width=True)

            with st.expander("📋 Tabel"):
                cols = ["Country/Region"]
                for c in [metric, per_label, "Deaths", "Recovered", "Active", pop_col]:
                    if c and c in top_df.columns and c not in cols:
                        cols.append(c)
                st.dataframe(top_df[cols].reset_index(drop=True))


elif page == "⏱️ Timelapse":
    st.header("⏱️ Timelapse")

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
                color_continuous_scale="Reds",
                title=f"Timelapse — {tl_metric}",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))

            try:
                if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
                    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = speed
                    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = max(int(speed * 0.6), 0)
            except Exception:
                pass

            st.plotly_chart(fig, use_container_width=True)


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

    st.write(f"Menampilkan {dataset_name} — {df.shape[0]:,} baris, {df.shape[1]} kolom.")
    st.dataframe(df.head(500))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV (subset ditampilkan)", data=csv, file_name=f"{dataset_name}.csv", mime="text/csv")


elif page == "ℹ️ About":
    st.title("ℹ️ About — COVID-19 Global Dashboard")

    st.write(
        "Aplikasi ini adalah dashboard interaktif berbasis Streamlit untuk memvisualisasikan perkembangan COVID-19 "
        "secara global, per negara, hingga level county di USA."
    )

    st.subheader("Isi Halaman")
    st.write(
        "- Overview\n"
        "- Global Timeline Story Mode\n"
        "- Global Map\n"
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
