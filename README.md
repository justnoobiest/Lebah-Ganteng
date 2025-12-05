# 🦠 COVID-19 Global Dashboard – Kelompok Lebah Ganteng

Aplikasi ini adalah dashboard interaktif berbasis **Streamlit** untuk memvisualisasikan perkembangan kasus COVID-19 di dunia, per negara, hingga level county di USA.  
Data bersumber dari dataset Kaggle: **COVID-19 Corona Virus Report** (imdevskp).

---

## 👥 Kelompok Lebah Ganteng

Aplikasi ini disusun untuk memenuhi tugas Praktikum Big Data:

- **Muhammad Dimas Sudirman** – NIM **021002404001**  
- **Ari Wahyu Patriangga** – NIM **021002404007**  
- **Lola Aritasari** – NIM **021002404004**

Repository GitHub: `https://github.com/justnoobiest/Lebah-Ganteng`  
Aplikasi Streamlit: (`https://lebah-ganteng.streamlit.app`)

---

## 📊 Fitur Utama Aplikasi

1. **🏠 Overview**
   - Ringkasan global (total confirmed, deaths, recovered, active).
   - Line chart tren global dari waktu ke waktu.
   - Bar chart **kasus baru harian** (New cases).
   - **Pie chart komposisi** kasus global (Active vs Recovered vs Deaths).
   - **Top 10 negara** dengan kasus terkonfirmasi tertinggi.
   - Heatmap **korelasi** antar indikator global (Confirmed, Deaths, Recovered, Active, dst).

2. **🌍 Global Map**
   - Peta dunia (choropleth) berdasarkan indikator yang dipilih:
     - Confirmed, Deaths, Recovered, Active
     - Deaths / 100 Cases, Recovered / 100 Cases, dst.
   - Tabel ringkas per negara.

3. **📊 Country Dashboard**
   - Pilih 1 negara, tampil:
     - Ringkasan total & perubahan harian.
     - Line chart tren waktu (Confirmed, Deaths, Recovered, Active, New cases, dll).
     - **Pie chart komposisi kasus per negara**.
     - **Bar chart kasus baru harian** per negara.
     - Tabel detail harian negara tersebut.

4. **📈 Country Comparison**
   - Bandingkan beberapa negara sekaligus.
   - Pilih metrik (Confirmed, Deaths, Recovered, Active, New cases, New deaths).
   - Pilih rentang tanggal dengan **date slider**.
   - Line chart perbandingan & tabel snapshot pada tanggal tertentu.

5. **🗽 USA View**
   - Pilih **State** di USA.
   - Line chart perkembangan kasus (Confirmed & Deaths).
   - **Top 10 county** dengan kasus tertinggi di state tersebut.
   - Tabel data mentah per county.

6. **📑 Data Explorer**
   - Eksplorasi langsung beberapa dataset:
     - `day_wise`
     - `full_grouped`
     - `country_wise_latest`
     - `worldometer_data`
     - `usa_county_wise`
     - `covid_19_clean_complete`
   - Preview data & tombol **download CSV**.

---

## 🗂️ Struktur File Penting

```text
Lebah-Ganteng/
├─ app.py                      
├─ day_wise.csv
├─ full_grouped.csv
├─ country_wise_latest.csv
├─ worldometer_data.csv
├─ usa_county_wise.csv
├─ covid_19_clean_complete.csv
├─ requirements.txt            
└─ README.md                   
