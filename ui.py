import streamlit as st
import pandas as pd
import pickle
from lib_regresi import gdpc, konsumsi_per_tahun, total_proporsi, proporsi_jbkp_jbt, proporsi_jbu
import numpy as np
from collections import defaultdict

with open('model.pkl', 'rb') as f:
    model_params = pickle.load(f)

slope = model_params['slope']
intercept = model_params['intercept']

st.set_page_config(page_title="Proyeksi Konsumsi BBM", page_icon=":bar_chart:", layout="wide")

# ---- SIDEBAR ----

###### ---- TOTAL KONSUMSI BBM ----
st.sidebar.header("Total Konsumsi BBM")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.number_input("Tahun Awal Prediksi", min_value=2024, max_value=2050, key="start_year")
with col2:
    end_year = st.number_input("Tahun Akhir Prediksi", min_value=2024, max_value=2050, key="end_year")
if start_year > end_year:
    st.sidebar.warning("Tahun awal harus lebih kecil dari tahun akhir.")

gdp_text = st.sidebar.text_input(f"GDP Tahun {start_year - 1} (Miliar Rupiah)", value="0", key="gdp_text")
gdp_before = {}
if gdp_text:
    try:
        gdp_before = float(gdp_text.replace(".", ""))
    except ValueError:
        st.sidebar.warning("Masukkan nilai yang valid untuk GDP.")
        gdp_before = 0
else:
    gdp_before = 0

angka_pertumbuhan_list = []
capita_list = []

for i in range(start_year, end_year + 1):
    ape1, c1, gdpc1 = st.sidebar.columns(3)
    with ape1:
        angka_text = st.sidebar.text_input(f"Angka Pertumbuhan Ekonomi Tahun {i} (%)", value="0", key=f"angka_text_{i}")
        angka_pertumbuhan = {}
        if angka_text:
            try:
                angka_pertumbuhan = float(angka_text.replace(".", "").replace(",", "."))
            except ValueError:
                st.sidebar.warning("Masukkan nilai yang valid untuk angka pertumbuhan.")
                angka_pertumbuhan = 0.0
        else:
            angka_pertumbuhan = 0.0
        angka_pertumbuhan_list.append(float(angka_pertumbuhan))
    with c1:
        capita_text = st.sidebar.text_input(f"Capita Tahun {i} (Juta Jiwa)", value="0", key=f"capital_text_{i}")
        capita = {}
        if capita_text:
            try:
                capita = float(capita_text.replace(".", "").replace(",", "."))
            except ValueError:
                st.sidebar.warning("Masukkan nilai yang valid untuk capita.")
                capita = 0.0
        else:
            capita = 0.0
        capita_list.append(float(capita))
    with gdpc1:
        gdp_list, gdp_per_capita_list = gdpc(gdp_before, start_year, angka_pertumbuhan_list, capita_list)

###### ---- PROYEKSI BBM PER PRODUK ----

st.sidebar.header("Proyeksi BBM Per Jenis:")  

template_df = pd.DataFrame({
    "JENIS KEBIJAKAN": [],
    "JENIS BBM": [],
    "KONSUMSI BBM": []
})

csv = template_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Template CSV",
    data=csv,
    file_name='template.csv',
    mime='text/csv'
)

uploaded_file = st.sidebar.file_uploader("Upload File CSV", type=["csv"])

product = pd.DataFrame()
jenis_bbm = []
filtered_product = pd.DataFrame()
proporsi_list = []
konsumsi_bbm = {}
bbm = []
proporsi = []

if uploaded_file is None:
    st.sidebar.warning("Silakan upload file CSV untuk melanjutkan.")
else:
    product = pd.read_csv(uploaded_file)

    bbm = product['JENIS BBM']
    proporsi = product['KONSUMSI BBM']

    all_options = product['JENIS BBM'].unique().tolist()
    pilih_semua = st.sidebar.checkbox("Pilih Semua Jenis BBM")
    if pilih_semua:
        jenis_bbm = st.sidebar.multiselect("Jenis BBM", options=all_options, default=all_options)
    else:
        jenis_bbm = st.sidebar.multiselect("Jenis BBM", options=all_options)
    
    konsumsi_bbm = np.sum([float(str(i).replace(",", "")) for i in product['KONSUMSI BBM'] if i != "0"])
    st.sidebar.markdown(f"Total Konsumsi BBM Tahun {start_year - 1}:<br> {konsumsi_bbm:,.0f}".replace(",", ".") + " KL", unsafe_allow_html=True)
    
    filtered_product = product[product['JENIS BBM'].isin(jenis_bbm)]
    proporsi_list = [float(str(i).replace(",", "")) for i in filtered_product['KONSUMSI BBM']]


# ---- MAINPAGE ----
st.title(":bar_chart: Proyeksi Konsumsi BBM")
st.markdown("###")

###### ---- TOTAL KONSUMSI BBM ----
st.subheader("Total Konsumsi BBM (KL)")

konsumsi_per_tahun = konsumsi_per_tahun(start_year, end_year, gdp_per_capita_list, intercept, slope)

tahun_list = list(range(start_year, end_year + 1))
total = pd.DataFrame(list(zip(tahun_list, gdp_per_capita_list, konsumsi_per_tahun)), columns=["Tahun", "GDP/C", "Total Konsumsi BBM"])
total["Total Konsumsi BBM"] = total["Total Konsumsi BBM"].apply(lambda x: f"{x:,.0f}".replace(",", "."))
total["Tahun"] = total["Tahun"].astype(str)
total["GDP/C"] = total["GDP/C"].apply(lambda x: f"{x:,.0f}".replace(",", "."))
st.markdown(total.style.hide(axis="index").to_html(), unsafe_allow_html=True)

###### ---- PROYEKSI PER PRODUK ----
def format_angka(df):
    df_formatted = df.copy()
    for col in df_formatted.columns:
        if col != "Tahun" and pd.api.types.is_numeric_dtype(df_formatted[col]):
            df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    return df_formatted

grouped_proporsi = defaultdict(float)
for jenis, value in zip(bbm, proporsi):
    clean_value = float(str(value).replace(",", ""))
    grouped_proporsi[jenis] += clean_value

filtered_grouped_proporsi = {k: v for k, v in grouped_proporsi.items() if k in jenis_bbm}

final_total_proporsi = total_proporsi(konsumsi_bbm, konsumsi_per_tahun, filtered_grouped_proporsi, tahun_list)

df_proporsi = pd.DataFrame(final_total_proporsi)
df_proporsi_fmt = format_angka(df_proporsi)
st.subheader("Proyeksi Konsumsi BBM Per Jenis (KL)")
st.markdown(df_proporsi_fmt.style.hide(axis="index").to_html(), unsafe_allow_html=True)


###### ---- PROYEKSI PER KEBIJAKAN JBT JBKP----

st.subheader("Proyeksi Konsumsi JBT JBKP (KL)")

hasil_jbkp = []
if uploaded_file is None:
    st.warning("Silakan upload file CSV untuk melanjutkan.")
else:
    filtered_product = product[
    (product['JENIS KEBIJAKAN'].isin(["JBT", "JBKP", "jbt", "jbkp"])) &
    (product['JENIS BBM'].isin(jenis_bbm))
    ]   

    filtered_product["KONSUMSI BBM"] = (filtered_product["KONSUMSI BBM"].astype(str).str.replace(",", "").astype(float))

    hasil_jbkp = proporsi_jbkp_jbt(grouped_proporsi,filtered_product,final_total_proporsi,jenis_bbm,tahun_list)


df_proporsi_jbkp = pd.DataFrame(hasil_jbkp)
df_proporsi_jbkp_fmt = format_angka(df_proporsi_jbkp)
st.markdown(df_proporsi_jbkp_fmt.style.hide(axis="index").to_html(), unsafe_allow_html=True)

###### ---- PROYEKSI PER KEBIJAKAN JBU----
st.subheader("Proyeksi Konsumsi JBU (KL)")

kolom_urut = df_proporsi.columns.tolist()

hasil_jbu = proporsi_jbu(final_total_proporsi, hasil_jbkp, jenis_bbm, tahun_list)
df_proporsi_jbu = pd.DataFrame(hasil_jbu)
df_proporsi_jbu = df_proporsi_jbu[[col for col in kolom_urut if col in df_proporsi_jbu.columns]]
df_proporsi_jbu_fmt = format_angka(df_proporsi_jbu)
st.markdown(df_proporsi_jbu_fmt.style.hide(axis="index").to_html(), unsafe_allow_html=True)

# Create labeled sections:
def convert_3_df_to_one_csv(dfs, labels):
    csv_parts = []
    for df, label in zip(dfs, labels):
        csv_parts.append(f"{label}\n")
        csv_parts.append(df.to_csv(index=False))
        csv_parts.append("\n")
    return ''.join(csv_parts).encode('utf-8')

csv_combined = convert_3_df_to_one_csv(
    [total, df_proporsi_fmt, df_proporsi_jbkp_fmt, df_proporsi_jbu_fmt],
    ["Proyeksi Total Konsumsi BBM", "Proyeksi Konsumsi BBM Per Jenis", "Proyeksi Konsumsi JBT JBKP", "Proyeksi Konsumsi JBU"]
)

# Download button
st.download_button(
    label="Download CSV",
    data=csv_combined,
    file_name='proyeksi.csv',
    mime='text/csv'
)
