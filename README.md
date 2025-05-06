# User Manual

## Buka Tampilan UI

Untuk membuka aplikasi, buka link di bawah atau buka file .exe

```bash
https://konsumsi-bbm.streamlit.app/
```

---

## Total Konsumsi BBM

1. Masukkan GDP tahun sebelumnya.
   - Format ribuan dipisahkan menggunakan `.` (titik) atau tanpa pemisah.
2. Masukkan angka pertumbuhan ekonomi dan capita untuk tahun yang akan diprediksi.
   - Format desimal dipisahkan menggunakan `,` (koma).
3. Klik enter untuk menampilkan hasil.

---

## Total Komsumsi BBM Per Jenis dan Kebijakan

1. Upload file CSV
   - Klik "Browse files" dan pilih file CSV
   - Pastikan file yang diunggah berformat CSV dan memiliki nama kolom dengan huruf kapital:
     - `JENIS KEBIJAKAN` seperti JBU, JBT, JBKP
     - `JENIS BBM` seperti BENSIN 90, SOLAR 48, KEROSENE, dan lain - lain.
     - `KONSUMSI BBM` berisi jumlah konsumsi BBM dalam KL.
   - Jika belum memiliki file sesuai format, bisa menyesuaikan dengan mendownload template CSV yang ada.
2. Pilih jenis BBM dengan checkbox “Pilih Semua Jenis BBM” untuk memilih semua, atau pilih manual via multiselect.

---

## Hasil Perhitungan

1. Dashboard akan menampilkan hasil perhitungan total konsumsi BBM, proporsi BBM per jenis dan, proporsi per kebijakan
2. Hasil perhitungan dapat diexport ke dalam bentuk CSV melalui tombol download yang tersedia.

---
