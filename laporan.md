# UAS DATA MINING

## Tugas Proyek Akhir: Preprocessing dan Unsupervised Learning (Clustering)

**Nama:** Lintang Maulana Yusuf  
**NIM:** 241011401753

---

# 1. Pemilihan Dataset

Dataset yang digunakan adalah **E-Commerce Data** yang diperoleh dari Kaggle.

Sumber Dataset:
https://www.kaggle.com/datasets/carrie1/ecommerce-data

Dataset ini berisi data transaksi penjualan pada sebuah toko online dengan atribut seperti **InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID,** dan **Country**. Dataset dipilih karena sesuai untuk proses clustering dan memiliki data yang dapat dilakukan tahap preprocessing.

---

# 2. Tahap Cleaning Data

## A. Penanganan Missing Value

Langkah pertama adalah melakukan pengecekan missing value pada setiap kolom menggunakan fungsi `isnull().sum()`.

```python
df.isnull().sum()
```

Selanjutnya dilakukan penghapusan data yang memiliki nilai kosong menggunakan fungsi `dropna()` pada kolom **CustomerID**.

```python
df = df.dropna(subset=["CustomerID"])
```

Kemudian dilakukan pengecekan kembali untuk memastikan tidak ada missing value.

---

## B. Penanganan Data Duplikat

Data duplikat diperiksa menggunakan fungsi `duplicated()`.

```python
df.duplicated().sum()
```

Selanjutnya data duplikat dihapus menggunakan fungsi `drop_duplicates()`.

```python
df = df.drop_duplicates()
```

Jumlah data sebelum dan sesudah proses penghapusan ditampilkan untuk memastikan hasil preprocessing.

---

## C. Scaling Data

Sebelum proses clustering dilakukan, data numerik distandarisasi agar memiliki skala yang sama.

Kolom yang digunakan adalah:

- Quantity
- UnitPrice

Standarisasi dilakukan menggunakan **StandardScaler** dari Scikit-Learn.

```python
X = df[['Quantity','UnitPrice']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

# 3. Implementasi Clustering

Pada tugas ini digunakan algoritma **K-Means Clustering**.

Jumlah cluster ditentukan sebanyak **3 cluster**.

```python
kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(X_scaled)
```

Hasil clustering kemudian disimpan pada kolom **Cluster** sehingga setiap data memiliki label kelompok.

---

# 4. Evaluasi dan Visualisasi

## A. Visualisasi Hasil Clustering

Hasil clustering divisualisasikan menggunakan **Scatter Plot 2D** dengan atribut **Quantity** sebagai sumbu X dan **UnitPrice** sebagai sumbu Y. Warna yang berbeda menunjukkan kelompok (cluster) yang berbeda.

Visualisasi ini membantu melihat penyebaran data berdasarkan hasil pengelompokan K-Means.

---

## B. Evaluasi Menggunakan Elbow Method

Evaluasi dilakukan menggunakan **Elbow Method**.

Metode ini digunakan untuk membantu menentukan jumlah cluster yang optimal berdasarkan nilai **WCSS (Within Cluster Sum of Squares)**.

Grafik Elbow menunjukkan perubahan nilai WCSS pada setiap jumlah cluster sehingga dapat digunakan sebagai dasar dalam menentukan jumlah cluster yang digunakan pada proses K-Means.

---

# 5. Kesimpulan

Berdasarkan proses yang telah dilakukan, dataset E-Commerce berhasil melalui tahapan preprocessing yang meliputi penanganan missing value, penghapusan data duplikat, dan standarisasi data numerik.

Selanjutnya dilakukan proses clustering menggunakan algoritma **K-Means** dengan **3 cluster**. Hasil clustering berhasil divisualisasikan menggunakan scatter plot dan dievaluasi menggunakan Elbow Method sehingga proses pengelompokan data dapat dilakukan dengan baik.