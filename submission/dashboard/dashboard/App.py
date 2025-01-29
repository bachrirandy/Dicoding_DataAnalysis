import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Submission Project Analisis Data dengan Python by DICODING')
st.write("Mochammad Randy Surya Bachri")

st.header("About Dataset")
st.write("Dataset Links: [E-Commerce Public Dataset](https://drive.google.com/file/d/1MsAjPM7oKtVfJL_wRp1qmCajtSG1mdcK/view)")


st.header("Question")
st.write("""
         1. What are the most frequently purchased products and from what categories?
         2. What is the customer review score for each product category?
         3. What is the average delivery time from each seller to the customer by location?""")

st.header("1. Data Wrangling")
st.subheader("Displaying the Top 5 and Bottom 5 Rows of the Dataset")

df = pd.read_csv(r"D:\CODE\DICODING\ANALISIS DATA\submission\dashboard\final_data.csv")
data = pd.concat([df.head(5), df.tail(5)])
st.write(data)

st.header("2. Data Cleaning")
st.write(df.isnull().sum())

columns_to_keep = ['review_comment_title', 'review_comment_message']
df = df.dropna(subset=[col for col in df.columns if col not in columns_to_keep])

st.write(df.isnull().sum())

st.write("Data shape")
st.write(df.shape)

st.write("Data duplikat")
st.write(df.duplicated().sum())

st.write("""
         **Insight:**
- Data null atau kosong terdapat pada beberapa kolom dalam dataset, seperti: order_id, product_id, seller_id, shipping_limit_date, price, product_category_name, product_name_length, product_description_length, customer_id, customer_city, customer_state, payment_sequential, payment_type, dan payment_value. 

- setelah melakukan pembersihan, data berkurang dari 118310 menjadi 113216. Saya sengaja tidak menghapus null di kolom review_comment_title dan review_comment_message karena banyak orang yang memang tidak mau memberikan komentar.
         """)

st.header("3. Exploratory Data Analysis")

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)

st.write(df.describe())

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Antar Variabel")

st.pyplot(plt)

st.write("""
         **Insight:**
- Harga Produk bervariasi luas (Rp0,85 – Rp6,73 juta), rata-rata Rp119 ribu, menunjukkan adanya produk murah hingga premium.
- Skor ulasan rata-rata 4,08 (mayoritas positif)
- Rata-rata berat 2,1 kg, namun ada produk hingga 40 kg, menunjukkan variasi ukuran signifikan.
- Biaya pengiriman berkorelasi dengan harga (0.41) dan berat produk (0.61), menunjukkan produk mahal dan berat cenderung memiliki biaya kirim lebih tinggi.
         """)


st.header("4. Visualization & Explanatory Analysis")

st.subheader(" A. Pertanyaan 1: Apa saja produk yang sering dibeli dan dari kategori apa? ")

top_products = df["product_category_name_english"].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.xlabel("Jumlah Pembelian", fontsize=12)
plt.ylabel("Kategori Produk (Inggris)", fontsize=12)
plt.title("10 Kategori Produk yang Paling Sering Dibeli", fontsize=14)
plt.grid(axis="x", linestyle="--", alpha=0.7)
st.pyplot(plt)

st.write("""
         **Insight:**
- Kategori Terlaris: Produk kategori bed_bath_table paling banyak dibeli, menunjukkan kebutuhan tinggi untuk perlengkapan kamar tidur dan kamar mandi.

- Kesehatan & Kecantikan: health_beauty berada di posisi kedua, menandakan minat besar terhadap produk perawatan diri.

- Olahraga & Furnitur: sports_leisure dan furniture_decor juga populer, mencerminkan permintaan tinggi untuk gaya hidup sehat dan dekorasi rumah.

- Elektronik & Aksesori: computers_accessories cukup diminati, menunjukkan kebutuhan akan perangkat teknologi.

- Produk Lain: Kategori seperti housewares, watches_gifts, dan telephony juga memiliki pembelian tinggi, menunjukkan preferensi pelanggan yang beragam."""
         )

st.subheader(" B. Pertanyaan 2: Bagaimana skor ulasan pelanggan (review score) untuk setiap kategori produk?")

# Hitung rata-rata review score untuk setiap kategori produk
review_scores = df.groupby("product_category_name_english")["review_score"].mean().reset_index()

# Urutkan berdasarkan rata-rata skor ulasan
review_scores = review_scores.sort_values(by="review_score", ascending=False)

# Ambil 10 kategori dengan skor review tertinggi dan terendah
top_10_highest = review_scores.head(10)
top_10_lowest = review_scores.tail(10)

# Plot kategori dengan skor tertinggi
plt.figure(figsize=(12, 6))
sns.barplot(x="review_score", y="product_category_name_english", data=top_10_highest, palette="Blues_r")
plt.xlabel("Rata-rata Skor Ulasan")
plt.ylabel("Kategori Produk")
plt.title("10 Kategori Produk dengan Skor Ulasan Tertinggi")
plt.xlim(3, 5)  # Skor ulasan berkisar antara 1-5
plt.grid(axis="x", linestyle="--", alpha=0.7)
st.pyplot(plt)

# Plot kategori dengan skor terendah
plt.figure(figsize=(12, 6))
sns.barplot(x="review_score", y="product_category_name_english", data=top_10_lowest, palette="Reds_r")
plt.xlabel("Rata-rata Skor Ulasan")
plt.ylabel("Kategori Produk")
plt.title("10 Kategori Produk dengan Skor Ulasan Terendah")
plt.xlim(1, 5)
plt.grid(axis="x", linestyle="--", alpha=0.7)
st.pyplot(plt)

st.write("""
         **Insight:**
- Produk dengan Ulasan Tertinggi: Kategori fashion_childrens_clothes, cds_dvds_musicals, dan books_imported memiliki skor ulasan tertinggi, menunjukkan kepuasan pelanggan yang tinggi terhadap produk ini.

- Produk dengan Ulasan Terendah: Kategori furniture_mattress_and_upholstery, audio, dan fixed_telephony mendapat skor ulasan rendah, mungkin karena masalah kualitas, ketahanan, atau ekspektasi pelanggan yang tidak terpenuhi.

- Tren Umum: Produk yang berhubungan dengan hiburan dan buku cenderung mendapatkan ulasan positif, sementara produk elektronik dan perabotan memiliki ulasan lebih kritis.
         """)

st.subheader("C. Pertanyaan 3: Berapa rata-rata waktu pengiriman dari setiap seller ke pelanggan berdasarkan lokasi?")

# Konversi tanggal ke format datetime
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], dayfirst=True)
df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])

# Hitung waktu pengiriman dalam hari
df["delivery_time"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days

# Hapus nilai yang tidak valid (misalnya pesanan yang belum dikirim)
df = df.dropna(subset=["delivery_time"])

# Hitung rata-rata waktu pengiriman berdasarkan lokasi pelanggan
delivery_times = df.groupby("customer_state")["delivery_time"].mean().reset_index()
delivery_times = delivery_times.sort_values("delivery_time") 

plt.figure(figsize=(12, 6))
sns.barplot(data=delivery_times, x="customer_state", y="delivery_time", palette="viridis")

# Tambahkan label
plt.title("Rata-rata Waktu Pengiriman ke Setiap Provinsi Pelanggan")
plt.xlabel("Provinsi Pelanggan")
plt.ylabel("Rata-rata Waktu Pengiriman (hari)")
plt.xticks(rotation=45)
st.pyplot(plt)

st.write("""
         **Insight:**
- Rata-rata waktu pengiriman bervariasi antar provinsi. Provinsi seperti São Paulo (SP) memiliki waktu pengiriman tercepat, sedangkan Roraima (RR) memiliki waktu pengiriman paling lama.
         """)

st.header("Conclusion")
st.write("""
         - Pertanyaan 1:
Produk kategori bed_bath_table menjadi yang paling laris, menunjukkan kebutuhan tinggi untuk perlengkapan rumah tangga. Minat terhadap health_beauty menandakan pentingnya perawatan diri bagi konsumen. Popularitas sports_leisure dan furniture_decor mencerminkan tren gaya hidup sehat dan dekorasi rumah. Permintaan tinggi pada computers_accessories menunjukkan ketergantungan konsumen pada perangkat teknologi. Selain itu, kategori seperti housewares, watches_gifts, dan telephony menunjukkan preferensi pelanggan yang beragam dalam berbelanja.

- Pertanyaan 2:
Produk seperti fashion_childrens_clothes, cds_dvds_musicals, dan books_imported mendapatkan ulasan tertinggi, menunjukkan tingkat kepuasan pelanggan yang tinggi. Sebaliknya, kategori furniture_mattress_and_upholstery, audio, dan fixed_telephony memiliki skor ulasan rendah, kemungkinan karena masalah kualitas atau ekspektasi yang tidak terpenuhi. Secara umum, produk terkait hiburan dan buku lebih disukai pelanggan, sedangkan elektronik dan perabotan cenderung mendapat ulasan lebih kritis.

- Pernyataan 3:
Waktu pengiriman pesanan berbeda-beda di setiap provinsi. São Paulo (SP) memiliki waktu pengiriman tercepat, kemungkinan karena infrastruktur logistik yang lebih baik dan kedekatan dengan pusat distribusi. Sebaliknya, Roraima (RR) memiliki waktu pengiriman paling lama, yang mungkin disebabkan oleh jarak geografis yang lebih jauh dan keterbatasan akses transportasi. Hal ini menunjukkan bahwa faktor lokasi berpengaruh signifikan terhadap efisiensi pengiriman.
         """)
