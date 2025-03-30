# Gerekli kütüphaneleri içe aktaralım
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Görsellerin kaydedileceği dizini oluştur
if not os.path.exists("images"):
    os.makedirs("images")

# Pandas'ın tüm sütunları göstermesi için ayar
pd.set_option('display.max_columns', None)

# Veri setini içe aktar
df = pd.read_csv("data/heart.csv")

# Veri setinin ilk 5 satırını görüntüle
print("Veri Setinin İlk 5 Satırı:")
print(df.head())

# Veri setinin genel bilgilerini görüntüle
print("\nVeri Seti Hakkında Genel Bilgiler:")
print(df.info())

# Veri setinin boyutlarını yazdır
print(f"\nVeri Setinin Boyutu: {df.shape}")

# Eksik veri kontrolü
print("\nEksik Veri Kontrolü:")
print(df.isnull().sum())

# Eksik veri yüzdelerini hesaplayalım
missing_values = df.isnull().sum()
missing_percent = (missing_values / df.shape[0]) * 100

# Eksik veri yüzdesi 0'dan büyük olan sütunları yazdıralım
missing_df = pd.DataFrame({'Eksik Veri Sayısı': missing_values, 'Eksik Veri (%)': missing_percent})
missing_df = missing_df[missing_df["Eksik Veri Sayısı"] > 0]

if not missing_df.empty:
    print("\nEksik veri içeren sütunlar:")
    print(missing_df)
else:
    print("\nVeri setinde eksik veri bulunmamaktadır.")

# Eksik veri içeren sütunları doldurma (örnek: ortalama ile doldurma)
df.fillna(df.mean(), inplace=True)

# Tekrar eksik veri olup olmadığını kontrol edelim
print("\nEksik Veri Kontrolü (Doldurma İşleminden Sonra):")
print(df.isnull().sum())

# Temel istatistikleri yazdır
print("\nTemel İstatistikler:")
print(df.describe())

# Korelasyon matrisini çizdir
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Özellikler Arasındaki Korelasyon")

# Grafiği kaydet
plt.savefig("images/korelasyon_matrisi.png")
plt.show()

# Yaşa göre hastalık durumunun dağılımı
plt.figure(figsize=(12, 5))
sns.histplot(df["age"], bins=20, kde=True, color="blue")
plt.xlabel("Yaş")
plt.ylabel("Frekans")
plt.title("Yaş Dağılımı")
plt.savefig("images/yas_dagilimi.png")
plt.show()
