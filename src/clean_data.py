import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def clean_iris_data():
    # Yol ayarları
    input_path = os.path.join("data", "raw", "iris.csv")
    output_path = os.path.join("data", "processed", "iris_processed.csv")

    # Veriyi yükle
    df = pd.read_csv(input_path)
    df_clean = df.copy()
    
    # Eksik değer kontrolü
    if df_clean.isnull().sum().sum() > 0:
        print("⚠️ Eksik değerler bulundu, dolduruluyor...")
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Kategorik değişken encode
    if "species" in df_clean.columns:
        le_species = LabelEncoder()
        df_clean["species_encoded"] = le_species.fit_transform(df_clean["species"])
    
    # Yeni özellik: taç yaprak oranı
    if "petal_length" in df_clean.columns and "petal_width" in df_clean.columns:
        df_clean["petal_ratio"] = df_clean["petal_length"] / df_clean["petal_width"]
    
    # Çıktı dizinini oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Temizlenmiş veriyi kaydet
    df_clean.to_csv(output_path, index=False)
    
    print(" Iris verisi temizlendi ve kaydedildi:", output_path)
    print(f"Orijinal boyut: {df.shape}")
    print(f"Temizlenmiş boyut: {df_clean.shape}")
    print(f"Eksik değerler: {df_clean.isnull().sum().sum()} toplam eksik değer")
    
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width", "petal_ratio", "species_encoded"]
    print(f"Model özellikleri: {features}")
    
    return df_clean, features

if __name__ == "__main__":
    clean_iris_data()
