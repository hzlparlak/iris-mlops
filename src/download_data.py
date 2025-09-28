import pandas as pd
import seaborn as sns
from pathlib import Path

def download_iris_data():
    """Seaborn'dan Iris veri setini indir ve data/raw içine kaydet"""

    # data/raw klasörünü oluştur
    raw_data_path = Path("data") / "raw"
    raw_data_path.mkdir(parents=True, exist_ok=True)

    # Iris veri setini yükle
    df = sns.load_dataset("iris")

    # CSV olarak kaydet
    file_path = raw_data_path / "iris.csv"
    df.to_csv(file_path, index=False)

    print(f"Iris veri seti indirildi: {file_path}")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")

    return df

if __name__ == "__main__":
    download_iris_data()
