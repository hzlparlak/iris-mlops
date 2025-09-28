import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

def train_model(model_type='random_forest'):
    """Iris veri seti için model eğit ve kaydet"""
    
    # İşlenmiş veriyi yükle
    df = pd.read_csv('data/processed/iris_processed.csv')
    
    # X ve y'yi ayır
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols]
    y = df['species']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model seç
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    else:  # logistic_regression
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    
    # Model eğit
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    
    # Model kaydet
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_iris_model.pkl'
    joblib.dump(model, model_path)
    
    # Metrikleri kaydet
    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    with open('models/iris_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Özellik listesini kaydet
    with open('models/iris_features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"✅ Model eğitildi: {model_type}")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"💾 Model kaydedildi: {model_path}")
    
    # Detaylı rapor
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, metrics

if __name__ == "__main__":
    # Random Forest modeli eğit
    model_rf, metrics_rf = train_model('random_forest')
    
    # Logistic Regression modeli eğit
    model_lr, metrics_lr = train_model('logistic_regression')
    
    print("\n🎯 Her iki model de eğitildi ve kaydedildi!")
