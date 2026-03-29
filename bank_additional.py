import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, List, Optional

def clean_and_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Очищення даних на основі EDA."""
    df = df.copy()
    
    # 1. Обробка pdays (створення нової ознаки)
    df['was_contacted'] = np.where(df['pdays'] == 999, 0, 1)
    
    # 2. Обробка викидів у campaign (Capping за 95-м перцентилем)
    upper_limit = df['campaign'].quantile(0.95)
    df['campaign'] = np.where(df['campaign'] > upper_limit, upper_limit, df['campaign'])
    
    # 3. Заповнення unknown модою
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].replace('unknown', df[col].mode()[0])
    
    return df

def get_features_and_targets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Відокремлює ознаки від цільової змінної та видаляє мультиколінеарні колонки."""
    X = df.drop(columns=[target_col, 'nr.employed', 'emp.var.rate'], errors='ignore')
    y = df[target_col]
    return X, y

def get_train_val_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """Розділяє дані на три частини: Train (70%), Val (15%), Test (15%)."""
    # Спочатку відділяємо 15% на фінальний тест
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Решту (85%) ділимо на Train та Val (15% від загалу це ~17.6% від залишку)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_numeric_features(train_df, val_df, test_df, numeric_cols):
    """Масштабує числові ознаки."""
    scaler = MinMaxScaler().fit(train_df[numeric_cols])
    
    for df in [train_df, val_df, test_df]:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        
    return train_df, val_df, test_df, scaler

def encode_categorical_features(train_df, val_df, test_df, categorical_cols):
    """Кодує категоріальні ознаки."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    def transform_df(df):
        encoded = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoded_cols, index=df.index)
        return pd.concat([df.drop(columns=categorical_cols), encoded], axis=1)
    
    return transform_df(train_df), transform_df(val_df), transform_df(test_df), encoder

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = True):
    """Повний цикл обробки: повертає Train, Val та Test вибірки."""
    target_col = 'y'
    cleaned_df = clean_and_feature_engineering(raw_df)
    X, y = get_features_and_targets(cleaned_df, target_col)
    
    # 1. Спліт на 3 частини
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_split(X, y)
    
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes('object').columns.tolist()
    
    # 2. Кодування
    X_train, X_val, X_test, encoder = encode_categorical_features(X_train, X_val, X_test, categorical_cols)
    
    # 3. Масштабування
    scaler = None
    if scaler_numeric:
        X_train, X_val, X_test, scaler = scale_numeric_features(X_train, X_val, X_test, numeric_cols)
        
    input_cols = X_train.columns.tolist()
    return X_train, y_train, X_val, y_val, X_test, y_test, input_cols, scaler, encoder