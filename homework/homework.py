# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import pandas as pd
import gzip
import pickle
import json
import zipfile
from glob import glob
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Leer los archivos comprimidos y cargarlos en DataFrames
def load_zip_data(directory: str) -> list[pd.DataFrame]:
    data_frames = []
    for zip_file in sorted(glob(os.path.join(directory, "*.zip"))):
        with zipfile.ZipFile(zip_file, "r") as zf:
            for file_name in zf.namelist():
                with zf.open(file_name) as file:
                    df = pd.read_csv(file, sep=",", index_col=0)
                    data_frames.append(df)
    return data_frames


# Preprocesamiento y limpieza de los datos
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    df.dropna(inplace=True)
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v > 4 else v)
    df["EDUCATION"] = df["EDUCATION"].astype(str)
    
    return df


# Función principal que entrena el modelo y guarda los resultados
def execute_model_training():
    data_frames = load_zip_data("files/input")
    train_df, test_df = data_frames  # Suponemos que hay dos archivos de entrada

    # Limpiar los datos
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Separar variables predictoras y variable objetivo
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]
    
    # Crear el pipeline con OneHotEncoder y RandomForestClassifier
    categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[('categorical', onehot_encoder, categorical_columns)],
        remainder="passthrough"
    )
    
    rf_classifier = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=[('preprocessing', preprocessor), ('classifier', rf_classifier)])
    
    # Definir el espacio de hiperparámetros a optimizar
    param_grid = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True, verbose=2)
    grid_search.fit(X_train, y_train)

    # Guardar el modelo entrenado
    model_output_path = 'files/models/model.pkl.gz'
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with gzip.open(model_output_path, 'wb') as f:
        pickle.dump(grid_search, f)

    # Generar predicciones
    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)

    # Calcular métricas para el conjunto de entrenamiento y prueba
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, zero_division=0),
        "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
    }

    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
    }

    # Calcular las matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Crear un diccionario para las matrices de confusión
    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": cm_train[0, 0], "predicted_1": cm_train[0, 1]},
        "true_1": {"predicted_0": cm_train[1, 0], "predicted_1": cm_train[1, 1]},
    }

    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": cm_test[0, 0], "predicted_1": cm_test[0, 1]},
        "true_1": {"predicted_0": cm_test[1, 0], "predicted_1": cm_test[1, 1]},
    }

    # Guardar las métricas y matrices de confusión en un único archivo JSON
    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(train_metrics, f, ensure_ascii=False)
        f.write("\n")  # Línea nueva entre registros
        json.dump(test_metrics, f, ensure_ascii=False)
        f.write("\n")
        json.dump(cm_train_dict, f, ensure_ascii=False)
        f.write("\n")
        json.dump(cm_test_dict, f, ensure_ascii=False)

    print("Proceso completado y archivos guardados.")


# Ejecutar la función
if __name__ == "__main__":
    execute_model_training()
