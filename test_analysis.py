import pandas as pd
from analysis import load_heart, run_logistic_model

def test_load_heart_ok():
    df = load_heart()
    # Verifica que tenga las columnas esperadas
    expected_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalch", "exang", "oldpeak", "num"
    ]
    assert list(df.columns) == expected_cols
    # Verifica que tenga filas
    assert len(df) > 0


def test_run_logistic_model_basic():
    # Prepara dataset pequeño
    df = pd.DataFrame({
        "age": [40, 50, 60, 70],
        "sex_male": [1, 0, 1, 0],
        "disease": [0, 1, 1, 0]
    })

    res = run_logistic_model(df, ["age", "sex_male"], target="disease", test_size=0.5, random_state=42)

    # Checks básicos
    assert isinstance(res, dict)
    assert "Accuracy" in res
    assert "Sensitivity" in res
    assert "Specificity" in res
    # Métricas entre 0 y 1
    for k in ["Accuracy", "Sensitivity", "Specificity"]:
        assert 0 <= res[k] <= 1

def test_end_to_end_pipeline():
    # 1. Cargar datos reales
    df = load_heart()
    df["disease"] = (df["num"] > 0).astype(int)

    # 2. Preprocesar con get_dummies
    df_d = pd.get_dummies(
        df,
        columns=["sex", "cp", "restecg", "fbs", "exang"],
        drop_first=True
    )

    # Normalizar columnas para evitar problemas de mayúsculas/minúsculas
    df_d.columns = [c.lower() for c in df_d.columns]

    # 3. Variables para el modelo (usa la dummy de sexo + edad)
    sex_cols = [c for c in df_d.columns if c.startswith("sex_")]
    subset = ["age"] + sex_cols[:1]  # age + una dummy de sex

    # 4. Ejecutar modelo completo
    res = run_logistic_model(df_d, subset, target="disease", test_size=0.2, random_state=123)

    # 5. Validaciones
    assert isinstance(res, dict)
    for k in ["Accuracy", "Sensitivity", "Specificity"]:
        assert k in res and 0 <= res[k] <= 1