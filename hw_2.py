import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing

BUCKET = "test-bucket-nicolas-1"

# Прочитаем данные.
housing = fetch_california_housing(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

# Создать новый эксперимент
exp_name = "NicolasPal"

try:
    experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"s3://{BUCKET}/mlflow/{exp_name}")
    
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

mlflow.set_experiment(exp_name)

models = dict(zip(["RandomForest", "LinearRegression", "HistGB"], 
                  [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

# Создадим parent run.
with mlflow.start_run(run_name="@nikolai022022", experiment_id = experiment_id, description = "parent") as parent_run:
    for model_name in models.keys():
        # Запустим child run на каждую модель.
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:
            model = models[model_name]
            
            # Обучим модель.
            model.fit(pd.DataFrame(X_train), y_train)
        
            # Сделаем предсказание.
            prediction = model.predict(X_val)
        
            # Создадим валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val
        
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "logreg", signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )
