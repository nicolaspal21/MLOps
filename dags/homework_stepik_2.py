import io
import json
import mlflow
from mlflow.models import infer_signature
import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import fetch_california_housing
from sqlalchemy import create_engine

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, NoReturn, Literal

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# копируем из прошлого дага
DEFAULT_ARGS = {
    "owner" : "Nicolas Pal",
    "retry" : 3, # количество перезапусков
    "retry_delay" : timedelta(minutes=1) # задержка между запусками
}


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

# TO-DO: Вписать свой ник в телеграме
NAME = "nikolai022022"

# TO-DO: Вписать свой бакет
BUCKET = "test-bucket-nicolas-1"
DATA_PATH = "nikolai022022/datasets/california_housing.pkl"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"

# TO-DO: Создать словарь моделей
models = dict(zip(["rf", "lr", "hgb"], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

# TO-DO: Заполнить своими данными: настроить владельца и политику retries.
dag = DAG(
    dag_id = "3_models_dag_home_2",   # уникальный id, можно использовать имя модели
    schedule_interval = "0 1 * * *",  # расписание: раз в день по ночам
    start_date = days_ago(2),  # в прошлом, иначе не запустится 
    catchup = False,     # чтобы не стал запускать сам себя за все прошлые даты
    tags = ["mlops"],   # уникальный 
    default_args = DEFAULT_ARGS)   

   
def init(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать start_tiemstamp, run_id, experiment_name, experiment_id.
    metrics = {}
    metrics["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    exp_name = "parent_run_home_14"
    
    #client = mlflow.tracking.MlflowClient()
    
    try:
        experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"s3://{BUCKET}/mlflow/{exp_name}")
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    mlflow.set_experiment(exp_name)
    
    return metrics

# функция чтения из postgres и загрузки на s3
def get_data_from_postgres(**kwargs) -> Dict[str, Any]:

    # TO-DO 1 metrics: В этом шаге собрать data_download_start, data_download_end.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")
    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2 connections: Создать коннекторы.
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()

    # TO-DO 3 Postgres: Прочитать данные.
    data = pd.read_sql_query("SELECT * FROM california_housing", con)

    # TO-DO 4 Postgres: Сохранить данные на S3 в формате pickle в папку {NAME}/datasets/.
    s3_hook = S3Hook("aws_default")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    # Сохранить файл в формате pkl на S3
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)

    _LOG.info("Data download finished.")

    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics 

def prepare_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "get_data")

    # TO-DO 1 metrics: В этом шаге собрать data_preparation_start, data_preparation_end.
    metrics["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    
    # TO-DO 2 connections: Создать коннекторы.
    s3_hook = S3Hook("aws_default")
    # TO-DO 3 S3: Прочитать данные с S3.
    file = s3_hook.download_file(key=DATA_PATH, bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # TO-DO 4 Сделать препроцессинг.
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    # TO-DO 5 Разделить данные на train/test.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # TO-DO 6 Подготовить 4 обработанных датасета.
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Сохранить готовые данные на S3
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET,
                        f"nikolai022022/datasets/{name}.pkl").put(Body=pickle_byte_obj)

    _LOG.info("Data preparation finished.")
    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics 


def train_mlflow_model(model: Any, m_name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series, metrics) -> None:

    # TO-DO 1: Обучить модель.
    model = models[m_name]

     # Создаем копию массива с флагом WRITEABLE=True
    X_train = X_train.copy()
    y_train = y_train.copy()
    
    metrics[f"{m_name}_train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    model.fit(X_train, y_train)

    # TO-DO 2: Сделать predict.
    prediction = model.predict(X_test)
    metrics[f"{m_name}_train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    y_test = y_test.copy()
    # метрики для записи в airflow
    metrics[f"{m_name}_R^2_score"] = r2_score(y_test, prediction)
    metrics[f"{m_name}_rmse"] = mean_squared_error(y_test, prediction)**0.5
    metrics[f"{m_name}_mae"] = median_absolute_error(y_test, prediction)

    # TO-DO 3: Сохранить результаты обучения с помощью MLFlow.
    # Получить описание данных
    signature = infer_signature(X_test, prediction)
    # Сохранить модель в артифактори
    model_info = mlflow.sklearn.log_model(model, m_name, signature=signature)
    # Сохранить метрики модели
    mlflow.evaluate(
        model_info.model_uri,
        data=X_test,
        targets=y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )

    
# функция подтягивает нужную модельку, имя модели берет из словаря моделей
def train_model(**kwargs) -> Dict[str, Any]:
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "prepare_data")
    m_name = kwargs["model_name"] 

    # Получение experiment_id из XCom
    experiment_id = ti.xcom_pull(task_ids="init", key="experiment_id")
    
    # считываем данные из s3
    #s3_hook = S3Hook("s3_connection")
    s3_hook = S3Hook("aws_default")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f'dataset/{name}.pkl', bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    _LOG.info("Данные загружены")

    with mlflow.start_run(run_name="parent_run_1", experiment_id=experiment_id, description = "parent_1") as parent_run:
        for m_name in models.keys():
            with mlflow.start_run(run_name=m_name, experiment_id=experiment_id, nested=True) as child_run:
                train_mlflow_model(models[m_name], m_name, data["X_train"], data["X_test"], data["y_train"], data["y_test"],metrics)

    return metrics 

    
def save_results(**kwargs) -> None:

    ti = kwargs["ti"]
    models_metrics = ti.xcom_pull(task_ids=["train_rf", "train_lr", "train_hgb"])
    result = {}
    for model_metrics in models_metrics:
        result.update(model_metrics)

    # TO-DO 1 metrics: В этом шаге собрать end_timestamp.
    result["end_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    date = datetime.now().strftime("%Y_%m_%d_%H")

    # TO-DO 2: сохранить результаты обучения на S3
    s3_hook = S3Hook("aws_default")
    
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    # сохраним в формате json на s3 в бакет в папку results
    json_byte_object = json.dumps(result)
    resource.Object(
        BUCKET,
        f"nikolai022022/results/metrics_{date}.json").put(Body=json_byte_object)

 


#task_init = PythonOperator(task_id="init", python_callable=get_data_from_postgres, dag=dag, provide_context=True)
task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

#task_get_data = PythonOperator(task_id="get_data", python_callable=init, dag=dag, provide_context=True)
task_get_data = PythonOperator(task_id="get_data",
                               python_callable=get_data_from_postgres,
                               dag=dag,
                               provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data",
                                   python_callable=prepare_data,
                                   dag=dag,
                                   provide_context=True)
    
task_train_models = [
    PythonOperator(task_id=f"train_{model_name}",
                   python_callable=train_model,
                   dag=dag,
                   provide_context=True,
                   op_kwargs={"model_name": model_name})
    for model_name in models.keys()
]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

        
# архитектура дага
task_init >> task_get_data >> task_prepare_data >> task_train_models >> task_save_results




