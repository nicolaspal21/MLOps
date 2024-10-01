import io
import json
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

BUCKET = "test-bucket-nicolas-1"
DATA_PATH = "datasets/california_housing.pkl"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"

models = dict(zip(["rf", "lr", "hgb"], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

dag = DAG(
    dag_id = "3_models_dag",   # уникальный id, можно использовать имя модели
    schedule_interval = "0 1 * * *",  # расписание: раз в день по ночам
    start_date = days_ago(2),  # в прошлом, иначе не запустится 
    catchup = False,     # чтобы не стал запускать сам себя за все прошлые даты
    tags = ["mlops"],   # уникальный 
    default_args = DEFAULT_ARGS)   

   
def init() -> Dict[str, Any]:
    metrics = {}
    metrics["start_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    return metrics 

# функция чтения из postgres и загрузки на s3
def get_data_from_postgres(**kwargs) -> Dict[str, Any]:

    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")
    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # Использовать созданный ранее PG connection
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()

    # Прочитать все данные из таблицы california_housing
    data = pd.read_sql_query("SELECT * FROM california_housing", con)

    # сохраняем результаты обучения на s3
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
    
    # скачиваем данные с s3
    s3_hook = S3Hook("aws_default")
    file = s3_hook.download_file(key=DATA_PATH, bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # Сделать препроцессинг
    # Разделить на фичи и таргет
    X, y = data[FEATURES], data[TARGET]

    # Разделить данные на обучение и тест
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Обучить стандартизатор на train
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
                        f"datasets/{name}.pkl").put(Body=pickle_byte_obj)

    _LOG.info("Data preparation finished.")
    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    return metrics 

# функция подтягивает нужную модельку, имя модели берет из словаря моделей
def train_model(**kwargs) -> Dict[str, Any]:
    
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids = "prepare_data")
    m_name = kwargs["model_name"] 

    # считываем данные из s3
    #s3_hook = S3Hook("s3_connection")
    s3_hook = S3Hook("aws_default")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f'dataset/{name}.pkl', bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    _LOG.info("Данные загружены")

    # обучаем модель
    X_train = data["X_train"].copy()
    y_train = data["y_train"].copy()

    model = models[m_name]
    
    metrics[f"{m_name}_train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    model.fit(X_train, y_train)
    
    prediction = model.predict(data["X_test"])
    metrics[f"{m_name}_train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    y_test = data["y_test"].copy()
    # метрики
    #result = {}
    metrics[f"{m_name}_R^2_score"] = r2_score(y_test, prediction)
    metrics[f"{m_name}_rmse"] = mean_squared_error(y_test, prediction)**0.5
    metrics[f"{m_name}_mae"] = median_absolute_error(y_test, prediction)

    return metrics 

    
def save_results(**kwargs) -> None:

    ti = kwargs["ti"]
    models_metrics = ti.xcom_pull(task_ids=["train_rf", "train_lr", "train_hgb"])
    result = {}
    for model_metrics in models_metrics:
        result.update(model_metrics)

    result["end_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    date = datetime.now().strftime("%Y_%m_%d_%H")
    s3_hook = S3Hook("aws_default")
    # сохраняем результаты обучения на s3
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    # сохраним в формате json на s3 в бакет в папку results
    json_byte_object = json.dumps(result)
    resource.Object(
        BUCKET,
        f"results/metrics_{date}.json").put(Body=json_byte_object)

 


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




