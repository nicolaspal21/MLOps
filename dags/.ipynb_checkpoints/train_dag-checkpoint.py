import io
import json
import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import fetch_california_housing
from sqlalchemy import create_engine

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import NoReturn

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

dag = DAG(
    dag_id = "train",   # уникальный id, можно использовать имя модели
    schedule_interval = "0 1 * * *",  # расписание: раз в день по ночам
    start_date = days_ago(2),  # в прошлом, иначе не запустится 
    catchup = False,     # чтобы не стал запускать сам себя за все прошлые даты
    tags = ["mlops"],   # уникальный 
    default_args = DEFAULT_ARGS 
)    

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = "test-bucket-nicolas-1"
DATA_PATH = "datasets/california_housing.pkl"
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"


# для замера времени шагов и сбора метрик
def init() -> NoReturn:
    _LOG.info("Train pipeline started.")

# функция чтения из postgres и загрузки на s3
def get_data_from_postgres() -> NoReturn:
    
    # хук для чтения данных (хуки - инструменты взаимодействия с внешними данными)
    # используем ранее созданный PG connection
    # pg_hook = PostgresHook("pg_connection")
    # con = pg_hook.get_conn()
    
    # c помощью конекшена читаем данные из california_housing
    #data = pd.read_sql_query("SELECT * FROM california_housing", con)

    data_initial = fetch_california_housing()
    # Объединим фичи и таргет в один np.array
    dataset = np.concatenate([data_initial['data'], data_initial['target'].reshape([data_initial['target'].shape[0],1])],axis=1)
    # Преобразуем в dataframe.
    data= pd.DataFrame(dataset, columns = data_initial['feature_names']+data_initial['target_names'])
    

    # положим данные в S3 на яндекс клоуд, для этого используем созданный ранее S3 хук
    # используем созданный ранее s3 connection
    #s3_hook = S3Hook("s3_connection")
    s3_hook = S3Hook("aws_default")
    #session = s3_hook.get_session("kz1")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    # положим датасет как пикл на S3
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)

    # добавим лог что загрузка данных завершена
    _LOG.info("Data download finished.")

# шаг выгрузки из s3, предобработки данных и снова записи в s3
def prepare_data() -> NoReturn:

    # берем s3 хук 
    #s3_hook = S3Hook("s3_connection")
    s3_hook = S3Hook("aws_default")
    # читаем ранее сохраненный файл
    file = s3_hook.download_file(key = DATA_PATH, bucket_name = BUCKET)
    data = pd.read_pickle(file)

    # делим данные на X и y
    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # делаем стандартизацию
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # сохраняем датасетики на s3 в цикле
    #session = s3_hook.get_session("kz1")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                      [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f'dataset/{name}.pkl').put(Body=pickle_byte_obj)

    # лог о завершении стадии
    _LOG.info("Data preparation finished.")

# скачиваем предобработанные данные из s3 и обучаем модель
def train_model() -> NoReturn:

    # считываем данные из s3
    #s3_hook = S3Hook("s3_connection")
    s3_hook = S3Hook("aws_default")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f'dataset/{name}.pkl', bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    _LOG.info("Данные загружены")
    
    # обучаем модель
    model = RandomForestRegressor()
    _LOG.info("Модель подгружена")

    X_train = data["X_train"].copy()
    y_train = data["y_train"].copy()

    model.fit(X_train, y_train)
    _LOG.info("Модель обучена")
    
    prediction = model.predict(data["X_test"])
    _LOG.info("Данные предсказаны")

    y_test = data["y_test"].copy()
    
    # метрики
    result = {}
    result['R^2_score'] = r2_score(y_test, prediction)
    result['rmse'] = mean_squared_error(y_test, prediction)**0.5
    result['mae'] = median_absolute_error(y_test, prediction)
    _LOG.info("Считаем метрики")
    
    # сохраняем результаты обучения на s3
    date = datetime.now().strftime("%Y_%m_%d_%H")
    #session = s3_hook.get_session("kz1")
    session = s3_hook.get_session("eu-north-1")
    resource = session.resource("s3")

    # сохраним в формате json на s3 в бакет в папку results
    json_byte_object = json.dumps(result)
    resource.Object(BUCKET, f'result/{date}.json').put(Body=json_byte_object)
    
    _LOG.info("Model training finished.")


def save_results() -> NoReturn:
    _LOG.info("Все посчитталось")

# обернем функции в операторы и пропишем архитектуру данных

task_init = PythonOperator(task_id="init", python_callable=get_data_from_postgres, dag=dag)

task_get_data = PythonOperator(task_id="get_data", python_callable=init, dag=dag)

task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag)
    
task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag)

task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results
    