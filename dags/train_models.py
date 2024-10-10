import io
import json
import logging
import numpy as np
import pandas as pd
import pickle
import time

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

from airflow.models import Variable

# копируем из прошлого дага
DEFAULT_ARGS = {
    "owner" : "Nicolas Pal",
    "retry" : 3, # количество перезапусков
    "retry_delay" : timedelta(minutes=1) # задержка между запусками
}


_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"

models = dict(zip(["rf", "lr", "hgb"], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

# напишем функцию, которая будет создавать даги
def create_dag(dag_id, m_name: Literal["rf", "lr", "hgb"]):
   
    def init() -> Dict[str, Any]:
        metrics = {}
        metrics["model"] = m_name
        metrics["start_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
        return metrics 

    # функция чтения из postgres и загрузки на s3
    def get_data(**kwargs) -> Dict[str, Any]:

        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="init")
        # время начало загрузки
        metrics["get_data_start"] = datetime.now().strftime("%Y%m%d %H:%M")

        # качаем данные
        data_initial = fetch_california_housing()
        # Объединим фичи и таргет в один np.array
        dataset = np.concatenate([data_initial['data'], data_initial['target'].reshape([data_initial['target'].shape[0],1])],axis=1)
        # Преобразуем в dataframe.
        data= pd.DataFrame(dataset, columns = data_initial['feature_names']+data_initial['target_names'])

        # сохраняем результаты обучения на s3
        s3_hook = S3Hook("s3_connection")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")

        # Сохраняем файл в формате pkl на S3
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET,f"NicolasPal/{m_name}/datasets/california_housing.pkl").put(Body=pickle_byte_obj)

        _LOG.info("Данные загружены из б/д и сохранены на S3")

        # время конца загрузки
        metrics["get_data_end"] = datetime.now().strftime("%Y%m%d %H:%M")
        
        # размер датасета
        metrics["data_shape"] = data.shape

        return metrics  

    def prepare_data(**kwargs) -> Dict[str, Any]:
        
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "get_data")
        
        metrics["prepare_data_start"] = datetime.now().strftime("%Y%m%d %H:%M")
        
        # скачиваем данные с s3
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(key=f"NicolasPal/{m_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
        data = pd.read_pickle(file)
        
        _LOG.info("Данные для обработки из S3 загружены")

        # Обработка
        # Делим на фичи и таргет
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
        s3_hook = S3Hook("s3_connection")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
        
        for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
            pickle_byte_obj = pickle.dumps(data)
            resource.Object(BUCKET,
                            f"NicolasPal/{m_name}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
        
        _LOG.info("Данные подготовлены и сохранены на S3")
        
        metrics["prepare_data_end"] = datetime.now().strftime("%Y%m%d %H:%M")
        
        metrics["features_name"] = FEATURES

        return metrics


    def train_model(**kwargs) -> Dict[str, Any]:
        
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "prepare_data")
        
        metrics["train_model_start"] = datetime.now().strftime("%Y%m%d %H:%M")

        m_name = metrics["model"] 

        # считываем данные из s3
        s3_hook = S3Hook("s3_connection")
        
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(key=f"NicolasPal/{m_name}/datasets/{name}.pkl", bucket_name=BUCKET)
            data[name] = pd.read_pickle(file)
    
        _LOG.info("Данные для обучения модели из S3 загружены")
        
        # обучаем модель
        X_train = data["X_train"].copy()
        y_train = data["y_train"].copy()

        model = models[m_name]
        
        model.fit(X_train, y_train)
        
        prediction = model.predict(data["X_test"])
    
        y_test = data["y_test"].copy()
        
        metrics["train_model_end"] = datetime.now().strftime("%Y%m%d %H:%M")
        
        # метрики
        metrics['R^2_score'] = round(r2_score(y_test, prediction),2)
        metrics['rmse'] = round(mean_squared_error(y_test, prediction)**0.5,2)
        metrics['mae'] = round(median_absolute_error(y_test, prediction),2)
        
        _LOG.info("Модели обучены, метрики сохранены")
        
        return metrics 

        
    def save_results(**kwargs) -> None:

        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "train_model")

        metrics["end_tiemstamp"] = datetime.now().strftime("%Y%m%d %H:%M")

        s3_hook = S3Hook("s3_connection")
        # сохраняем результаты обучения на s3
        date = datetime.now().strftime("%Y_%m_%d_%H")

        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
    
        # сохраним в формате json на s3 в бакет в папку results
        json_byte_object = json.dumps(metrics)
        resource.Object(BUCKET, f"NicolasPal/{m_name}/result/{metrics['model']}_{date}.json").put(Body=json_byte_object)

        _LOG.info("Результаты сохранены в папку result")

    dag = DAG(
        dag_id = dag_id,   # уникальный id, можно использовать имя модели
        schedule_interval = "0 1 * * *",  # расписание: раз в день по ночам
        start_date = days_ago(2),  # в прошлом, иначе не запустится 
        catchup = False,     # чтобы не стал запускать сам себя за все прошлые даты
        tags = ["mlops"],   # уникальный 
        default_args = DEFAULT_ARGS)    

    with dag:
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag, provide_context=True)

        task_get_data = PythonOperator(task_id="get_data", python_callable=get_data, dag=dag, provide_context=True)
        
        task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, provide_context=True)
            
        task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag, provide_context=True)
        
        task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, provide_context=True)

        
        # архитектура дага
        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results


for model_name in models.keys():

    create_dag(f"{model_name}_train", model_name)
            
