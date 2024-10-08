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

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

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

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"

models = dict(zip(["rf", "lr", "hgb"], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))




# напишем функцию, которая будет создавать даги
def create_dag(dag_id, m_name: Literal["rf", "lr", "hgb"]):
   
    def init() -> Dict[str, Any]:
        metrics = {}
        metrics["model"] = m_name
        metrics["start_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
        
        
        # создаем эксперимент
        exp_name = f"NicolasPal_p1_{m_name}"
    
        try:
            experiment_id = mlflow.create_experiment(exp_name, artifact_location=f"s3://{BUCKET}/mlflow/project1/{exp_name}")
            metrics["experiment_id"] = experiment_id
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id
            metrics["experiment_id"] = experiment_id

        mlflow.set_experiment(exp_name)
        
        
        
        return metrics 

    # функция чтения из postgres и загрузки на s3
    def get_data(**kwargs) -> Dict[str, Any]:

        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids="init")
        # время начало загрузки
        metrics["get_data_start"] = datetime.now().strftime("%Y%m%d %H:%M")

        # Используем созданный ранее PG connection
        pg_hook = PostgresHook("pg_connection")
        con = pg_hook.get_conn()

        # качаем данные
        data_initial = fetch_california_housing()
        # Объединим фичи и таргет в один np.array
        dataset = np.concatenate([data_initial['data'], data_initial['target'].reshape([data_initial['target'].shape[0],1])],axis=1)
        # Преобразуем в dataframe.
        data= pd.DataFrame(dataset, columns = data_initial['feature_names']+data_initial['target_names'])

        # Читаем все данные из таблицы california_housing
        #data = pd.read_sql_query("SELECT * FROM california_housing", con)

        # сохраняем результаты обучения на s3
        s3_hook = S3Hook("s3_connection")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")

        # Сохраняем файл в формате pkl на S3
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET,f"NicolasPal/project1/{m_name}/datasets/california_housing.pkl").put(Body=pickle_byte_obj)

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
        file = s3_hook.download_file(key=f"NicolasPal/project1/{m_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
        data = pd.read_pickle(file)
        
        _LOG.info("Данные для обработки из S3 загружены")

        # Обработка
        # Делим на фичи и таргет
        X, y = data[FEATURES], data[TARGET]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        # Сохранить готовые данные на S3
        s3_hook = S3Hook("s3_connection")
        session = s3_hook.get_session("eu-north-1")
        resource = session.resource("s3")
        
        for name, data in zip(["X_train", "X_test", "X_val", "y_train", "y_test", "y_val"],
                          [X_train, X_test, X_val, y_train, y_test, y_val]):
            pickle_byte_obj = pickle.dumps(data)
            resource.Object(BUCKET,
                            f"NicolasPal/project1/{m_name}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
        
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
        for name in ["X_train", "X_test", "X_val", "y_train", "y_test", "y_val"]:
            file = s3_hook.download_file(key=f'NicolasPal/project1/{m_name}/datasets/{name}.pkl', bucket_name=BUCKET)
            data[name] = pd.read_pickle(file)
    
        _LOG.info("Данные для обучения модели из S3 загружены")
        
        X_train = pd.DataFrame(data["X_train"], columns = FEATURES).copy()
        y_train = pd.Series(data["y_train"]).copy()
        
        X_test = pd.DataFrame(data["X_test"], columns = FEATURES).copy()
        y_test = pd.Series(data["y_test"]).copy()
        
        X_val = pd.DataFrame(data["X_val"], columns = FEATURES).copy()
        y_val = pd.Series(data["y_val"]).copy()
        
        exp_id = metrics["experiment_id"]
        
        
        # обучаем модель
        with mlflow.start_run(experiment_id=exp_id) as run:

            model = models[m_name]
            
            model.fit(pd.DataFrame(X_train), y_train)
            
            # Сделаем предсказание.
            prediction = model.predict(X_val)
        
        
            # Создадим валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val
            
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "logreg", registered_model_name=f"sk-learn-{m_name}-reg-model", signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )
        
        metrics["train_model_end"] = datetime.now().strftime("%Y%m%d %H:%M")
        
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
        resource.Object(BUCKET, f'NicolasPal/project1/{m_name}/result/{metrics['model']}_{date}.json').put(Body=json_byte_object)

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
            
