from datetime import timedelta
from typing import NoReturn


from airflow.models import DAG  
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner" : "Nicolas Pal",
    "retry" : 3, # количество перезапусков
    "retry_delay" : timedelta(minutes=1) # задержка между запусками
}
    
    


dag = DAG(
    dag_id = "mlops_dag_1",   # уникальный id, можно использовать имя модели
    schedule_interval = "0 1 * * *",  # расписание: раз в день по ночам
    start_date = days_ago(2),  # в прошлом, иначе не запустится 
    catchup = False,     # чтобы не стал запускать сам себя за все прошлые даты
    tags = ["mlops"],   # уникальный 
    default_args = DEFAULT_ARGS 
)    


def init() -> NoReturn:  # функция под задание
    print("Hello, World")

# пишем задание
task_init = PythonOperator(task_id = "init", python_callable = init, dag=dag)

# архитектура задания
task_init