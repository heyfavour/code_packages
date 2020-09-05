# -*- coding: utf-8 -*-
from datetime import time,timedelta,datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from hk_hold.hk_hold import hk_hold_into_db

default_args = {
    'owner': 'QT',
    'depends_on_past': True,
    'start_date': datetime(2020,8,20),
    'email': ['619511821@qq.com'],
    'email_on_failure': True,
    'wait_for_downstream': True,
    #'email_on_retry': False,
    #'sla': timedelta(minutes=5),
    #'task_concurrency':5,#具有相同执行日期的任务运行的并发限制
    #'retries': 1,
    #'retry_delay': timedelta(seconds=5),
    #'execution_timeout': timedelta(seconds=5*60),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    dag_id='hk_hold',
    default_args=default_args,
    schedule_interval='0 6 * * *',
)


into_db = PythonOperator(
    task_id='into_db',
    python_callable=hk_hold_into_db,
    dag=dag,
    provide_context=False,
    op_kwargs={
        'runday': "{{execution_date.strftime('%Y%m%d')}}",
    }
)
