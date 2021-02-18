# -*- coding: utf-8 -*-
from datetime import time, timedelta, datetime

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from hk_hold.hk_hold import hk_hold_into_db

default_args = {
    'owner': 'Mr.Wang',
    'depends_on_past': True,
    'email': ['619511821@qq.com'],
    'email_on_failure': True,
    'wait_for_downstream': True,
    # 'email_on_retry': False,
    # 'sla': timedelta(minutes=5),
    # 'task_concurrency':5,#具有相同执行日期的任务运行的并发限制
    # 'retries': 1,
    # 'retry_delay': timedelta(seconds=5),
    # 'execution_timeout': timedelta(seconds=5*60),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
schedule_interval = '0 6 * * *'
start_date = datetime(2021, 1, 6)


@dag(default_args=default_args, schedule_interval=schedule_interval, start_date=start_date, tags=["share"])
def hk_hold():
    @task()
    def generate_execution_date(**context):
        context = get_current_context()
        execution_date = (context['execution_date']+timedelta(days=1)).strftime('%Y%m%d')
        print(f"execution_date=>{execution_date}")
        return execution_date

    @task()
    def hk_hold_into_db_task(execution_date):
        message = hk_hold_into_db(execution_date)
        return message

    execution_date = generate_execution_date()
    result = hk_hold_into_db_task(execution_date)


hk_hold_dag = hk_hold()
