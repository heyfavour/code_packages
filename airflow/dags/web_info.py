# -*- coding: utf-8 -*-
from datetime import time, timedelta, datetime

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from web_analysis.log_extra import log_into_backend_info
from web_analysis.log_transform import backend_info_analysis,update_ip_access_unnormal_address

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
schedule_interval = '10 6 * * *'
start_date = datetime(2021, 2, 2)


@dag(default_args=default_args,schedule_interval=schedule_interval,start_date=start_date,tags=["web"],max_active_runs=1)
def web_info():
    @task()
    def generate_execution_date():
        context = get_current_context()
        execution_date = (context['execution_date']+timedelta(days=1)).strftime('%Y%m%d')
        return execution_date

    @task()
    def log_into_backend_info_task(execution_date):
        message = log_into_backend_info(execution_date)
        return message

    @task()
    def backend_info_analysis_task():
        context = get_current_context()
        execution_date = (context['execution_date']+timedelta(days=1)).strftime('%Y%m%d')
        if int(execution_date)<=20210216:return
        backend_info_analysis()
        return True
    
    @task()
    def update_ip_access_unnormal_address_task():
        context = get_current_context()
        execution_date = (context['execution_date']+timedelta(days=1)).strftime('%Y%m%d')
        if int(execution_date)<=20210216:return
        update_ip_access_unnormal_address()
        return True
    
    execution_date = generate_execution_date()
    result = log_into_backend_info_task(execution_date)

    web_analysis = backend_info_analysis_task()
    update_ip_address = update_ip_access_unnormal_address_task()

    result >> web_analysis >> update_ip_address


web_info_dag = web_info()
