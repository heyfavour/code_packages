ps -ef|egrep 'scheduler|airflow'|grep -v grep|awk '{print $2}'|xargs kill -9
nohup airflow webserver -p 8083 > webserver.log &
nohup airflow scheduler > scheduler.log &
