import os
import datetime
os.environ["HADOOP_USER_NAME"] = "root"

import requests

from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col,lit,desc

from models.web_models import * 
from db.session import session, engine

def backend_info_analysis():
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf
				).master("local"
				).config("hive.metastore.uris","thrift://baidu:9083"
                                ).config("hive.exec.dynamic.partition.mode", "nonstrict"
				).enableHiveSupport(
 				).getOrCreate()


    spark.sparkContext.setLogLevel("ERROR")

    #truncate mysql by sqlalchemy
    session.query(WEB_ANALYSIS_API_ACCESS_NORMAL).delete()    
    session.query(WEB_ANALYSIS_API_ACCESS_UNNORMAL).delete()    
    session.query(WEB_ANALYSIS_IP_ACCESS_UNNORMAL).delete()    

    log = """
        select log_level,log_date,log_time,access_ip,access_type,access_api,access_code,access_time 
        from web.backend_info 
    """
    LOG = spark.sql(log)
    LOG.persist()
    #jdbc mysql info
    data = {
        "user":"root",
        "password":"wzx940516",
        "driver":"com.mysql.cj.jdbc.Driver",
    }
    #正常接口访问排名
    DF = LOG.filter("access_code = '200'")
    DF = DF.groupBy('access_api').count().orderBy(desc('count')).limit(15)
    DF.repartition(1).write.jdbc("jdbc:mysql://tencent:3306/DWDB","web_analysis_api_access_normal","append",data)
    #异常接口访问排名
    DF = LOG.filter("access_code != '200'")
    DF = DF.groupBy('access_api').count().orderBy(desc('count'))
    DF.repartition(1).write.jdbc("jdbc:mysql://tencent:3306/DWDB","web_analysis_api_access_unnormal","append",data)
    #异常IP访问排名
    DF = LOG.filter("access_code != '200'")
    DF = DF.groupBy('access_ip').count().orderBy(desc('count')).limit(20)
    DF = DF.withColumn("country",lit("unkown"))
    DF = DF.withColumn("city",lit("unkown"))
    DF.repartition(1).write.jdbc("jdbc:mysql://tencent:3306/DWDB","web_analysis_ip_access_unnormal","append",data)
    LOG.unpersist()
    return True


def get_ip_by_api(ip):
    ip = str(ip,encoding="utf-8")
    url=f'http://ip-api.com/json/{ip}'
    resp = requests.get(url=url)
    data = resp.json()
    return data

def update_ip_access_unnormal_address():
    ip_list = session.query(WEB_ANALYSIS_IP_ACCESS_UNNORMAL).all()
    for ip in ip_list:
        ip_json = get_ip_by_api(ip.access_ip)
        data = {
            "country":ip_json['country'],
            "city":ip_json['city']
        }
        session.query(WEB_ANALYSIS_IP_ACCESS_UNNORMAL).filter(WEB_ANALYSIS_IP_ACCESS_UNNORMAL.access_ip==ip.access_ip).update(data)

if __name__=="__main__":
    #backend_info_analysis()
    update_ip_access_unnormal_address()
    pass
