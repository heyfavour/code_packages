import datetime

from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col,lit

import os
os.environ["HADOOP_USER_NAME"] = "root"

def log_into_backend_info(execute_date):
    conf = SparkConf()
    spark = SparkSession.builder.config(conf=conf
				).master("local"
				).config("hive.metastore.uris","thrift://baidu:9083"
                                ).config("hive.exec.dynamic.partition.mode", "nonstrict"
				).enableHiveSupport(
 				).getOrCreate()


    spark.sparkContext.setLogLevel("ERROR")

    log_date = datetime.datetime.strptime(execute_date,'%Y%m%d').strftime('%Y-%m-%d') 
    log_file = f"file:///public/logs/backend/backend_info.log.{log_date}"

    schema = StructType([
        StructField("log_level", StringType(), True),
        StructField("log_date", StringType(), True),
        StructField("log_time", StringType(), True),
        StructField("access_ip", StringType(), True),
        StructField("access_type", StringType(), True),
        StructField("access_api", StringType(), True),
        StructField("access_code", StringType(), True),
        StructField("access_time", StringType(), True),
    ])

    DF = spark.read.option("inferSchema", "true"
                  ).option("delimiter"," "
                  ).csv(log_file,schema=schema)

    DF = DF.withColumn("year",DF.log_date[1:4])
    DF = DF.withColumn("month",DF.log_date[6:2])
    DF.repartition(1).write.format("Hive").partitionBy('year','month').mode("append").saveAsTable("web.backend_info")

    return True


if __name__=="__main__":
    #execute_date = "20210202"
    #log_into_backend_info(execute_date)
    pass
