import sys, time
from pyspark.sql  import SparkSession
from pyspark.sql.functions import explode, split, col, desc, window, current_timestamp

print("Job started.")
spark = SparkSession.builder.appName('streaming').getOrCreate()
print("Spark session acquired.")

lines = spark.readStream \
             .format('socket') \
             .option('host', 'localhost') \
             .option('port', 9999) \
             .load()

words = lines.select(explode(split(col('value'), '\\s')).alias('word')).withColumn('eventTime', current_timestamp())
counts = words.groupBy('word', window('eventTime', '1 minute', '30 second')).count().sort(desc('count'))

writer = counts.writeStream \
               .format('console') \
               .outputMode('complete') \
               .trigger(processingTime='5 second') \
               .option('checkpointLocation', 'hdfs://localhost:9000/user/busi4720/')

streamingQuery = writer.start()
print("StreamingQuery started.")
