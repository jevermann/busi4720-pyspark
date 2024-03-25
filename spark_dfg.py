import sys
from pyspark.sql import SparkSession

print("Job started.")
spark = (SparkSession.builder.appName('dfg').getOrCreate())
print("Spark session acquired.")

logSchema = 'caseID STRING, activity STRING, ts TIMESTAMP'

fname = sys.argv[1]
# fname='hdfs://localhost:9000/user/busi4720/eventlog.short.log'

data = spark.read \
.format('csv') \
.option('delimiter', '\t') \
.option('header', 'false') \
.schema(logSchema) \
.load(fname)

data.createOrReplaceTempView('log')
print("Data loaded.")

sql_query = \
'SELECT COUNT(*), l1.activity AS activity1, l2.activity AS activity2, AVG(l2.ts - l1.ts) AS dtime \
FROM log AS l1 JOIN log AS l2 ON l1.caseid=l2.caseid \
WHERE l2.ts = (SELECT MIN(ts) FROM log l3 \
WHERE l3.caseid=l1.caseid AND l3.ts > l1.ts) \
GROUP BY GROUPING SETS((l1.activity, l2.activity))'

dfg = spark.sql(sql_query)
print("Query executed.")

dfg.write.format('csv').mode('overwrite').save(fname + '.results')
print("Results written.")
dfg.show(dfg.count(), truncate=False)

spark.stop()
print("Spark session relinquished. Job completed.")