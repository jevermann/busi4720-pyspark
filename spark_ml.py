import sys
from pyspark.sql import SparkSession

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler, \
    StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml import PipelineModel

print("Job started.")
spark = (SparkSession.builder.appName('ml').getOrCreate())
print("Spark session acquired.")

the_schema = 'class STRING, `cap-diameter` DOUBLE, \
    `cap-shape` STRING, `cap-surface` STRING, \
    `cap-color` STRING, `does-bruise-or-bleed` STRING, \
    `gill-attachment` STRING, `gill-spacing` STRING, \
    `gill-color` STRING, `stem-height` DOUBLE, \
    `stem-width` DOUBLE, `stem-root` STRING, \
    `stem-surface` STRING, `stem-color` STRING, \
    `veil-type` STRING, `veil-color` STRING, \
    `has-ring` STRING, `ring-type` STRING, \
    `spore-print-color` STRING, habitat STRING, \
    season STRING'

# fname='hdfs://localhost:9000/user/busi4720/mushrooms.csv'
fname = sys.argv[1]

data = spark.read \
    .format('csv') \
    .option('delimiter', ',') \
    .option('header', 'true') \
    .schema(the_schema) \
    .load(fname)

data = data.drop('veil-type')
data = data.fillna('NULL')

numFeatures = VectorAssembler(
    inputCols = ['cap-diameter', 'stem-width',
                 'stem-height'],
    outputCol = 'numFeatures')
# data = numFeatures.transform(data)

scaler = StandardScaler(inputCol='numFeatures',
                        outputCol='numFeaturesS')
# data = scaler.fit(data).transform(data)

categoricalCols = \
    [name for (name, dtype) in data.dtypes \
        if dtype=='string']
indexOutputCols = \
    [x + 'index' for x in categoricalCols]
oheOutputCols = \
    [x + 'ohe' for x in categoricalCols]

stringIndexer = StringIndexer(
    inputCols = categoricalCols,
    outputCols = indexOutputCols,
    handleInvalid='skip')
# data = stringIndexer.fit(data).transform(data)

oheEncoder = OneHotEncoder(
    inputCols = indexOutputCols,
    outputCols = oheOutputCols)
# data = oheEncoder.fit(data).transform(data)

vecAssembler = VectorAssembler(
    inputCols = oheOutputCols+['numFeaturesS'],
    outputCol = 'feature_vec')
# data = vecAssembler.transform(data)

stringIndexTarget = StringIndexer(
    inputCols = ['class'],
    outputCols = ['classIndex'],
    handleInvalid='skip')
# data = stringIndexTarget.fit(data).transform(data)

logReg = LogisticRegression(
    featuresCol = 'feature_vec',
    labelCol = 'classIndex')
# logRegModel = logReg.fit(data)

pipeline = Pipeline(stages=[
    numFeatures,
    scaler,
    stringIndexer,
    oheEncoder,
    vecAssembler,
    stringIndexTarget,
    logReg])

train_data, test_data = \
    data.randomSplit([.66, .33], seed=1)

print("Fitting model")
# Fit to the training data
pipelineModel = pipeline.fit(train_data)

summary = pipelineModel.stages[-1].summary
print("Training accuracy:")
print(summary.accuracy)
print("Training ROC:")
print(summary.areaUnderROC)
print("Training F-measures:")
summary.fMeasureByThreshold.show()
print("Training precision:")
print(summary.precisionByLabel)
print("Training recall:")
print(summary.recallByLabel)

print("Predicting training data")
# Predict from the training data
trainPred = pipelineModel.transform(train_data)
# Predict from the test data
print("Predicting test data")
testPred = pipelineModel.transform(test_data)

evaluator = BinaryClassificationEvaluator(
    labelCol='classIndex')
print("Evaluating AUC on training data")
print(evaluator.evaluate(trainPred))
print("Evaluating AUC on testing data")
print(evaluator.evaluate(testPred))

print("Writing model")
pipelineModel.write().overwrite().save('myFirstModel')
print("Reading model")
savedModel = PipelineModel.load('myFirstModel')
