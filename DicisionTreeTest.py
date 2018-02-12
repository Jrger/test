from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext

# --- Point 1, 2 ---
# Load and parse the data file into an RDD of LabeledPoint.
sc = SparkContext()
data = MLUtils.loadLibSVMFile(sc, 'E:\data\mllib\sample_libsvm_data.txt')
# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# --- Point 3, 4, 5 ---
# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
#testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
#print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())