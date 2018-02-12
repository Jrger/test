from pyspark.ml import Estimator, Transformer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, Param, Params, TypeConverters, VectorAssembler, VectorIndexer, Tokenizer, \
    HashingTF, OneHotEncoder, QuantileDiscretizer, Normalizer
from pyspark.ml.linalg import VectorUDT, Vectors, DenseVector
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark=SparkSession.builder.master("local[1]").appName("test").getOrCreate()

df = spark.read.csv(
    "data.txt",
    encoding="utf-8",
    header=False,
    schema=StructType(
        [StructField("text", StringType()),
         StructField("index", FloatType())]))


string_index = StringIndexer(inputCol="text",outputCol="text_number")
encoder = OneHotEncoder(inputCols=["text_number"], outputCols=["text_onehot"])
pipline = Pipeline(stages=[string_index,encoder])
pipline.fit(df).transform(df).show()
