# -*- coding: utf-8 -*-
"""

@author: Bannikov Maxim
"""

#spark-submit --master spark://mint-VirtualBox:7077 /home/mint/Downloads/Spark_script.py
from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession

# load in the data
spark = SparkSession.builder.appName("ALS").getOrCreate()
data = spark.read.option("header", True).option("inferSchema", True).csv("/home/mint/Downloads/ratings.csv")
ratings = data.select(['user_idx', 'movie_idx', 'rating'])

# split into train and test
train, test = ratings.randomSplit([0.8, 0.2])

# train the model
K = 10
epochs = 10
model = ALS.train(train, K, epochs)

# train
x = train.rdd.map(lambda row: (row.user_idx, row.movie_idx))
p = model.predictAll(x).toDF(['user_idx','movie_idx','predictions'])
ratesAndPreds = train.join(p, [train.user_idx == p.user_idx, train.movie_idx == p.movie_idx])
mse = ratesAndPreds.rdd.map(lambda r: (r.rating - r.predictions)**2).mean()
print("train mse: %s" % mse)

# test
x = test.rdd.map(lambda row: (row.user_idx, row.movie_idx))
p = model.predictAll(x).toDF(['user_idx','movie_idx','predictions'])
ratesAndPreds = test.join(p, [test.user_idx == p.user_idx, test.movie_idx == p.movie_idx])
mse = ratesAndPreds.rdd.map(lambda r: (r.rating - r.predictions)**2).mean()
print("test mse: %s" % mse)
