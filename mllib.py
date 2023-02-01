from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

# Start a Spark session
spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()

# Prepare the training data
training_data = spark.createDataFrame([
    (1.0, Vectors.dense([1.0, 2.0, 3.0])),
    (0.0, Vectors.dense([0.0, 1.0, 1.0])),
    (1.0, Vectors.dense([2.0, 1.0, 0.0]))
], ["label", "features"])

# Create the linear regression model
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Train the model on the training data
model = lr.fit(training_data)

# Print the coefficients and intercept of the model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Stop the Spark session
spark.stop()
    