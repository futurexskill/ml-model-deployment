from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("SparkMLRegressionExample").getOrCreate()

# Create a dataframe with sample data
data = [(30, 50000, 1.0), (40, 60000, 0.0), (50, 70000, 1.0), (60, 80000, 0.0), (70, 90000, 1.0)]
df = spark.createDataFrame(data, ["age", "salary", "purchased"])

# Convert features into a vector using the VectorAssembler
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["age", "salary"], outputCol="features")
vector_df = assembler.transform(df)

# Split the data into training and test sets
training_df, test_df = vector_df.randomSplit([0.7, 0.3], seed=42)

# Build the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="purchased")
model = lr.fit(training_df)

# Make predictions on the test data
predictions = model.transform(test_df)

# Evaluate the model's performance using mean squared error
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="purchased", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
print("Mean Squared Error: ", mse)

# Stop the Spark session
spark.stop()
