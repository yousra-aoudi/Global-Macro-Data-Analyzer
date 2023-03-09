from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col
from pyspark.ml.regression import LinearRegression as SparkLinearRegression
from pyspark.ml import Pipeline as SparkPipeline

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


"""
We now split the data into training and testing sets using the train_test_split function from sklearn.model_selection. 
We then use a Pipeline to preprocess the data and build the model. The Pipeline consists of two steps: preprocess and 
model. The preprocess step uses the ColumnTransformer we defined earlier to preprocess the data, while the model step 
uses the model specified in the model parameter when creating an instance of the GlobalMacroDataPipelineSpark class.

We fit the Pipeline to the training data using the fit method, and then use it to make predictions on the testing data
using the predict method. We plot the predictions against the true values and print the model's performance metrics, 
including the R^2 score, mean absolute error, mean squared error, and root mean squared error.
"""

"""
Difference between Spark and AirFlow
There could be several reasons for that. Spark is a distributed computing framework that is well-suited for processing 
large volumes of data in parallel across a cluster of machines. However, setting up and maintaining a Spark cluster can 
be complex and require significant resources.

On the other hand, Airflow is a workflow management platform that is designed to automate and schedule data processing 
tasks. While Airflow can also work with Spark, it can also integrate with other tools and technologies, such as Python, 
SQL, and Docker.

In some cases, using Airflow may be more practical and cost-effective than setting up a Spark cluster, especially for 
smaller projects or for teams without significant resources or expertise in managing distributed systems.

Both Spark and Airflow are capable of handling large volumes of data. The choice between them depends on specific 
requirements and use cases. Spark is more suited for data processing and analysis, while Airflow is more focused on 
workflow management and scheduling. In general, Spark is used for more compute-intensive tasks, while Airflow is used 
for orchestrating and scheduling tasks.

It may be beneficial to create two separate classes - one that utilizes Apache Airflow and another that utilizes 
Apache Spark - to allow for greater flexibility in handling different types and sizes of data. This would allow the 
team to choose the appropriate tool depending on the specific needs and constraints of each project. However, it would 
also require additional development and maintenance resources, so the decision should be made based on the specific 
needs and resources of the organization.

Pipeline is a class in the sklearn library. It is a tool for chaining multiple steps together, allowing for easy and 
efficient preprocessing and modeling. The Pipeline class takes a list of steps as input, where each step is a tuple 
containing the name of the step and the transformation or model to apply. For example, a preprocessing step could be 
StandardScaler, followed by a model step using LinearRegression.
"""

"""
Both Spark ML and scikit-learn provide machine learning libraries, but they are designed for different use cases.

Spark ML is designed for distributed computing and big data processing. It uses the power of Apache Spark's distributed 
computing engine to scale machine learning algorithms across many nodes in a cluster. This makes it well-suited for 
processing large amounts of data in parallel and building models on datasets that are too large to fit into memory on a
single machine.

On the other hand, scikit-learn is designed for machine learning tasks on small to medium-sized datasets that can fit 
into memory on a single machine. It provides a wide range of machine learning algorithms and tools for data 
preprocessing, model selection, and evaluation. While scikit-learn is not designed for distributed computing, it is 
often faster than Spark ML for small to medium-sized datasets.

In summary, Spark ML is designed for distributed computing and big data processing, while scikit-learn is designed for 
machine learning on small to medium-sized datasets that can fit into memory on a single machine.
"""


class GlobalMacroDataPipelineSpark:
    DATA_SOURCES = {"Financial data from stock exchanges": ["Yahoo Finance", "Google Finance", "Alpha Vantage"],
                    "Economic data from government agencies": ["Bureau of Labor Statistics", "US Census Bureau",
                                                               "Federal Reserve Economic Data (FRED)"],
                    "Alternative data sources": ["Twitter", "News APIs", "Social media sentiment data providers"]
                    }

    def __init__(self, data, ticker=None, target_variable=None, train_ratio=None, end_date=None, start_date=None,
                 preprocessor=None, model=None, random_seed=None):
        self.model = model
        self.preprocessor = preprocessor
        self.train_ratio = train_ratio
        self.end_date = end_date
        self.start_date = start_date
        self.data = data
        self.spark = SparkSession.builder.appName("GlobalMacroDataPipelineSpark").getOrCreate()
        self.ticker = ticker
        self.target_variable = target_variable
        self.random_seed = random_seed
        self.data_sources = {
            'financial': ['yahoo', 'alpha_vantage'],
            'alternative': ['twitter', 'reddit']
        }

    def choose_data_source(self):
        print("Please choose a data source type:")
        for key in self.data_sources.keys():
            print(f"- {key}")
        data_source_type = input().lower()

        while data_source_type not in self.data_sources.keys():
            print("Invalid input. Please choose a valid data source type.")
            data_source_type = input().lower()

        print(f"Available {data_source_type} data sources:")
        for source in self.data_sources[data_source_type]:
            print(f"- {source}")
        data_source = input("Choose a data source: ")

        return data_source

    def load_data(self, data_source):
        if data_source == 'yahoo':
            # Load data from Yahoo Finance API
            self.data = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}'
                                    f'?period1={int(self.start_date.timestamp())}&period2={int(self.end_date.timestamp())}'
                                    f'&interval=1d&events=history&includeAdjustedClose=true')
            self.data = self.spark.createDataFrame(self.data)
        elif data_source == 'alpha_vantage':
            # Load data from Alpha Vantage API
            pass
        elif data_source == 'twitter':
            # Load data from Twitter API
            pass
        elif data_source == 'reddit':
            # Load data from Reddit API
            pass
        else:
            print("Invalid data source.")

    def preprocess_data(self):
        # clean and preprocess the data to handle inconsistencies in data collection and formatting across countries,
        # and deal with outliers and missing data points
        self.data = self.data.dropna()
        self.data = self.data.select([col(c).cast("double") for c in self.data.columns])
        self.data = self.data.dropna()
        self.data = self.data.toPandas()
        self.data = pd.DataFrame(self.data).reset_index()
        print('data.head ', self.data.head())
        print('data.index ', self.data.index)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.index = self.data['Date']

    def explore_data(self):
        # create visualizations such as time series plots, heat maps, and scatter plots to identify trends and patterns
        plt.plot(self.data[self.target_variable])
        plt.title(f"{self.target_variable} over Time")
        plt.show()

    def build_model(self):
        # use data analysis and statistical modeling techniques to identify trends and patterns in the data
        X = self.data.drop(columns=[self.target_variable])
        y = self.data[self.target_variable]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use a pipeline to preprocess the data and build the model
        pipeline = SklearnPipeline([
            ('preprocess', self.preprocessor),
            ('model', self.model)
        ])

        # Fit the pipeline to the training data
        pipeline.fit(X_train, y_train)

        # Print the model's coefficients and plot predictions against the true values
        print("Intercept:", pipeline.named_steps['model'].intercept_)
        print("Coefficients:", pipeline.named_steps['model'].coef_)
        predictions = pipeline.predict(X_test)
        plt.plot(y_test, predictions, 'o')
        plt.title(f"Predicted vs. Actual {self.target_variable}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()

        # Print the model's performance metrics
        """
        Parameters:
        - y_test (array-like): The true target values.
        - predictions (array-like): The predicted target values.
        """
        print("R^2 Score:", r2_score(y_test, predictions))
        print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
        print("Mean Squared Error:", mean_squared_error(y_test, predictions))
        print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predictions)))

    def run(self):
        # Read data from data source
        df = spark.read.csv(self.data)

        # Preprocess data using Spark DataFrame operations
        df = df.dropna()
        df = df.withColumn('Date', to_date(col('Date')))
        df = df.withColumnRenamed(self.target_variable, 'label')
        feature_cols = [col for col in df.columns if col not in ['Date', 'label']]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        df = assembler.transform(df).select('Date', 'features', 'label')

        # Split data into train and test sets
        train_df, test_df = df.randomSplit([self.train_ratio, 1 - self.train_ratio], seed=self.random_seed)

        # Train linear regression model
        lr = SparkLinearRegression(labelCol='label', featuresCol='features')
        lr_model = lr.fit(train_df)

        # Make predictions on test set
        predictions = lr_model.transform(test_df)

        # Evaluate model performance
        evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='r2')
        r2 = evaluator.evaluate(predictions)
        mae = evaluator.evaluate(predictions, {evaluator.metricName: 'mae'})
        mse = evaluator.evaluate(predictions, {evaluator.metricName: 'mse'})
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: 'rmse'})

        # Print model performance metrics
        print("R^2 Score:", r2)
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("Root Mean Squared Error:", rmse)


if __name__ == "__main__":
    # Create a SparkSession
    spark = SparkSession.builder.appName("GlobalMacroDataPipelineSpark").getOrCreate()

    # Load the data from a CSV file
    data = spark.read.csv("data.csv", header=True)

    # Create a GlobalMacroDataPipeline instance
    pipeline = GlobalMacroDataPipelineSpark(data, target_variable="Close")

    # Run the pipeline
    pipeline.run()

    # Stop the SparkSession
    spark.stop()
