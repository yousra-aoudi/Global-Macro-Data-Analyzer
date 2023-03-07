import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.vector_ar.var_model import VAR


class DataAnalyzer:
    data_sources = {
        'yahoo': 'Yahoo Finance',
        'alpha_vantage': 'Alpha Vantage',
        'twitter': 'Twitter',
        'reddit': 'Reddit'
    }

    def __init__(self, ticker, target_variable, data_source='yahoo', date_col='Date'):
        self.ticker = ticker
        self.target_variable = target_variable
        self.data_source = data_source
        self.date_col = date_col
        self.data = None

    def load_data(self):
        if self.data_source == 'yahoo':
            # Load data from Yahoo Finance API
            ticker = yf.Ticker(self.ticker)
            self.data = ticker.history(period="max")
            self.data = self.data.reset_index()
            self.data = self.data.rename(columns={self.date_col: 'date'})

            # Select the target variable and date column
            self.data = self.data[['date', self.target_variable]]

            # Remove missing values
            self.data = self.data.dropna()

            # Convert date column to datetime format
            self.data['date'] = pd.to_datetime(self.data['date'])
        elif self.data_source == 'alpha_vantage':
            # Load data from Alpha Vantage API
            pass
        elif self.data_source == 'twitter':
            # Load data from Twitter API
            pass
        elif self.data_source == 'reddit':
            # Load data from Reddit API
            pass
        else:
            print("Invalid data source.")

    def preprocess_data(self):
        pd.options.mode.chained_assignment = None  # default='warn'
        # Replace missing values with the mean
        self.data = self.data.fillna(self.data.mean())

        # Convert date strings to datetime objects
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Filter out irrelevant variables and rows
        self.data = self.data[[self.target_variable, 'date']]
        self.data = self.data.dropna()
        print(' data info ', self.data.info())
        print('data head ', self.data.head())

    def explore_data(self):
        # Plot the target variable over time
        plt.plot(self.data[self.target_variable])
        plt.title(f"{self.target_variable} over Time")
        plt.show()

    def build_model(self):
        date_col = self.data['date']
        # Convert date strings to datetime objects
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Select the features and target variable
        print('Before dropping : X shape :', self.data.shape)
        print('Before dropping : X columns names :', self.data.columns)
        X = self.data[['Close', 'date']]
        X['date'] = X['date'].astype(np.int64) // 10 ** 9
        print('After dropping : X shape :', X.shape)
        y = self.data[self.target_variable]
        print('y shape :', y.shape)
        print('y name :', y.index)

        # Create a decision tree regression model
        model = DecisionTreeRegressor(random_state=42)

        # Fit the model to the data
        model.fit(X, y)

        # Use the model to make predictions
        predictions = model.predict(X)

        # Plot the predictions against the true values
        plt.plot(y, predictions, 'o')
        plt.title(f"Predicted vs. Actual {self.target_variable}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()

    def build_var_model(self):
        # Select the variables to include in the VAR model
        cols = [col for col in self.data.columns if col != 'date']
        data = self.data[cols]

        # Create a VAR model and fit it to the data
        model = VAR(data)
        results = model.fit()

        # Print the model summary
        print(results.summary())

        # Plot the results
        results.plot()
        plt.show()


if __name__ == "__main__":
    # Create an instance of the DataAnalyzer class with Yahoo Finance as the default data source
    analyzer = DataAnalyzer('AAPL', 'Close', data_source='yahoo', date_col='Date')

    # Load and preprocess the data
    analyzer.load_data()
    analyzer.preprocess_data()

    # Explore the data and build a model
    analyzer.explore_data()
    analyzer.build_model()

    # Generate recommendations
    # analyzer.generate_recommendations()
