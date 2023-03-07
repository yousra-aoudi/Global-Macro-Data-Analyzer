from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.vector_ar.var_model import VAR

# sources to import
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import tweepy
import praw


class GlobalMacroDataAnalyzer:
    DATA_SOURCES = {"Financial data from stock exchanges": ["Yahoo Finance", "Google Finance", "Alpha Vantage"],
                    "Economic data from government agencies": ["Bureau of Labor Statistics", "US Census Bureau",
                                                               "Federal Reserve Economic Data (FRED)"],
                    "Alternative data sources": ["Twitter", "News APIs", "Social media sentiment data providers"]
                    }

    def __init__(self, data_source, ticker, target_variable):
        self.end_date = None
        self.start_date = None
        self.data = None
        self.ticker = ticker
        self.target_variable = None
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
            self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        elif data_source == 'alpha_vantage':
            # Load data from Alpha Vantage API
            ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
            self.data, _ = ts.get_daily(symbol=self.ticker, outputsize='full')
            self.data = self.data.sort_index()
        elif data_source == 'twitter':
            # Load data from Twitter API
            auth = tweepy.OAuthHandler("YOUR_CONSUMER_KEY", "YOUR_CONSUMER_SECRET")
            auth.set_access_token("YOUR_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_SECRET")
            api = tweepy.API(auth)
            # Get tweets related to the specified keyword
            tweets = api.search(q=self.keyword, count=100, tweet_mode='extended')
            # Convert tweets to a pandas DataFrame
            self.data = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['text'])
            self.data['date'] = pd.to_datetime([tweet.created_at for tweet in tweets])
        elif data_source == 'reddit':
            # Load data from Reddit API
            reddit = praw.Reddit(client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET',
                                 user_agent='YOUR_USER_AGENT')
            subreddit = reddit.subreddit(self.subreddit)
            # Get top posts from the subreddit
            posts = subreddit.top(limit=self.limit)
            # Convert posts to a pandas DataFrame
            self.data = pd.DataFrame(data=[post.title for post in posts], columns=['title'])
            self.data['Date'] = pd.to_datetime([datetime.fromtimestamp(post.created_utc) for post in posts])
        else:
            print("Invalid data source.")

    def preprocess_data(self):
        # clean and preprocess the data to handle inconsistencies in data collection and formatting across countries,
        # and deal with outliers and missing data points
        self.data = self.data.dropna()
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

        # create a linear regression model to make predictions
        model = LinearRegression()
        model.fit(X, y)

        # print the model's coefficients and plot predictions against the true values
        print("Intercept:", model.intercept_)
        print("Coefficients:", model.coef_)
        predictions = model.predict(X)
        plt.plot(y, predictions, 'o')
        plt.title(f"Predicted vs. Actual {self.target_variable}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.show()

    def build_var_model(self):
        # use more complex models such as VAR models, which allow for multiple variables to be analyzed simultaneously
        # and account for the dynamic relationships between them
        cols = [col for col in self.data.columns if col != 'date']
        data = self.data[cols]
        model = VAR(data)
        results = model.fit()
        print(results.summary())

    def generate_recommendations(self):
        # use insights and findings from the analysis to develop recommendations for managing risk and maximizing
        # returns
        print(f"Based on the trends and patterns identified in the {self.target_variable} data, we recommend...")


if __name__ == "__main__":
    analyzer = GlobalMacroDataAnalyzer('yahoo', 'AAPL', 'Close')
    analyzer.choose_data_source()
    analyzer.load_data('yahoo')
    analyzer.preprocess_data()
    analyzer.explore_data()
    analyzer.build_model()
    analyzer.generate_recommendations()