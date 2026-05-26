import pandas as pd

def load_portfolio_data():

    df = pd.read_csv("data/raw/portfolio.csv")

    return df
