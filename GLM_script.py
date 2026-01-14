import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# -----------------------------------------
# Config
# -----------------------------------------

DATA_DIR = "."  # current folder

PRODUCT_CONFIG_PATH = os.path.join(DATA_DIR, "product_config.csv")
WEEKLY_SALES_PATH = os.path.join(DATA_DIR, "weekly_sales.csv")
WEEKLY_MEDIA_PATH = os.path.join(DATA_DIR, "weekly_media.csv")

# DB connection. Two options:
# 1) Hard code a connection string here
# 2) Or store it in an .env file as DB_URL
#
# Example Postgres URL:
# postgresql+psycopg2://user:password@host:5432/dbname

load_dotenv()
DB_URL = os.getenv("DB_URL", "postgresql+psycopg2://user:password@localhost:5432/mydb")

RESULTS_TABLE = "glm_coefficients"


# -----------------------------------------
# Data loading helpers
# -----------------------------------------

def load_data():
    print("Loading data...")

    product_cfg = pd.read_csv(PRODUCT_CONFIG_PATH)

    sales = pd.read_csv(WEEKLY_SALES_PATH, parse_dates=["week_start_date"])
    media = pd.read_csv(WEEKLY_MEDIA_PATH, parse_dates=["week_start_date"])

    # Basic sanity checks
    required_sales_cols = {"week_start_date", "product_id", "retailer_id", "net_sales"}
    required_media_cols = {"week_start_date", "product_id", "retailer_id", "spend"}

    missing_sales = required_sales_cols - set(sales.columns)
    missing_media = required_media_cols - set(media.columns)

    if missing_sales:
        raise ValueError(f"weekly_sales.csv missing columns: {missing_sales}")
    if missing_media:
        raise ValueError(f"weekly_media.csv missing columns: {missing_media}")

    return product_cfg, sales, media


def build_model_frame(product_cfg, sales, media):
    print("Building modeling dataset...")

    df = (
        sales.merge(
            media,
            on=["week_start_date", "product_id", "retailer_id"],
            how="left",
            suffixes=("", "_media"),
        )
        .merge(product_cfg, on="product_id", how="left")
    )

    # Create minimal features for V1
    # You can add more later (channel splits, promos, seasonality, etc)
    df["log_spend_plus1"] = np.log1p(df["spend"])
    df = df.dropna(subset=["net_sales", "log_spend_plus1"])

    return df


# -----------------------------------------
# GLM logic
# -----------------------------------------

def fit_global_glm(df):
    """
    V1: single GLM across all product x week x retailer.
    Target: net_sales
    Feature(s): log_spend_plus1
    """

    print("Fitting GLM...")

    y = df["net_sales"]
    X = df[["log_spend_plus1"]]

    # Add intercept
    X = sm.add_constant(X)

    model = sm.GLM(y, X, family=sm.families.Gaussian())
    results = model.fit()

    print(results.summary())

    coef_df = results.params.reset_index()
    coef_df.columns = ["term", "coefficient"]

    return coef_df


# -----------------------------------------
# DB write
# -----------------------------------------

def write_coefficients_to_db(coef_df):
    print(f"Writing coefficients to DB table '{RESULTS_TABLE}'...")

    engine = create_engine(DB_URL)

    # if_exists='replace' for V1. Later you can switch to 'append' and add keys.
    coef_df.to_sql(RESULTS_TABLE, engine, if_exists="replace", index=False)

    print("Write complete.")


# -----------------------------------------
# Main entry point
# -----------------------------------------

def main():
    product_cfg, sales, media = load_data()
    df = build_model_frame(product_cfg, sales, media)
    coef_df = fit_global_glm(df)
    write_coefficients_to_db(coef_df)


if __name__ == "__main__":
    main()