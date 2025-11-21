import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

API_KEYS = {
    "fred": os.getenv("FRED_API_KEY"),
    "te": os.getenv("TRADING_ECONOMICS_KEY"),
    "polygon": os.getenv("POLYGON_KEY"),
    "finnhub": os.getenv("FINNHUB_KEY"),
    "fmp": os.getenv("FMP_KEY"),
    "bot_tokens": {
        "policy_rate": os.getenv("BOT_POLICY_RATE_TOKEN"),
        "statistics": os.getenv("BOT_STATISTICS_TOKEN"),
        "exchange_rates": os.getenv("BOT_EXCHANGE_RATES_TOKEN"),
    },
}

PG_CONFIG = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "database": os.getenv("PG_DB"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASS"),
}

DEFAULT_START_DATE = os.getenv("DEFAULT_START_DATE", "2020-01-01")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
SQLALCHEMY_ECHO = os.getenv("SQLALCHEMY_ECHO", "0") == "1"

__all__ = [
    "API_KEYS",
    "PG_CONFIG",
    "DEFAULT_START_DATE",
    "REQUEST_TIMEOUT",
    "SQLALCHEMY_ECHO",
    "BASE_DIR",
]