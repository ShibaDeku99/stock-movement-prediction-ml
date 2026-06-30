import logging
import pandas as pd
from pathlib import Path
import yfinance as yf

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_stock_data(ticker: str = "AAPL", start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Download stock price data from Yahoo Finance.
    """
    logging.info(f"Downloading data for {ticker} from {start} to {end or 'today'}...")
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
            
        # Xử lý lỗi MultiIndex column thường gặp ở yfinance bản mới
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        logging.info(f"Successfully downloaded {len(df)} rows of data.")
        
        return df
        
    except Exception as e:
        logging.error(f"Error downloading data for {ticker}: {e}")
        raise


def save_raw_data(df: pd.DataFrame, ticker: str):
    """
    Save raw data to data/raw folder.
    """
    # Trỏ đường dẫn tương đối từ file đang chạy về thư mục gốc của project
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{ticker}_raw.csv"
    df.to_csv(output_path, index=False)

    logging.info(f"Saved data to: {output_path}")


if __name__ == "__main__":
    ticker = "AAPL"

    df = download_stock_data(
        ticker=ticker,
        start="2015-01-01"
    )

    print(df.head())
    print(df.tail())
    print(df.shape)

    save_raw_data(df, ticker)