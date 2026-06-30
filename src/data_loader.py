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


def download_market_data(start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Tải dữ liệu thị trường chung: SPY (S&P 500 ETF), QQQ (Nasdaq 100 ETF), ^VIX (Chỉ số biến động).
    Trả về DataFrame gộp theo ngày, với các cột đã được prefix theo ticker.
    """
    market_tickers = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "VIX": "^VIX"
    }
    
    merged = None
    
    for label, ticker in market_tickers.items():
        logging.info(f"Downloading market data: {label} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            
            if df.empty:
                logging.warning(f"No data found for {label} ({ticker}), skipping.")
                continue
            
            # Xử lý MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.reset_index(inplace=True)
            
            # Chỉ lấy cột Close và Volume, rename thêm prefix
            market_df = df[["Date", "Close", "Volume"]].copy()
            market_df = market_df.rename(columns={
                "Close": f"{label}_Close",
                "Volume": f"{label}_Volume"
            })
            
            if merged is None:
                merged = market_df
            else:
                merged = pd.merge(merged, market_df, on="Date", how="inner")
                
            logging.info(f"{label}: {len(market_df)} rows downloaded.")
            
        except Exception as e:
            logging.error(f"Error downloading {label}: {e}")
    
    if merged is None:
        raise ValueError("Failed to download any market data.")
    
    logging.info(f"Market data merged: {merged.shape}")
    return merged


def download_earnings_dates(ticker: str) -> pd.DataFrame:
    """
    Tải lịch công bố kết quả kinh doanh (Earnings) từ yfinance.
    Trả về DataFrame chứa các ngày earnings.
    """
    logging.info(f"Downloading earnings dates for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        
        # Lấy earnings dates từ yfinance
        earnings = stock.get_earnings_dates(limit=100)
        
        if earnings is None or earnings.empty:
            logging.warning(f"No earnings data found for {ticker}.")
            return pd.DataFrame(columns=['Earnings_Date'])
        
        earnings_df = pd.DataFrame()
        earnings_df['Earnings_Date'] = pd.to_datetime(earnings.index).normalize()
        earnings_df = earnings_df.drop_duplicates().sort_values('Earnings_Date').reset_index(drop=True)
        
        logging.info(f"Found {len(earnings_df)} earnings dates for {ticker}.")
        return earnings_df
        
    except Exception as e:
        logging.error(f"Error downloading earnings dates: {e}")
        return pd.DataFrame(columns=['Earnings_Date'])


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
    start_date = "2015-01-01"

    # 1. Tải dữ liệu cổ phiếu chính
    df = download_stock_data(ticker=ticker, start=start_date)
    print(df.head())
    print(df.shape)
    save_raw_data(df, ticker)
    
    # 2. Tải dữ liệu thị trường chung (SPY, QQQ, VIX)
    market_df = download_market_data(start=start_date)
    print(market_df.head())
    print(market_df.shape)
    save_raw_data(market_df, "MARKET")
    
    # 3. Tải lịch Earnings
    earnings_df = download_earnings_dates(ticker)
    if not earnings_df.empty:
        save_raw_data(earnings_df, f"{ticker}_earnings")
        print(f"\nEarnings dates: {len(earnings_df)} records")
        print(earnings_df.head())