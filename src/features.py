import logging
import pandas as pd
import numpy as np
from pathlib import Path
from ta import add_all_ta_features

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# CONSTANTS
# ============================================================
FORWARD_DAYS = 5          # Số ngày nhìn về phía trước để tính Target
THRESHOLD_PCT = 0.015     # Ngưỡng tăng tối thiểu (1.5%) để xếp vào class 1


def generate_features(df: pd.DataFrame, market_df: pd.DataFrame = None, earnings_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Tạo các đặc trưng từ dữ liệu giá thô, bao gồm:
    - Technical indicators (thư viện ta)
    - Features chất lượng cao (Momentum, Volatility, Volume Profile)
    - Dữ liệu thị trường chung (SPY, QQQ, VIX) nếu có
    - Target V2: giá 5 ngày tới có tăng đủ mạnh không
    """
    logging.info("Starting feature generation V2...")

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # ============================================================
    # 1. TARGET V2: Giá 5 ngày tới tăng >= 1.5% ?
    # ============================================================
    df["Future_Return"] = df["Close"].shift(-FORWARD_DAYS) / df["Close"] - 1
    df["Target"] = (df["Future_Return"] >= THRESHOLD_PCT).astype(float)
    # 5 dòng cuối cùng chưa có tương lai -> NaN
    df.loc[df.index[-FORWARD_DAYS:], "Target"] = np.nan
    
    logging.info(f"Target V2: 1 nếu giá {FORWARD_DAYS} ngày tới tăng >= {THRESHOLD_PCT*100:.1f}%")

    # ============================================================
    # 2. BASIC RETURN FEATURES
    # ============================================================
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # ============================================================
    # 3. FEATURES CHẤT LƯỢNG CAO
    # ============================================================
    logging.info("Calculating high-quality custom features...")
    
    # --- Momentum Features ---
    for period in [5, 10, 20]:
        df[f"Momentum_{period}d"] = df["Close"].pct_change(period)
    
    # --- Volatility Features ---
    for period in [10, 20]:
        df[f"Volatility_{period}d"] = df["Daily_Return"].rolling(window=period).std()
    
    # --- Volume Profile ---
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
    
    # --- Price Position (Giá hiện tại nằm ở đâu trong khoảng High-Low gần đây) ---
    for period in [20, 50]:
        rolling_high = df["High"].rolling(window=period).max()
        rolling_low = df["Low"].rolling(window=period).min()
        df[f"Price_Position_{period}d"] = (df["Close"] - rolling_low) / (rolling_high - rolling_low + 1e-10)
    
    # --- Gap Feature (Khoảng cách giữa Open hôm nay và Close hôm qua) ---
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    
    # --- Intraday Range (Biên độ giao dịch trong ngày) ---
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]
    
    # --- Volume Spike (Đột biến khối lượng: Volume > 2x trung bình 20 ngày) ---
    df["Volume_Spike"] = (df["Volume"] > 2 * df["Volume_SMA_20"]).astype(int)
    df["Volume_Spike_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] * 2 + 1e-10)  # Mức độ spike
    
    # --- Volatility 5D (bổ sung cho đủ bộ: 5d, 10d, 20d) ---
    df["Volatility_5d"] = df["Daily_Return"].rolling(window=5).std()
    
    # --- ATR thủ công (Average True Range) ---
    # ATR = trung bình của True Range trong N ngày
    # True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    df["True_Range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = df["True_Range"].rolling(window=14).mean()
    df["ATR_Ratio"] = df["True_Range"] / (df["ATR_14"] + 1e-10)  # Hôm nay biến động hơn hay kém ATR trung bình

    # ============================================================
    # 4. MERGE DỮ LIỆU THỊ TRƯỜNG CHUNG (SPY, QQQ, VIX)
    # ============================================================
    if market_df is not None:
        logging.info("Merging market context data (SPY, QQQ, VIX)...")
        market_df = market_df.copy()
        market_df["Date"] = pd.to_datetime(market_df["Date"])
        
        df = pd.merge(df, market_df, on="Date", how="left")
        
        # --- Relative Strength vs SPY (AAPL mạnh/yếu hơn thị trường chung) ---
        if "SPY_Close" in df.columns:
            df["SPY_Return"] = df["SPY_Close"].pct_change()
            df["Relative_Strength_SPY"] = df["Daily_Return"] - df["SPY_Return"]
            
            for period in [5, 10, 20]:
                df[f"RS_SPY_{period}d"] = df["Relative_Strength_SPY"].rolling(window=period).sum()
        
        # --- Relative Strength vs QQQ ---
        if "QQQ_Close" in df.columns:
            df["QQQ_Return"] = df["QQQ_Close"].pct_change()
            df["Relative_Strength_QQQ"] = df["Daily_Return"] - df["QQQ_Return"]
        
        # --- VIX Features (Biến động thị trường) ---
        if "VIX_Close" in df.columns:
            df["VIX_Change"] = df["VIX_Close"].pct_change()
            df["VIX_SMA_20"] = df["VIX_Close"].rolling(window=20).mean()
            df["VIX_Ratio"] = df["VIX_Close"] / df["VIX_SMA_20"]
            # VIX Level bins: thấp(<15), trung bình(15-25), cao(25-35), cực cao(>35)
            df["VIX_Level"] = pd.cut(
                df["VIX_Close"],
                bins=[0, 15, 25, 35, 100],
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        # --- Market Sentiment Proxy ---
        # Tạo chỉ số tâm lý thị trường tổng hợp từ: VIX + SPY momentum + Volume
        # Score cao = thị trường lạc quan (Greed), thấp = thị trường sợ hãi (Fear)
        logging.info("Calculating Market Sentiment Proxy...")
        sentiment_components = []
        
        if "VIX_Close" in df.columns:
            # VIX thấp = lạc quan (+), VIX cao = sợ hãi (-)
            vix_norm = 1 - (df["VIX_Close"] - df["VIX_Close"].rolling(60).min()) / \
                       (df["VIX_Close"].rolling(60).max() - df["VIX_Close"].rolling(60).min() + 1e-10)
            sentiment_components.append(vix_norm)
        
        if "SPY_Close" in df.columns:
            # SPY momentum 20d dương = lạc quan
            spy_mom = df["SPY_Close"].pct_change(20)
            spy_norm = (spy_mom - spy_mom.rolling(60).min()) / \
                       (spy_mom.rolling(60).max() - spy_mom.rolling(60).min() + 1e-10)
            sentiment_components.append(spy_norm)
        
        # Volume trend: Volume cao + giá tăng = Greed
        vol_price = df["Volume_Ratio"] * np.sign(df["Daily_Return"])
        vol_norm = (vol_price - vol_price.rolling(60).min()) / \
                   (vol_price.rolling(60).max() - vol_price.rolling(60).min() + 1e-10)
        sentiment_components.append(vol_norm)
        
        if sentiment_components:
            df["Sentiment_Proxy"] = pd.concat(sentiment_components, axis=1).mean(axis=1)
            # Smooth bằng SMA 5 ngày để giảm nhiễu
            df["Sentiment_Proxy_SMA5"] = df["Sentiment_Proxy"].rolling(5).mean()
        
        logging.info("Market context features added successfully.")
    else:
        logging.warning("No market data provided. Skipping market context features.")

    # ============================================================
    # 4B. EARNINGS WEEK (Sự kiện công bố KQKD)
    # ============================================================
    if earnings_df is not None and not earnings_df.empty:
        logging.info("Adding Earnings Week features...")
        earnings_df = earnings_df.copy()
        earnings_df["Earnings_Date"] = pd.to_datetime(earnings_df["Earnings_Date"], utc=True).dt.tz_localize(None).dt.normalize()
        earnings_dates = set(earnings_df["Earnings_Date"])
        
        # Earnings_Week: 1 nếu ngày hiện tại nằm trong khoảng [-5, +2] ngày so với ngày earnings
        # (Thị trường thường biến động mạnh trước và sau ngày công bố)
        def is_near_earnings(date):
            for ed in earnings_dates:
                diff = (date - ed).days
                if -5 <= diff <= 2:
                    return 1
            return 0
        
        df["Earnings_Week"] = df["Date"].apply(is_near_earnings)
        
        # Earnings_Distance: Số ngày còn lại đến ngày earnings gần nhất (tương lai)
        sorted_earnings = sorted(earnings_dates)
        def days_to_next_earnings(date):
            for ed in sorted_earnings:
                diff = (ed - date).days
                if diff >= 0:
                    return min(diff, 90)  # Cap tại 90 ngày
            return 90
        
        df["Earnings_Distance"] = df["Date"].apply(days_to_next_earnings)
        logging.info(f"Earnings features added. {df['Earnings_Week'].sum()} trading days near earnings.")
    else:
        logging.warning("No earnings data provided. Skipping earnings features.")

    # ============================================================
    # 5. TECHNICAL INDICATORS (thư viện ta)
    # ============================================================
    logging.info("Calculating technical indicators...")

    try:
        df = add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=False
        )
        logging.info("Successfully added technical indicators.")
    except Exception as e:
        logging.error(f"Error adding TA features: {e}")

        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # ============================================================
    # 6. CLEANUP: Xử lý inf, NaN, cột thừa
    # ============================================================
    original_shape = df.shape
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Parabolic SAR up/down là 2 cột loại trừ lẫn nhau
    psar_cols = ['trend_psar_up', 'trend_psar_down']
    df = df.drop(columns=[col for col in psar_cols if col in df.columns])
    
    # Drop cột trung gian không cần thiết cho training
    drop_cols = ["Future_Return"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    df = df.dropna().reset_index(drop=True)

    logging.info(f"Cleanup done. Shape changed from {original_shape} to {df.shape}")
    
    # Log phân phối target
    if "Target" in df.columns:
        target_dist = df["Target"].value_counts(normalize=True)
        logging.info(f"Target distribution:\n{target_dist.to_string()}")

    return df


def save_features(df: pd.DataFrame, ticker: str):
    """
    Lưu dữ liệu đã xử lý vào thư mục data/processed.
    """
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{ticker}_features.csv"
    df.to_csv(output_path, index=False)
    
    logging.info(f"Saved processed features to: {output_path}")


if __name__ == "__main__":
    ticker = "AAPL"
    
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / f"{ticker}_raw.csv"
    market_path = project_root / "data" / "raw" / "MARKET_raw.csv"
    
    if not raw_path.exists():
        logging.error(f"Raw data not found at {raw_path}. Run data_loader.py first.")
    else:
        logging.info(f"Loading raw data from {raw_path}")
        df_raw = pd.read_csv(raw_path)
        
        # Tải market data nếu có
        market_df = None
        if market_path.exists():
            logging.info(f"Loading market data from {market_path}")
            market_df = pd.read_csv(market_path)
        else:
            logging.warning("Market data not found. Run data_loader.py to download SPY/QQQ/VIX.")
        
        # Tải earnings data nếu có
        earnings_df = None
        earnings_path = project_root / "data" / "raw" / f"{ticker}_earnings_raw.csv"
        if earnings_path.exists():
            logging.info(f"Loading earnings data from {earnings_path}")
            earnings_df = pd.read_csv(earnings_path)
        else:
            logging.warning("Earnings data not found. Run data_loader.py to download.")
        
        # Tạo features
        df_features = generate_features(df_raw, market_df=market_df, earnings_df=earnings_df)
        
        # Lưu file
        save_features(df_features, ticker)
        
        # Hiển thị tóm tắt
        print(f"\nTotal features: {df_features.shape[1]} columns, {df_features.shape[0]} rows")
        print(f"\nFeature Preview (First 5 rows):")
        print(df_features[['Date', 'Close', 'Target', 'Daily_Return']].head())
