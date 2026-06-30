import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# CONSTANTS
# ============================================================
TRANSACTION_COST = 0.001  # 0.1% phí mỗi lần thay đổi vị thế (mua/bán)


def find_best_model(ticker: str):
    """Tìm file model, feature_columns và threshold đã lưu."""
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models"

    model_files = list(model_dir.glob(f"{ticker}_best_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("Không tìm thấy mô hình. Hãy chạy train.py trước.")

    model_path = model_files[0]
    feature_cols_path = model_dir / f"{ticker}_feature_columns.pkl"
    threshold_path = model_dir / f"{ticker}_threshold.pkl"

    return model_path, feature_cols_path, threshold_path


def load_backtest_data(ticker: str):
    """Tải dữ liệu để backtest."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / f"{ticker}_features.csv"

    if not data_path.exists():
        raise FileNotFoundError("Không tìm thấy dữ liệu. Hãy chạy features.py trước.")

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


# ============================================================
# TRADE LOG: Phân tích chi tiết từng lệnh
# ============================================================
def generate_trade_log(test_df: pd.DataFrame):
    """
    Tạo trade log chi tiết: ngày vào lệnh, ngày ra lệnh, lợi nhuận từng lệnh.
    Một "trade" bắt đầu khi Signal chuyển từ 0->1 và kết thúc khi chuyển từ 1->0.
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None

    for i, row in test_df.iterrows():
        signal = row['Signal_Binary']

        if signal == 1 and not in_trade:
            # Vào lệnh
            in_trade = True
            entry_date = row['Date']
            entry_price = row['Close']

        elif signal == 0 and in_trade:
            # Ra lệnh
            in_trade = False
            exit_date = row['Date']
            exit_price = row['Close']
            pnl_pct = (exit_price - entry_price) / entry_price
            # Trừ phí giao dịch 2 chiều (mua + bán)
            pnl_pct_net = pnl_pct - (TRANSACTION_COST * 2)
            holding_days = (exit_date - entry_date).days

            trades.append({
                "Entry_Date": entry_date.strftime('%Y-%m-%d'),
                "Exit_Date": exit_date.strftime('%Y-%m-%d'),
                "Entry_Price": round(entry_price, 2),
                "Exit_Price": round(exit_price, 2),
                "PnL_%": round(pnl_pct * 100, 2),
                "PnL_Net_%": round(pnl_pct_net * 100, 2),
                "Holding_Days": holding_days,
                "Result": "WIN" if pnl_pct_net > 0 else "LOSS"
            })

    # Nếu đang giữ lệnh ở cuối giai đoạn test
    if in_trade:
        last_row = test_df.iloc[-1]
        pnl_pct = (last_row['Close'] - entry_price) / entry_price
        pnl_pct_net = pnl_pct - TRANSACTION_COST  # Chỉ trừ phí 1 chiều vì chưa bán
        trades.append({
            "Entry_Date": entry_date.strftime('%Y-%m-%d'),
            "Exit_Date": last_row['Date'].strftime('%Y-%m-%d') + " (OPEN)",
            "Entry_Price": round(entry_price, 2),
            "Exit_Price": round(last_row['Close'], 2),
            "PnL_%": round(pnl_pct * 100, 2),
            "PnL_Net_%": round(pnl_pct_net * 100, 2),
            "Holding_Days": (last_row['Date'] - entry_date).days,
            "Result": "WIN" if pnl_pct_net > 0 else "LOSS"
        })

    return pd.DataFrame(trades)


# ============================================================
# PERFORMANCE METRICS: Win Rate, Profit Factor, etc.
# ============================================================
def calculate_performance_metrics(trade_log_df: pd.DataFrame):
    """Tính các chỉ số hiệu suất giao dịch chi tiết."""
    if trade_log_df.empty:
        logging.warning("Không có lệnh nào được thực hiện.")
        return {}

    total_trades = len(trade_log_df)
    wins = trade_log_df[trade_log_df['Result'] == 'WIN']
    losses = trade_log_df[trade_log_df['Result'] == 'LOSS']

    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = wins['PnL_Net_%'].mean() if len(wins) > 0 else 0
    avg_loss = losses['PnL_Net_%'].mean() if len(losses) > 0 else 0
    best_trade = trade_log_df['PnL_Net_%'].max()
    worst_trade = trade_log_df['PnL_Net_%'].min()
    avg_holding = trade_log_df['Holding_Days'].mean()

    # Profit Factor = Tổng lãi / Tổng lỗ (tuyệt đối)
    gross_profit = wins['PnL_Net_%'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['PnL_Net_%'].sum()) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    metrics = {
        "Total Trades": total_trades,
        "Win Rate": f"{win_rate * 100:.1f}%",
        "Avg Win": f"{avg_win:.2f}%",
        "Avg Loss": f"{avg_loss:.2f}%",
        "Best Trade": f"{best_trade:.2f}%",
        "Worst Trade": f"{worst_trade:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}",
        "Avg Holding Days": f"{avg_holding:.1f}",
    }

    return metrics


# ============================================================
# MONTHLY RETURNS: Bảng lãi/lỗ theo tháng
# ============================================================
def calculate_monthly_returns(test_df: pd.DataFrame, return_col: str = 'Strategy_Binary_Return'):
    """Tính lợi nhuận hàng tháng cho heatmap."""
    df = test_df[['Date', return_col]].copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    monthly = df.groupby(['Year', 'Month'])[return_col].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    monthly.columns = ['Year', 'Month', 'Return']

    pivot = monthly.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    return pivot


# ============================================================
# MAIN BACKTEST
# ============================================================
def run_backtest(ticker: str, test_size: float = 0.2, initial_capital: float = 10000.0):
    logging.info(f"=== BACKTEST V3 (PRODUCTION-GRADE) CHO {ticker} ===")

    # 1. Tải dữ liệu và model
    df = load_backtest_data(ticker)
    model_path, feature_cols_path, threshold_path = find_best_model(ticker)

    logging.info(f"Sử dụng mô hình: {model_path.name}")
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)

    # Tải threshold tối ưu
    if threshold_path.exists():
        threshold = joblib.load(threshold_path)
        logging.info(f"Threshold tối ưu: {threshold}")
    else:
        threshold = 0.5
        logging.warning("Không tìm thấy threshold, dùng mặc định 0.5")

    # 2. Cắt lấy tập Test (Out-of-sample)
    split_idx = int(len(df) * (1 - test_size))
    test_df = df.iloc[split_idx:].copy()
    X_test = test_df[feature_cols]

    # 3. Sinh xác suất dự đoán
    y_proba = model.predict_proba(X_test)[:, 1]
    test_df['Probability'] = y_proba
    test_df['Signal_Binary'] = (y_proba >= threshold).astype(int)

    # Position Sizing theo xác suất (Aggressive Scaling nhưng có Cap)
    # Vì xác suất của Logistic Regression hiếm khi vượt 0.5-0.6,
    # ta phải scale mạnh tay thay vì đợi đến 1.0 mới dùng 100% vốn.
    # Tuy nhiên, để đúng bản chất "phân bổ vốn thận trọng", ta không bao giờ all-in.
    # - proba >= threshold + 0.1: 70% vốn (Luôn giữ 30% tiền mặt phòng rủi ro)
    # - proba >= threshold: 50% -> 70% vốn
    # - proba < threshold: 0% vốn
    
    test_df['Position_Size'] = np.where(
        y_proba >= threshold + 0.1, 
        0.7,
        np.where(
            y_proba >= threshold,
            0.5 + 0.5 * ((y_proba - threshold) / 0.1),
            0.0
        )
    ).clip(0, 0.7)

    # 4. Tính lợi nhuận (có tính phí giao dịch)
    # Phí phát sinh khi thay đổi vị thế
    test_df['Prev_Signal'] = test_df['Signal_Binary'].shift(1).fillna(0)
    test_df['Signal_Changed'] = (test_df['Signal_Binary'] != test_df['Prev_Signal']).astype(int)

    test_df['Prev_Position'] = test_df['Position_Size'].shift(1).fillna(0)
    test_df['Position_Diff'] = (test_df['Position_Size'] - test_df['Prev_Position']).abs()

    # Binary Strategy: Return - phí khi thay đổi vị thế
    test_df['Strategy_Binary_Return'] = (
        test_df['Signal_Binary'].shift(1) * test_df['Daily_Return']
        - test_df['Signal_Changed'] * TRANSACTION_COST
    )
    # Proba Strategy: Return theo position size - phí (dựa trên mức độ thay đổi vị thế)
    test_df['Strategy_Proba_Return'] = (
        test_df['Position_Size'].shift(1) * test_df['Daily_Return']
        - test_df['Position_Diff'] * TRANSACTION_COST
    )
    test_df['BnH_Return'] = test_df['Daily_Return']

    # Bỏ dòng đầu bị NaN
    test_df = test_df.dropna(subset=['Strategy_Binary_Return', 'Strategy_Proba_Return']).copy()

    # 5. Tính lũy kế vốn
    test_df['Capital_Binary'] = initial_capital * (1 + test_df['Strategy_Binary_Return']).cumprod()
    test_df['Capital_Proba'] = initial_capital * (1 + test_df['Strategy_Proba_Return']).cumprod()
    test_df['Capital_BnH'] = initial_capital * (1 + test_df['BnH_Return']).cumprod()

    # 6. Tổng hợp chỉ số rủi ro / lợi nhuận
    strategies = {
        "Binary Signal": ("Capital_Binary", "Strategy_Binary_Return"),
        "Proba Sizing": ("Capital_Proba", "Strategy_Proba_Return"),
        "Buy & Hold": ("Capital_BnH", "BnH_Return"),
    }

    logging.info(f"\n{'='*60}")
    logging.info(f"KẾT QUẢ BACKTEST V3 (OUT-OF-SAMPLE)")
    logging.info(f"Giai đoạn: {test_df['Date'].iloc[0].date()} → {test_df['Date'].iloc[-1].date()}")
    logging.info(f"Threshold: {threshold} | Phí GD: {TRANSACTION_COST*100:.1f}%/lệnh")
    logging.info(f"{'='*60}")

    summary_rows = []
    for strat_name, (cap_col, ret_col) in strategies.items():
        total_return = (test_df[cap_col].iloc[-1] / initial_capital) - 1
        final_capital = test_df[cap_col].iloc[-1]

        # Max Drawdown
        peak = test_df[cap_col].cummax()
        drawdown = (test_df[cap_col] - peak) / peak
        max_dd = drawdown.min()

        # Sharpe Ratio (annualized)
        daily_returns = test_df[ret_col]
        sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-10)) * np.sqrt(252)

        # Calmar Ratio = Annual Return / Max Drawdown
        annual_return = (1 + total_return) ** (252 / len(test_df)) - 1
        calmar = abs(annual_return / (max_dd + 1e-10))

        # Exposure Time & Number of Trades
        if strat_name == "Buy & Hold":
            exposure_time = 100.0
            n_trades = 1
        elif strat_name == "Binary Signal":
            exposure_time = (test_df['Signal_Binary'] > 0).mean() * 100
            n_trades = (test_df['Signal_Binary'] > test_df['Signal_Binary'].shift(1).fillna(0)).sum()
        else: # Proba Sizing
            exposure_time = (test_df['Position_Size'] > 0).mean() * 100
            n_trades = ((test_df['Position_Size'] > 0) & (test_df['Position_Size'].shift(1).fillna(0) == 0)).sum()

        summary_rows.append({
            "Strategy": strat_name,
            "Final Value": f"${final_capital:,.0f}",
            "Total Return": f"{total_return * 100:.2f}%",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Number of Trades": int(n_trades),
            "Exposure Time": f"{exposure_time:.1f}%",
        })

    summary_df = pd.DataFrame(summary_rows)
    logging.info("\n" + summary_df.to_string(index=False))

    # 7. Trade Log & Performance Metrics (cho Binary Signal)
    logging.info(f"\n{'='*60}")
    logging.info("CHI TIẾT GIAO DỊCH (Binary Signal Strategy)")
    logging.info(f"{'='*60}")

    trade_log_df = generate_trade_log(test_df)
    if not trade_log_df.empty:
        logging.info(f"\nTrade Log (Top 10 lệnh gần nhất):\n{trade_log_df.tail(10).to_string(index=False)}")

        perf_metrics = calculate_performance_metrics(trade_log_df)
        logging.info("\nPerformance Metrics:")
        for k, v in perf_metrics.items():
            logging.info(f"  {k}: {v}")

    # 8. Monthly Returns
    monthly_returns = calculate_monthly_returns(test_df)
    logging.info(f"\nMonthly Returns (%):\n{(monthly_returns * 100).round(2).to_string()}")

    # 9. Lưu tất cả reports
    project_root = Path(__file__).resolve().parent.parent
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(report_dir / f"{ticker}_backtest_summary.csv", index=False)
    trade_log_df.to_csv(report_dir / f"{ticker}_trade_log.csv", index=False)
    (monthly_returns * 100).round(2).to_csv(report_dir / f"{ticker}_monthly_returns.csv")

    logging.info(f"\nĐã lưu: backtest_summary.csv, trade_log.csv, monthly_returns.csv")

    # 10. Vẽ biểu đồ
    plot_backtest(test_df, ticker, threshold, monthly_returns)

    return summary_df, trade_log_df


# ============================================================
# VISUALIZATION
# ============================================================
def plot_backtest(test_df: pd.DataFrame, ticker: str, threshold: float, monthly_returns: pd.DataFrame):
    project_root = Path(__file__).resolve().parent.parent
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 1.2, 1.2]})

    # --- Chart 1: Equity Curve ---
    ax1 = axes[0]
    ax1.plot(test_df['Date'], test_df['Capital_Binary'],
             label=f'Binary Signal (thr={threshold})', linewidth=2, color='#2196F3')
    ax1.plot(test_df['Date'], test_df['Capital_Proba'],
             label='Proba Sizing', linewidth=2, color='#4CAF50')
    ax1.plot(test_df['Date'], test_df['Capital_BnH'],
             label='Buy & Hold', linewidth=1.5, color='gray', alpha=0.6, linestyle='--')

    ax1.set_title(f'Backtest V3 — {ticker} (incl. {TRANSACTION_COST*100:.1f}% transaction cost)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vốn ($)')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # --- Chart 2: Drawdown ---
    ax2 = axes[1]
    peak_binary = test_df['Capital_Binary'].cummax()
    dd_binary = (test_df['Capital_Binary'] - peak_binary) / peak_binary
    peak_bnh = test_df['Capital_BnH'].cummax()
    dd_bnh = (test_df['Capital_BnH'] - peak_bnh) / peak_bnh

    ax2.fill_between(test_df['Date'], dd_binary * 100, alpha=0.4, color='#2196F3', label='Binary Signal DD')
    ax2.fill_between(test_df['Date'], dd_bnh * 100, alpha=0.3, color='gray', label='Buy & Hold DD')
    ax2.set_title('Drawdown (%)', fontsize=11)
    ax2.set_ylabel('%')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Chart 3: Position Size ---
    ax3 = axes[2]
    ax3.fill_between(test_df['Date'], test_df['Position_Size'] * 100,
                     alpha=0.4, color='#4CAF50', label='Position Size')
    ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% vốn')
    ax3.set_title('Position Size — Phân bổ vốn theo xác suất', fontsize=11)
    ax3.set_ylabel('% Vốn')
    ax3.set_xlabel('Thời gian')
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.set_ylim(-5, 110)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    plot_path = report_dir / f"{ticker}_backtest_chart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Biểu đồ backtest đã được lưu tại: {plot_path}")

    # --- Heatmap Monthly Returns ---
    fig2, ax_heat = plt.subplots(figsize=(12, 4))
    data = (monthly_returns * 100).fillna(0)
    im = ax_heat.imshow(data.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

    ax_heat.set_xticks(range(len(data.columns)))
    ax_heat.set_xticklabels(data.columns)
    ax_heat.set_yticks(range(len(data.index)))
    ax_heat.set_yticklabels(data.index)

    # Thêm số vào ô
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            if not np.isnan(val) and val != 0:
                color = 'white' if abs(val) > 5 else 'black'
                ax_heat.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=color)

    ax_heat.set_title(f'Monthly Returns Heatmap (%) — {ticker}', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax_heat, label='Return %')
    plt.tight_layout()

    heatmap_path = report_dir / f"{ticker}_monthly_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Monthly heatmap đã được lưu tại: {heatmap_path}")


if __name__ == "__main__":
    run_backtest("AAPL")
