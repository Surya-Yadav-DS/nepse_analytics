"""
╔══════════════════════════════════════════════════════════════════╗
║          NEPSE STOCK ANALYSER & PREDICTOR  v2.0                 ║
║          Fresh build — optimally structured                     ║
╠══════════════════════════════════════════════════════════════════╣
║  FOLDER STRUCTURE:                                              ║
║  Training_Data/   → 500 days of historical CSVs (load once)    ║
║  New_Data/        → Drop new CSVs here after market close       ║
║                     Auto-detected on every app load             ║
║                                                                  ║
║  CSV NAMING: "Today's Price - 2026-03-23.csv"  (date in name)  ║
╠══════════════════════════════════════════════════════════════════╣
║  RUN:   streamlit run project.py                                ║
║  INSTALL: pip install streamlit pandas numpy matplotlib         ║
║           scikit-learn xgboost                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, glob, re, warnings, datetime, sqlite3, io, hashlib, time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ══════════════════════════════════════════════════════════════════
#  PATHS  — only change these
# ══════════════════════════════════════════════════════════════════
BASE = os.path.dirname(__file__)
#BASE          = r"C:\OneDrive\Desktop\nepse-stock-predictor"
TRAINING_DIR  = os.path.join(BASE, "Training_Data")   # 500 days CSVs
NEW_DATA_DIR  = os.path.join(BASE, "New_Data")          # daily drop folder
DB_PATH       = os.path.join(BASE, "nepse.db")          # auto-created

for d in [TRAINING_DIR, NEW_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  NEPSE CIRCUIT BREAKER
#  Max allowed price move in a single day = ±10%
#  Model predictions are clamped to this range to stay realistic
# ══════════════════════════════════════════════════════════════════
CIRCUIT_LIMIT = 0.10          # 10% max daily move
MIN_ROWS      = 10            # minimum sessions needed to train

# ══════════════════════════════════════════════════════════════════
#  NEPAL TRADING CALENDAR
#  NEPSE trades Monday - Friday . Saturday & Sunday are off.
# ══════════════════════════════════════════════════════════════════
HOLIDAYS = {
    # Add Nepal public holidays here (NEPSE closes these days)
    datetime.date(2026, 1, 11),
    datetime.date(2026, 2, 18),
    datetime.date(2026, 4, 14),
    datetime.date(2026, 5, 28),
    datetime.date(2026, 9, 19),
    datetime.date(2026, 10, 2),
    datetime.date(2026, 10, 24),
}

def is_trading_day(d: datetime.date) -> bool:
    return d.weekday() in {0, 1, 2, 3, 6} and d not in HOLIDAYS

def next_n_trading_days(from_date: datetime.date, n: int) -> list:
    days, cur = [], from_date
    while len(days) < n:
        cur += datetime.timedelta(days=1)
        if is_trading_day(cur):
            days.append(cur)
    return days

# ══════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════
def conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker       TEXT,
                date         TEXT,
                open         REAL DEFAULT 0,
                high         REAL DEFAULT 0,
                low          REAL DEFAULT 0,
                close        REAL NOT NULL,
                volume       REAL DEFAULT 0,
                trades       REAL DEFAULT 0,
                prev_close   REAL DEFAULT 0,
                source       TEXT DEFAULT 'csv',
                PRIMARY KEY (ticker, date)
            );
            CREATE TABLE IF NOT EXISTS ingested (
                file_hash TEXT PRIMARY KEY,
                filename  TEXT,
                records   INTEGER,
                loaded_at TEXT
            );
            CREATE INDEX IF NOT EXISTS ix_ticker ON prices(ticker);
            CREATE INDEX IF NOT EXISTS ix_date   ON prices(date);
        """)

def already_ingested(file_hash: str) -> bool:
    with conn() as c:
        return c.execute("SELECT 1 FROM ingested WHERE file_hash=?",
                         (file_hash,)).fetchone() is not None

def mark_ingested(file_hash: str, filename: str, records: int):
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO ingested VALUES (?,?,?,?)",
                  (file_hash, filename, records,
                   datetime.datetime.now().isoformat()))

def bulk_insert(rows: list[dict], source: str = "csv") -> int:
    if not rows:
        return 0
    inserted = 0
    with conn() as c:
        for r in rows:
            try:
                c.execute("""
                    INSERT OR IGNORE INTO prices
                    (ticker,date,open,high,low,close,volume,trades,prev_close,source)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(r["ticker"]).strip().upper(),
                    str(r["date"])[:10],
                    float(r.get("open")       or 0),
                    float(r.get("high")       or 0),
                    float(r.get("low")        or 0),
                    float(r["close"]),
                    float(r.get("volume")     or 0),
                    float(r.get("trades")     or 0),
                    float(r.get("prev_close") or 0),
                    source,
                ))
                inserted += c.execute("SELECT changes()").fetchone()[0]
            except Exception:
                pass
    return inserted

def get_stock(ticker: str) -> pd.DataFrame:
    with conn() as c:
        df = pd.read_sql(
            "SELECT * FROM prices WHERE ticker=? ORDER BY date",
            c, params=(ticker,)
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume", "trades": "Trades",
        "prev_close": "Prev_Close",
    })

def all_tickers() -> list[str]:
    with conn() as c:
        return [r[0] for r in
                c.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker")]

def db_stats() -> dict:
    with conn() as c:
        total   = c.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        tickers = c.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
        min_d   = c.execute("SELECT MIN(date) FROM prices").fetchone()[0]
        max_d   = c.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        files   = c.execute("SELECT COUNT(*) FROM ingested").fetchone()[0]
    return dict(total=total, tickers=tickers, min_d=min_d, max_d=max_d, files=files)

def db_version(ticker: str) -> str:
    """Changes when new data is added → busts model cache automatically."""
    with conn() as c:
        r = c.execute(
            "SELECT COUNT(*), MAX(date) FROM prices WHERE ticker=?",
            (ticker,)
        ).fetchone()
    return f"{ticker}|{r[0]}|{r[1]}"

# ══════════════════════════════════════════════════════════════════
#  CSV PARSER  (nepalstock.com format)
#  Columns: Id, Business Date, Security Id, Symbol, Security Name,
#           Open Price, High Price, Low Price, Close Price,
#           Total Traded Quantity, Total Traded Value,
#           Previous Day Close Price, ...
# ══════════════════════════════════════════════════════════════════
def parse_csv(content: bytes, filename: str = "") -> tuple[list, str]:
    """
    Returns (rows, error_string).
    rows is a list of dicts ready for bulk_insert().
    error_string is empty on success.
    """
    try:
        text = content.decode("utf-8", errors="replace")
        df   = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        df.columns = df.columns.str.strip()

        # ── Detect & normalise columns ────────────────────────────
        if "Symbol" in df.columns and "Close Price" in df.columns:
            # nepalstock.com official format
            df = df.rename(columns={
                "Symbol":                  "ticker",
                "Business Date":           "date",
                "Open Price":              "open",
                "High Price":              "high",
                "Low Price":               "low",
                "Close Price":             "close",
                "Total Traded Quantity":   "volume",
                "Total Trades":            "trades",
                "Previous Day Close Price":"prev_close",
            })

        elif "Symbol" in df.columns and "LTP" in df.columns:
            # merolagani / sharesansar format
            df = df.rename(columns={
                "Symbol":         "ticker",
                "As Of":          "date",
                "LTP":            "close",
                "Open":           "open",
                "High":           "high",
                "Low":            "low",
                "Volume":         "volume",
                "Previous Close": "prev_close",
            })
            df["trades"] = df.get("No. of Transaction", 0)

        else:
            # Try case-insensitive fallback
            lc = {c: c.lower().strip() for c in df.columns}
            df = df.rename(columns=lc)
            rmap = {}
            for c in df.columns:
                if c == "symbol":                            rmap[c] = "ticker"
                elif "close" in c and "prev" not in c and "close" not in rmap.values():
                                                             rmap[c] = "close"
                elif c == "open":                            rmap[c] = "open"
                elif c == "high":                            rmap[c] = "high"
                elif c == "low":                             rmap[c] = "low"
                elif "quantity" in c or "volume" in c:      rmap[c] = "volume"
                elif "trade" in c and "value" not in c:     rmap[c] = "trades"
                elif "previous" in c or "prev" in c:        rmap[c] = "prev_close"
                elif "date" in c or "as of" in c:           rmap[c] = "date"
            df = df.rename(columns=rmap)
            if "ticker" not in df.columns or "close" not in df.columns:
                return [], (f"Cannot recognise CSV format.\n"
                            f"Columns found: {list(df.columns)}\n"
                            f"Expected 'Symbol' and 'Close Price' columns.")

        # ── Resolve date ──────────────────────────────────────────
        if "date" not in df.columns or df["date"].isna().all():
            m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
            if not m:
                return [], ("Cannot find date in CSV or filename. "
                            "Name your file like: Today's Price - 2026-04-10.csv")
            df["date"] = m.group(1)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])

        # ── Clean numerics ────────────────────────────────────────
        for col in ["open", "high", "low", "close", "volume", "trades", "prev_close"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce",
                ).fillna(0)
            )

        # ── Filter ────────────────────────────────────────────────
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df[df["close"] > 0]
        df = df[df["ticker"].str.match(r"^[A-Z0-9]{1,15}$", na=False)]
        df = df.dropna(subset=["ticker", "date", "close"])

        if df.empty:
            return [], "No valid rows after parsing. Check the CSV content."

        rows = df[["ticker", "date", "open", "high", "low",
                   "close", "volume", "trades", "prev_close"]].to_dict("records")
        return rows, ""

    except Exception as e:
        return [], f"Parse error: {e}"

# ══════════════════════════════════════════════════════════════════
#  INGEST ENGINE  — handles both Training_Data and New_Data
# ══════════════════════════════════════════════════════════════════
def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def ingest_folder(folder: str, source_tag: str,
                  progress=None, silent: bool = False) -> tuple[int, int]:
    """
    Scan folder for CSVs. Skip already-ingested files (by MD5 hash).
    Returns (new_files_count, new_records_count).
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    new_files, new_records = 0, 0

    for i, path in enumerate(files):
        fhash = file_hash(path)
        if already_ingested(fhash):
            if progress:
                progress.progress((i + 1) / max(len(files), 1))
            continue

        rows, err = parse_csv(open(path, "rb").read(), os.path.basename(path))
        if err:
            if not silent:
                st.warning(f"⚠ {os.path.basename(path)}: {err}")
            continue

        n = bulk_insert(rows, source_tag)
        mark_ingested(fhash, os.path.basename(path), n)
        new_files   += 1
        new_records += n

        if progress:
            progress.progress((i + 1) / max(len(files), 1))

    return new_files, new_records

# ══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
FEATURES = [
    "Lag_1","Lag_2","Lag_3","Lag_5","Lag_10",
    "MA_5","MA_10","MA_20","MA_50",
    "Std_5","Std_10","Std_20",
    "Return_1d","Return_3d","Return_5d","Return_10d",
    "Price_Range","Range_Ratio",
    "Vol_MA5_Ratio","RSI_14","RSI_7",
    "MACD","BB_pos","Momentum_10",
    "DayOfWeek","Month","Quarter",
    "Volume","Trades",
    "Hit_Upper_Circuit","Hit_Lower_Circuit","Return_1d_capped",
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date")

    # Fill gaps
    if "Prev_Close" not in df.columns:
        df["Prev_Close"] = df["Close"].shift(1)
    df["Prev_Close"] = df["Prev_Close"].ffill().fillna(df["Close"])
    df["High"] = df["High"].replace(0, np.nan).ffill().fillna(df["Close"])
    df["Low"]  = df["Low"].replace(0, np.nan).ffill().fillna(df["Close"])

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"Lag_{lag}"] = df["Close"].shift(lag)

    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f"MA_{w}"] = df["Close"].rolling(w, min_periods=1).mean()

    # Volatility
    for w in [5, 10, 20]:
        df[f"Std_{w}"] = df["Close"].rolling(w, min_periods=1).std().fillna(0)

    # Returns
    for w in [1, 3, 5, 10]:
        df[f"Return_{w}d"] = df["Close"].pct_change(w)

    # Price range
    df["Price_Range"] = df["High"] - df["Low"]
    df["Range_Ratio"] = df["Price_Range"] / df["Close"].replace(0, np.nan)

    # Volume ratio
    vol_ma5 = df["Volume"].rolling(5, min_periods=1).mean()
    df["Vol_MA5_Ratio"] = df["Volume"] / (vol_ma5 + 1e-9)

    # RSI (14 and 7)
    for period in [14, 7]:
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        df[f"RSI_{period}"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD (12-26 EMA diff)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # Bollinger Band position
    bb_mid = df["MA_20"]
    bb_std = df["Std_20"]
    df["BB_pos"] = (df["Close"] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)

    # Momentum
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    # Calendar
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"]     = df["Date"].dt.month
    df["Quarter"]   = (df["Date"].dt.month - 1) // 3 + 1

    # Circuit breaker: flag days that hit the ±10% limit
    df["Hit_Upper_Circuit"] = (df["Close"] >= df["Prev_Close"] * 1.095).astype(int)
    df["Hit_Lower_Circuit"] = (df["Close"] <= df["Prev_Close"] * 0.905).astype(int)

    # Daily return capped at circuit limit for cleaner training signal
    df["Return_1d_capped"] = df["Return_1d"].clip(-CIRCUIT_LIMIT, CIRCUIT_LIMIT)

    # Target — the next day close, also clipped at ±10% for training
    df["Target"] = df["Close"].shift(-1)
    # Clamp target to realistic circuit range (removes outlier labels)
    max_target = df["Close"] * (1 + CIRCUIT_LIMIT)
    min_target = df["Close"] * (1 - CIRCUIT_LIMIT)
    df["Target"] = df["Target"].clip(lower=min_target, upper=max_target)

    return df

def clean_X(X: pd.DataFrame) -> pd.DataFrame:
    return (X.replace([np.inf, -np.inf], np.nan)
             .ffill().bfill().fillna(0)
             .clip(-1e9, 1e9))

# ══════════════════════════════════════════════════════════════════
#  MODEL TRAINING
#  Cache key = db_version (changes on new data → auto-retrain)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def train(_stock_df: pd.DataFrame, _db_version: str):
    """
    Train on all available data for this stock.
    Returns (featured_df, results_dict, preds_dict,
             test_dates, test_y, rf_full_model)
    """
    df = make_features(_stock_df).dropna(subset=FEATURES + ["Target"])
    X  = clean_X(df[FEATURES])
    y  = df["Target"]
    n  = len(X)

    if n < MIN_ROWS:
        return df, {}, {}, pd.Series(dtype=float), pd.Series(dtype=float), None

    split = int(n * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    dates_te  = df["Date"].iloc[split:]

    sc     = RobustScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xte_sc = sc.transform(Xte)

    model_defs = {
        "Ridge":         (Ridge(alpha=1.0), Xtr_sc, Xte_sc),
        "Random Forest": (RandomForestRegressor(
                            n_estimators=300, max_depth=12,
                            min_samples_leaf=2, random_state=42,
                            n_jobs=-1), Xtr, Xte),
    }
    if XGB_OK:
        model_defs["XGBoost"] = (
            XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42, verbosity=0,
            ), Xtr, Xte
        )

    results, preds = {}, {}
    for name, (model, Xr, Xe) in model_defs.items():
        model.fit(Xr, ytr)
        p = model.predict(Xe)
        results[name] = {
            "MAE":  round(mean_absolute_error(yte, p), 2),
            "RMSE": round(np.sqrt(mean_squared_error(yte, p)), 2),
            "R²":   round(r2_score(yte, p), 4),
            "MAPE": round(np.mean(np.abs((yte - p) / (yte + 1e-9))) * 100, 2),
        }
        preds[name] = p

    # Full RF trained on ALL data for forecasting
    rf_full = RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_full.fit(X, y)

    return df, results, preds, dates_te, yte, rf_full

# ══════════════════════════════════════════════════════════════════
#  RECURSIVE FORECAST
# ══════════════════════════════════════════════════════════════════
def forecast(rf_model, stock_df: pd.DataFrame, n_days: int) -> list:
    """
    Predict next n_days trading days from the latest date in stock_df.
    Uses rolling window — each prediction feeds the next.
    """
    df = make_features(stock_df).dropna(subset=FEATURES)
    if df.empty:
        return []

    last_date  = df["Date"].iloc[-1].date()
    last_close = df["Close"].iloc[-1]
    fut_dates  = next_n_trading_days(last_date, n_days)

    # Rolling close history for updating lag/MA features
    hist_closes = list(df["Close"].tail(60).values)
    current     = clean_X(df[FEATURES]).iloc[-1].copy()
    result      = []

    # Compute recent trend direction for mean-reversion damping
    recent_5  = hist_closes[-5:]  if len(hist_closes) >= 5  else hist_closes
    recent_trend = (recent_5[-1] - recent_5[0]) / recent_5[0] if recent_5[0] != 0 else 0
    # Damping factor: if recent trend already strong, reduce predicted continuation
    # This prevents the model from blindly extrapolating +10% every day

    for fdate in fut_dates:
        closes = hist_closes[-60:]

        def lag(n):
            return closes[-n] if len(closes) >= n else last_close

        # Update lags
        current["Lag_1"]  = lag(1);  current["Lag_2"]  = lag(2)
        current["Lag_3"]  = lag(3);  current["Lag_5"]  = lag(5)
        current["Lag_10"] = lag(10)

        # Update MAs
        for w, col in [(5,"MA_5"),(10,"MA_10"),(20,"MA_20"),(50,"MA_50")]:
            current[col] = np.mean(closes[-w:]) if len(closes) >= w else last_close

        # Update stds
        for w, col in [(5,"Std_5"),(10,"Std_10"),(20,"Std_20")]:
            current[col] = np.std(closes[-w:]) if len(closes) >= w else 0

        # Returns
        p = closes[-1]
        for w, col in [(1,"Return_1d"),(3,"Return_3d"),(5,"Return_5d"),(10,"Return_10d")]:
            current[col] = (p - closes[-w]) / closes[-w] if len(closes) >= w and closes[-w] != 0 else 0

        # Momentum
        current["Momentum_10"] = p - (closes[-10] if len(closes) >= 10 else p)

        # Bollinger position
        ma20 = current["MA_20"]
        std20= current["Std_20"]
        current["BB_pos"] = (p - (ma20 - 2*std20)) / (4*std20 + 1e-9)

        # Circuit breaker features for forecast step
        ret = (p - closes[-2]) / closes[-2] if len(closes) >= 2 and closes[-2] != 0 else 0
        capped_ret = float(np.clip(ret, -CIRCUIT_LIMIT, CIRCUIT_LIMIT))
        current["Return_1d_capped"]   = capped_ret
        current["Hit_Upper_Circuit"]  = 1 if ret >=  0.095 else 0
        current["Hit_Lower_Circuit"]  = 1 if ret <= -0.095 else 0

        # Calendar
        current["DayOfWeek"] = fdate.weekday()
        current["Month"]     = fdate.month
        current["Quarter"]   = (fdate.month - 1) // 3 + 1

        Xp   = clean_X(pd.DataFrame([current])[FEATURES])
        pred = rf_model.predict(Xp)[0]

        # NEPSE circuit breaker: clamp to ±10%
        prev = hist_closes[-1]
        raw_move = (pred - prev) / prev if prev != 0 else 0

        # Mean-reversion damping:
        # If model predicts large move in same direction as recent trend,
        # reduce it — markets rarely sustain max circuit moves consecutively.
        # Scale: full prediction when move < 3%, damped when larger.
        if abs(raw_move) > 0.03:
            # Damping increases with move size: 3-5% → 70%, 5-8% → 50%, 8-10% → 30%
            damp = 0.7 if abs(raw_move) < 0.05 else (0.5 if abs(raw_move) < 0.08 else 0.3)
            # Extra damping if model keeps predicting same large direction
            if len(result) > 0:
                prev_pred_move = (result[-1]["price"] - prev) / prev if prev != 0 else 0
                if raw_move * prev_pred_move > 0 and abs(prev_pred_move) > 0.03:
                    damp *= 0.6   # consecutive large same-direction → damp harder
            damped_move = raw_move * damp
            pred = prev * (1 + damped_move)

        # Final circuit clamp (hard safety net)
        pred = float(np.clip(pred, prev * (1 - CIRCUIT_LIMIT), prev * (1 + CIRCUIT_LIMIT)))

        result.append({"date": fdate, "price": pred})
        hist_closes.append(pred)

    return result

# ══════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════
BG   = "#0A0E1A"; SURF = "#111827"; GRID = "#1F2937"; TXT  = "#6B7280"
UP   = "#10B981"; DN   = "#EF4444"; BL   = "#3B82F6"
AMB  = "#F59E0B"; PUR  = "#8B5CF6"; TEAL = "#06B6D4"

plt.rcParams.update({
    "font.family":       "monospace",
    "axes.facecolor":    SURF,
    "figure.facecolor":  BG,
    "text.color":        TXT,
    "axes.labelcolor":   TXT,
    "xtick.color":       TXT,
    "ytick.color":       TXT,
    "axes.edgecolor":    GRID,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "grid.linewidth":    0.5,
})

def price_chart(df: pd.DataFrame, days: int):
    df    = make_features(df)
    data  = df.tail(days).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    facecolor=BG)
    # Price
    ax1.fill_between(data["Date"], data["Close"], alpha=0.07, color=BL)
    ax1.plot(data["Date"], data["Close"],  color=BL,  lw=1.6, label="Close",  zorder=3)
    ax1.plot(data["Date"], data["MA_5"],   color=AMB, lw=0.9, ls="--", label="MA 5",  alpha=0.85)
    ax1.plot(data["Date"], data["MA_20"],  color=PUR, lw=0.9, ls="--", label="MA 20", alpha=0.85)
    if len(data) >= 50:
        ax1.plot(data["Date"], data["MA_50"], color=TEAL, lw=0.9, ls=":", label="MA 50", alpha=0.7)
    # BB bands
    ax1.fill_between(data["Date"],
                     data["MA_20"] - 2*data["Std_20"],
                     data["MA_20"] + 2*data["Std_20"],
                     alpha=0.04, color=PUR, label="BB bands")
    ax1.set_ylabel("Price (NPR)", fontsize=10)
    ax1.legend(facecolor=SURF, edgecolor=GRID, fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    # Volume
    colors = [UP if c >= p else DN
              for c, p in zip(data["Close"], data["Prev_Close"].fillna(data["Close"]))]
    ax2.bar(data["Date"], data["Volume"], color=colors, alpha=0.75, width=0.8)
    vol_ma = data["Volume"].rolling(10, min_periods=1).mean()
    ax2.plot(data["Date"], vol_ma, color=AMB, lw=0.9, alpha=0.8)
    ax2.set_ylabel("Volume", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout(pad=1.2)
    return fig

def forecast_chart(df: pd.DataFrame, fcast: list,
                   anchor_close: float, anchor_date: datetime.date):
    df_f    = make_features(df)
    recent  = df_f.tail(120).copy()
    f_dates = [pd.Timestamp(f["date"]) for f in fcast]
    f_prices= [f["price"] for f in fcast]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.fill_between(recent["Date"], recent["Close"], alpha=0.06, color=BL)
    ax.plot(recent["Date"], recent["Close"], color=BL, lw=1.8, label="Historical")

    # Dot at anchor
    ax.scatter([pd.Timestamp(anchor_date)], [anchor_close],
               color=UP, s=80, zorder=6,
               label=f"Latest: NPR {anchor_close:,.2f} ({anchor_date})")

    if f_dates:
        # Bridge line from anchor to first forecast
        ax.plot([pd.Timestamp(anchor_date), f_dates[0]],
                [anchor_close, f_prices[0]],
                color=AMB, lw=1, ls="--", alpha=0.5)
        # Forecast line
        ax.plot(f_dates, f_prices, color=AMB, lw=2.2, marker="o",
                markersize=5, label=f"Forecast ({len(fcast)} days)", zorder=5)
        # Uncertainty band (widens over time)
        for i, (fd, fp) in enumerate(zip(f_dates, f_prices)):
            band = 0.02 + i * 0.005   # grows with horizon
            ax.fill_between([fd], [fp*(1-band)], [fp*(1+band)],
                            alpha=0.08, color=AMB)

    ax.set_ylabel("Price (NPR)", fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig

def rsi_chart(df: pd.DataFrame, days: int):
    df_f = make_features(df.tail(days + 30)).tail(days)
    fig, ax = plt.subplots(figsize=(13, 3), facecolor=BG)
    ax.plot(df_f["Date"], df_f["RSI_14"], color=AMB, lw=1.3, label="RSI 14")
    ax.plot(df_f["Date"], df_f["RSI_7"],  color=TEAL, lw=0.9, ls="--", alpha=0.8, label="RSI 7")
    ax.axhline(70, color=DN, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(30, color=UP, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(50, color=GRID, lw=0.5, ls="--", alpha=0.5)
    ax.fill_between(df_f["Date"], df_f["RSI_14"], 70,
                    where=df_f["RSI_14"] >= 70, alpha=0.12, color=DN)
    ax.fill_between(df_f["Date"], df_f["RSI_14"], 30,
                    where=df_f["RSI_14"] <= 30, alpha=0.12, color=UP)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI", fontsize=9)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig

def model_comparison_chart(dates_te, yte, preds: dict):
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.plot(dates_te.values, yte.values, color=BL, lw=2, label="Actual", zorder=4)
    palette = [UP, AMB, PUR, TEAL]
    for (name, p), col in zip(preds.items(), palette):
        ax.plot(dates_te.values, p, color=col, lw=1.1, ls="--", alpha=0.85, label=name)
    ax.set_ylabel("Price (NPR)", fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig

def macd_chart(df: pd.DataFrame, days: int):
    df_f = make_features(df.tail(days + 30)).tail(days)
    fig, ax = plt.subplots(figsize=(13, 3), facecolor=BG)
    ax.plot(df_f["Date"], df_f["MACD"], color=BL, lw=1.3, label="MACD")
    ax.axhline(0, color=GRID, lw=0.7, ls="--")
    ax.fill_between(df_f["Date"], df_f["MACD"], 0,
                    where=df_f["MACD"] >= 0, alpha=0.2, color=UP)
    ax.fill_between(df_f["Date"], df_f["MACD"], 0,
                    where=df_f["MACD"] < 0,  alpha=0.2, color=DN)
    ax.set_ylabel("MACD", fontsize=9)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig


# ══════════════════════════════════════════════════════════════════
#  INVESTABILITY SCORE ENGINE
#  Scores a stock 0-100 based on technical signals.
#  Used to drive the investor meter.
# ══════════════════════════════════════════════════════════════════
def compute_score(df: pd.DataFrame, latest: dict) -> dict:
    """
    Returns a dict with:
      score        → 0-100 overall investability
      signals      → list of (label, value, sentiment) tuples
      breakdown    → dict of sub-scores
    """
    ind = make_features(df.tail(60))
    if ind.empty or len(ind) < 5:
        return {"score": 50, "signals": [], "breakdown": {}}

    last       = ind.iloc[-1]
    close      = float(last["Close"])
    high_52w   = float(latest.get("high_52w", close))
    low_52w    = float(latest.get("low_52w",  close))
    ma20       = float(last["MA_20"])
    ma50       = float(last["MA_50"]) if not np.isnan(last["MA_50"]) else ma20
    rsi14      = float(last["RSI_14"])
    macd_v     = float(last["MACD"])
    bb_pos     = float(last["BB_pos"])
    vol        = float(last["Volume"])
    vol_ma5    = float(last["Vol_MA5_Ratio"])
    ret5       = float(last["Return_5d"])  if not np.isnan(last["Return_5d"])  else 0
    ret10      = float(last["Return_10d"]) if not np.isnan(last["Return_10d"]) else 0
    std20      = float(last["Std_20"])
    momentum   = float(last["Momentum_10"])

    signals = []
    scores  = {}

    # 1. RSI signal (0-25 pts)
    if rsi14 < 30:
        rsi_score = 22
        rsi_sent  = "bullish"
        rsi_label = f"RSI {rsi14:.1f} — Oversold (buy zone)"
    elif rsi14 > 70:
        rsi_score = 5
        rsi_sent  = "bearish"
        rsi_label = f"RSI {rsi14:.1f} — Overbought (caution)"
    elif 40 <= rsi14 <= 60:
        rsi_score = 18
        rsi_sent  = "neutral"
        rsi_label = f"RSI {rsi14:.1f} — Neutral zone"
    else:
        rsi_score = 12
        rsi_sent  = "neutral"
        rsi_label = f"RSI {rsi14:.1f}"
    signals.append(("RSI (14)", f"{rsi14:.1f}", rsi_sent))
    scores["RSI"] = rsi_score

    # 2. Trend vs MAs (0-25 pts)
    if close > ma20 and close > ma50:
        trend_score = 23
        trend_sent  = "bullish"
        trend_label = "Price above MA20 & MA50 — strong uptrend"
    elif close > ma20:
        trend_score = 15
        trend_sent  = "bullish"
        trend_label = "Price above MA20 — moderate uptrend"
    elif close < ma20 and close < ma50:
        trend_score = 4
        trend_sent  = "bearish"
        trend_label = "Price below MA20 & MA50 — downtrend"
    else:
        trend_score = 10
        trend_sent  = "neutral"
        trend_label = "Mixed trend signals"
    signals.append(("Trend", trend_label, trend_sent))
    scores["Trend"] = trend_score

    # 3. MACD (0-20 pts)
    if macd_v > 0:
        macd_score = 18
        macd_sent  = "bullish"
        macd_label = f"MACD {macd_v:+.2f} — Bullish momentum"
    else:
        macd_score = 5
        macd_sent  = "bearish"
        macd_label = f"MACD {macd_v:+.2f} — Bearish momentum"
    signals.append(("MACD", f"{macd_v:+.2f}", macd_sent))
    scores["MACD"] = macd_score

    # 4. 52-week position (0-15 pts)
    if high_52w > low_52w:
        pos_52w = (close - low_52w) / (high_52w - low_52w)
        if pos_52w >= 0.75:
            wk_score = 8
            wk_sent  = "neutral"
            wk_label = f"Near 52-week high ({pos_52w*100:.0f}th percentile)"
        elif pos_52w <= 0.25:
            wk_score = 14
            wk_sent  = "bullish"
            wk_label = f"Near 52-week low — potential value ({pos_52w*100:.0f}th percentile)"
        else:
            wk_score = 11
            wk_sent  = "neutral"
            wk_label = f"Mid 52-week range ({pos_52w*100:.0f}th percentile)"
    else:
        pos_52w  = 0.5
        wk_score = 10
        wk_sent  = "neutral"
        wk_label = "52-week range unavailable"
    signals.append(("52W Position", wk_label, wk_sent))
    scores["52W Range"] = wk_score

    # 5. Volume strength (0-15 pts)
    if vol_ma5 > 1.5:
        vol_score = 13
        vol_sent  = "bullish"
        vol_label = f"Volume {vol_ma5:.1f}x above 5-day avg — high activity"
    elif vol_ma5 > 1.0:
        vol_score = 10
        vol_sent  = "neutral"
        vol_label = f"Volume {vol_ma5:.1f}x avg — normal activity"
    else:
        vol_score = 5
        vol_sent  = "bearish"
        vol_label = f"Volume below avg ({vol_ma5:.1f}x) — low interest"
    signals.append(("Volume", f"{vol_ma5:.1f}x avg", vol_sent))
    scores["Volume"] = vol_score

    total = sum(scores.values())   # max possible ≈ 96

    # Normalize to 0-100
    score = min(100, int(total / 96 * 100))

    return {
        "score":     score,
        "signals":   signals,
        "breakdown": scores,
        "pos_52w":   pos_52w,
        "rsi14":     rsi14,
        "macd_v":    macd_v,
    }


def score_label(score: int) -> tuple:
    """Returns (label, color, emoji) for the score."""
    if score >= 80: return "Excellent",  "#10B981", "🟢"
    if score >= 65: return "Good",        "#34D399", "🟢"
    if score >= 50: return "Moderate",    "#F59E0B", "🟡"
    if score >= 35: return "Weak",        "#F97316", "🟠"
    return           "Poor",              "#EF4444", "🔴"


# ══════════════════════════════════════════════════════════════════
#  ADDITIONAL CHARTS
# ══════════════════════════════════════════════════════════════════
BG   = "#0A0E1A"; SURF = "#111827"; GRID = "#1F2937"; TXT  = "#6B7280"
UP   = "#10B981"; DN   = "#EF4444"; BL   = "#3B82F6"
AMB  = "#F59E0B"; PUR  = "#8B5CF6"; TEAL = "#06B6D4"

plt.rcParams.update({
    "font.family":       "monospace",
    "axes.facecolor":    SURF,
    "figure.facecolor":  BG,
    "text.color":        TXT,
    "axes.labelcolor":   TXT,
    "xtick.color":       TXT,
    "ytick.color":       TXT,
    "axes.edgecolor":    GRID,
    "grid.color":        GRID,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "grid.linewidth":    0.5,
})

def price_chart(df, days):
    df   = make_features(df)
    data = df.tail(days).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    gridspec_kw={"height_ratios":[3,1]}, facecolor=BG)
    ax1.fill_between(data["Date"], data["Close"], alpha=0.07, color=BL)
    ax1.plot(data["Date"], data["Close"],  color=BL,  lw=1.6, label="Close", zorder=3)
    ax1.plot(data["Date"], data["MA_5"],   color=AMB, lw=0.9, ls="--", label="MA 5",  alpha=0.8)
    ax1.plot(data["Date"], data["MA_20"],  color=PUR, lw=0.9, ls="--", label="MA 20", alpha=0.8)
    if len(data) >= 50:
        ax1.plot(data["Date"], data["MA_50"], color=TEAL, lw=0.9, ls=":", label="MA 50", alpha=0.7)
    ax1.fill_between(data["Date"],
                     data["MA_20"] - 2*data["Std_20"],
                     data["MA_20"] + 2*data["Std_20"],
                     alpha=0.04, color=PUR, label="Bollinger Bands")
    ax1.set_ylabel("Price (NPR)", fontsize=10)
    ax1.legend(facecolor=SURF, edgecolor=GRID, fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    colors = [UP if c >= p else DN
              for c, p in zip(data["Close"], data["Prev_Close"].fillna(data["Close"]))]
    ax2.bar(data["Date"], data["Volume"], color=colors, alpha=0.75, width=0.8)
    vol_ma = data["Volume"].rolling(10, min_periods=1).mean()
    ax2.plot(data["Date"], vol_ma, color=AMB, lw=0.9, alpha=0.8)
    ax2.set_ylabel("Volume", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig

def candlestick_chart(df, days):
    """Draw a candlestick chart using matplotlib patches."""
    import matplotlib.patches as mpatches
    data = df.tail(days).copy().reset_index(drop=True)
    if "High" not in data.columns or "Low" not in data.columns:
        return price_chart(df, days)

    # Fill missing OHLC
    data["Open"]  = data.get("Open",  data["Close"].shift(1).fillna(data["Close"]))
    data["High"]  = data["High"].replace(0, np.nan).fillna(data["Close"])
    data["Low"]   = data["Low"].replace(0, np.nan).fillna(data["Close"])
    if "Prev_Close" not in data.columns:
        data["Prev_Close"] = data["Close"].shift(1).fillna(data["Close"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                    gridspec_kw={"height_ratios":[3,1]}, facecolor=BG)

    for i, row in data.iterrows():
        is_up   = row["Close"] >= row["Open"]
        color   = UP if is_up else DN
        body_lo = min(row["Open"], row["Close"])
        body_hi = max(row["Open"], row["Close"])
        body_h  = max(body_hi - body_lo, row["Close"] * 0.001)  # min height

        # Wick
        ax1.plot([i, i], [row["Low"], row["High"]], color=color, lw=0.8, alpha=0.8)
        # Body
        rect = mpatches.FancyBboxPatch(
            (i - 0.35, body_lo), 0.7, body_h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor=color, alpha=0.85
        )
        ax1.add_patch(rect)

    # Moving averages on top
    if len(data) >= 5:
        ma5  = data["Close"].rolling(5,  min_periods=1).mean()
        ax1.plot(range(len(data)), ma5,  color=AMB, lw=0.9, ls="--", label="MA 5",  alpha=0.8)
    if len(data) >= 20:
        ma20 = data["Close"].rolling(20, min_periods=1).mean()
        ax1.plot(range(len(data)), ma20, color=PUR, lw=0.9, ls="--", label="MA 20", alpha=0.8)

    # X-axis: show dates
    tick_step = max(1, len(data) // 8)
    tick_idx  = list(range(0, len(data), tick_step))
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(
        [data["Date"].iloc[i].strftime("%d %b") for i in tick_idx],
        rotation=30, ha="right", fontsize=8
    )
    ax1.set_ylabel("Price (NPR)", fontsize=10)
    ax1.legend(facecolor=SURF, edgecolor=GRID, fontsize=8, loc="upper left")
    ax1.set_xlim(-1, len(data))

    # Volume
    vol_colors = [UP if c >= p else DN
                  for c, p in zip(data["Close"], data["Prev_Close"].fillna(data["Close"]))]
    ax2.bar(range(len(data)), data["Volume"], color=vol_colors, alpha=0.75, width=0.8)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(
        [data["Date"].iloc[i].strftime("%d %b") for i in tick_idx],
        rotation=30, ha="right", fontsize=8
    )
    ax2.set_ylabel("Volume", fontsize=9)
    ax2.set_xlim(-1, len(data))

    fig.tight_layout(pad=1.2)
    return fig


def forecast_chart(df, fcast, anchor_close, anchor_date):
    df_f    = make_features(df)
    recent  = df_f.tail(90).copy()
    f_dates = [pd.Timestamp(f["date"]) for f in fcast]
    f_prices= [f["price"] for f in fcast]
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.fill_between(recent["Date"], recent["Close"], alpha=0.06, color=BL)
    ax.plot(recent["Date"], recent["Close"], color=BL, lw=1.8, label="Historical")
    ax.scatter([pd.Timestamp(anchor_date)], [anchor_close],
               color=UP, s=80, zorder=6,
               label=f"Latest: NPR {anchor_close:,.2f}")
    if f_dates:
        ax.plot([pd.Timestamp(anchor_date), f_dates[0]],
                [anchor_close, f_prices[0]], color=AMB, lw=1, ls="--", alpha=0.5)
        ax.plot(f_dates, f_prices, color=AMB, lw=2.2, marker="o",
                markersize=5, label=f"AI Forecast ({len(fcast)} days)", zorder=5)
        for i, (fd, fp) in enumerate(zip(f_dates, f_prices)):
            band = 0.02 + i * 0.005
            ax.fill_between([fd], [fp*(1-band)], [fp*(1+band)], alpha=0.08, color=AMB)
    ax.set_ylabel("Price (NPR)", fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig

def rsi_chart(df, days):
    df_f = make_features(df.tail(days+30)).tail(days)
    fig, ax = plt.subplots(figsize=(13, 3), facecolor=BG)
    ax.plot(df_f["Date"], df_f["RSI_14"], color=AMB, lw=1.3, label="RSI 14")
    ax.plot(df_f["Date"], df_f["RSI_7"],  color=TEAL, lw=0.9, ls="--", alpha=0.8, label="RSI 7")
    ax.axhline(70, color=DN, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(30, color=UP, lw=0.8, ls="--", alpha=0.7)
    ax.axhline(50, color=GRID, lw=0.5, ls="--", alpha=0.5)
    ax.fill_between(df_f["Date"], df_f["RSI_14"], 70, where=df_f["RSI_14"]>=70, alpha=0.12, color=DN)
    ax.fill_between(df_f["Date"], df_f["RSI_14"], 30, where=df_f["RSI_14"]<=30, alpha=0.12, color=UP)
    ax.set_ylim(0, 100); ax.set_ylabel("RSI", fontsize=9)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2); return fig

def macd_chart(df, days):
    df_f = make_features(df.tail(days+30)).tail(days)
    fig, ax = plt.subplots(figsize=(13, 3), facecolor=BG)
    ax.plot(df_f["Date"], df_f["MACD"], color=BL, lw=1.3, label="MACD")
    ax.axhline(0, color=GRID, lw=0.7, ls="--")
    ax.fill_between(df_f["Date"], df_f["MACD"], 0, where=df_f["MACD"]>=0, alpha=0.2, color=UP)
    ax.fill_between(df_f["Date"], df_f["MACD"], 0, where=df_f["MACD"]<0,  alpha=0.2, color=DN)
    ax.set_ylabel("MACD", fontsize=9)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2); return fig

def model_comparison_chart(dates_te, yte, preds):
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.plot(dates_te.values, yte.values, color=BL, lw=2, label="Actual", zorder=4)
    for (name, p), col in zip(preds.items(), [UP, AMB, PUR]):
        ax.plot(dates_te.values, p, color=col, lw=1.1, ls="--", alpha=0.85, label=name)
    ax.set_ylabel("Price (NPR)", fontsize=10)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2); return fig

def volume_analysis_chart(df, days):
    df_f = make_features(df.tail(days+10)).tail(days)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6),
                                    gridspec_kw={"height_ratios":[2,1]}, facecolor=BG)
    # Price with volume color coding
    colors = [UP if c >= p else DN
              for c, p in zip(df_f["Close"], df_f["Prev_Close"].fillna(df_f["Close"]))]
    ax1.plot(df_f["Date"], df_f["Close"], color=BL, lw=1.5)
    ax1.set_ylabel("Price (NPR)", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    # Volume bars
    ax2.bar(df_f["Date"], df_f["Volume"], color=colors, alpha=0.75, width=0.8)
    vol_avg = df_f["Volume"].mean()
    ax2.axhline(vol_avg, color=AMB, lw=1, ls="--", alpha=0.8, label=f"Avg: {vol_avg:,.0f}")
    ax2.set_ylabel("Volume", fontsize=9)
    ax2.legend(facecolor=SURF, edgecolor=GRID, fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(pad=1.2); return fig

def returns_distribution_chart(df):
    df_f  = make_features(df)
    rets  = df_f["Return_1d"].dropna() * 100
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    n, bins, patches = ax.hist(rets, bins=40, color=BL, alpha=0.7, edgecolor=GRID, lw=0.3)
    for patch, b in zip(patches, bins):
        if b < 0:
            patch.set_facecolor(DN)
            patch.set_alpha(0.7)
    ax.axvline(0,    color=TXT,  lw=0.8, ls="--")
    ax.axvline(-10,  color=DN,   lw=0.8, ls=":",  alpha=0.7, label="Circuit -10%")
    ax.axvline(10,   color=UP,   lw=0.8, ls=":",  alpha=0.7, label="Circuit +10%")
    ax.axvline(rets.mean(), color=AMB, lw=1.2, ls="--", label=f"Mean {rets.mean():.2f}%")
    ax.set_xlabel("Daily Return (%)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(facecolor=SURF, edgecolor=GRID, fontsize=8)
    fig.tight_layout(pad=1.2); return fig


# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NEPSE Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #070B14; color: #E2E8F0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #1E2A3A;
}

/* Hide hamburger / header */
.stDeployButton { display: none; }
#MainMenu, footer { visibility: hidden; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0D1117;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: #2D4A6A; }
div[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .09em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 19px !important;
    color: #F1F5F9 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1117;
    border-bottom: 1px solid #1E2A3A;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748B;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: .05em;
    padding: .6rem 1.4rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
    background: transparent !important;
}

/* Selectbox & inputs */
div[data-baseweb="select"] > div {
    background: #0D1117 !important;
    border: 1px solid #1E2A3A !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
input { font-family: 'IBM Plex Mono', monospace !important; }

/* Divider */
hr { border-color: #1E2A3A; margin: 1.2rem 0; }

/* Custom classes */
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin-bottom: .25rem;
}
.stock-name {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 800;
    color: #F1F5F9;
    line-height: 1;
    margin-bottom: .25rem;
}
.stock-price {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 32px;
    font-weight: 500;
}
.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    vertical-align: middle;
}
.up   { background: #052E16; color: #10B981; border: 1px solid #059669; }
.dn   { background: #450A0A; color: #F87171; border: 1px solid #DC2626; }
.nt   { background: #0F172A; color: #64748B; border: 1px solid #334155; }
.bl   { background: #0C1A3A; color: #38BDF8; border: 1px solid #0284C7; }

/* Section headers */
.sec-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: .2em;
    padding-bottom: .35rem;
    border-bottom: 1px solid #1E2A3A;
    margin: 1.75rem 0 .85rem;
}

/* Prediction box */
.pred-wrap {
    background: linear-gradient(135deg, #0C1420, #0A1628);
    border: 1px solid #1D4ED844;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin: .75rem 0 1.25rem;
}
.pred-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: .15em;
    margin-bottom: .5rem;
}
.pred-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 38px;
    font-weight: 500;
    color: #F1F5F9;
}
.pred-sub { font-size: 13px; color: #64748B; margin-top: .35rem; }

/* Forecast cards */
.fc-row { display: flex; gap: 8px; flex-wrap: wrap; margin: .75rem 0 1rem; }
.fc-card {
    background: #0D1117;
    border: 1px solid #1E2A3A;
    border-radius: 10px;
    padding: .65rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    min-width: 115px;
    flex: 1;
    max-width: 150px;
    transition: border-color .2s;
}
.fc-card:hover { border-color: #2D4A6A; }
.fc-d  { font-size: 10px; color: #64748B; margin-bottom: 3px; }
.fc-p  { font-size: 15px; color: #F1F5F9; }
.fc-up { font-size: 11px; color: #10B981; }
.fc-dn { font-size: 11px; color: #F87171; }

/* Investability meter */
.meter-wrap {
    background: #0D1117;
    border: 1px solid #1E2A3A;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.meter-score {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 800;
    line-height: 1;
}
.meter-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-top: .25rem;
}
.signal-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: .45rem 0;
    border-bottom: 1px solid #0F172A;
    font-size: 13px;
}
.sig-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sig-bull { background: #10B981; }
.sig-bear { background: #F87171; }
.sig-neut { background: #F59E0B; }
.sig-lbl  { color: #94A3B8; font-family: 'IBM Plex Mono', monospace; font-size: 11px; flex: 0 0 90px; }
.sig-val  { color: #CBD5E1; font-size: 12px; }

/* 52-week bar */
.wk52-wrap {
    background: #0D1117;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: .5rem 0;
}
.wk52-bar-bg {
    background: #1E2A3A;
    border-radius: 4px;
    height: 6px;
    margin: .5rem 0;
    position: relative;
}
.wk52-bar-fill {
    background: linear-gradient(90deg, #EF4444, #F59E0B, #10B981);
    border-radius: 4px;
    height: 6px;
}
.wk52-marker {
    position: absolute;
    top: -4px;
    width: 14px;
    height: 14px;
    background: white;
    border: 2px solid #38BDF8;
    border-radius: 50%;
    transform: translateX(-50%);
}
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: .5rem 0;
}
.stat-item {
    background: #070B14;
    border-radius: 8px;
    padding: .5rem .75rem;
}
.stat-lbl { font-size: 10px; color: #475569; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .08em; }
.stat-val { font-size: 14px; color: #E2E8F0; font-family: 'IBM Plex Mono', monospace; margin-top: 2px; }

.stale-warn {
    background: #1C1407;
    border: 1px solid #92400E;
    border-radius: 8px;
    padding: .6rem 1rem;
    font-size: 13px;
    color: #FCD34D;
    margin-bottom: 1rem;
}
.disclaimer {
    font-size: 11px;
    color: #334155;
    font-family: 'IBM Plex Mono', monospace;
    padding: .75rem;
    border-top: 1px solid #1E2A3A;
    margin-top: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════
def main():
    today = datetime.date.today()
    init_db()

    # ── Developer panel: access via ?dev=1 in URL ──────────────────
    # e.g. http://localhost:8501/?dev=1
    #params = st.experimental_get_query_params()
    try:
        params = st.query_params
        symbol = params.get("symbol", "")
    except AttributeError:
        params = st.experimental_get_query_params()
        symbol = params.get("symbol", [""])[0]
    is_dev = params.get("dev", [""])[0] == "1"

    if is_dev:
        st.markdown("### ⚙️ Developer Panel")
        st.info("This panel is hidden from users. Access via `?dev=1` in the URL.")
        col_a, col_b = st.columns(2)
        with col_a:
            uploaded_files = st.file_uploader(
                "Upload New_Data CSV(s)",
                type=["csv"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                total_new = 0
                for uf in uploaded_files:
                    raw   = uf.read()
                    fhash = hashlib.md5(raw).hexdigest()
                    save_path = os.path.join(NEW_DATA_DIR, uf.name)
                    if not os.path.exists(save_path):
                        with open(save_path, "wb") as f:
                            f.write(raw)
                    if not already_ingested(fhash):
                        rows, err = parse_csv(raw, uf.name)
                        if err:
                            st.error(f"{uf.name}: {err}")
                        else:
                            n = bulk_insert(rows, "uploaded")
                            mark_ingested(fhash, uf.name, n)
                            total_new += n
                if total_new > 0:
                    st.cache_data.clear()
                    st.success(f"✅ {total_new:,} new records loaded!")
                    st.rerun()
        with col_b:
            if st.button("🔄 Force Reload New_Data Folder"):
                nf, nr = ingest_folder(NEW_DATA_DIR, "new_data", silent=False)
                st.cache_data.clear()
                st.success(f"Reloaded {nf} file(s), {nr:,} records.")
                st.rerun()
            s = db_stats()
            st.json(s)
        st.markdown("---")

    stats = db_stats()

    # ── First-run seed ─────────────────────────────────────────────
    if stats["total"] == 0:
        st.markdown("### ⚙️ Setting up database…")
        pb = st.progress(0, text="Loading Training_Data…")
        nf, nr = ingest_folder(TRAINING_DIR, "training", pb, silent=False)
        pb.empty()
        ingest_folder(NEW_DATA_DIR, "new_data", silent=True)
        st.success(f"✅ {nf} files · {nr:,} records loaded. Refreshing…")
        time.sleep(1); st.rerun()

    # ── Auto-ingest New_Data silently ─────────────────────────────
    nf, nr = ingest_folder(NEW_DATA_DIR, "new_data", silent=True)
    if nf > 0:
        st.cache_data.clear()

    stats   = db_stats()
    tickers = all_tickers()

    # ══════════════════════════════════════════════════════════════
    #  SIDEBAR — search only, clean
    # ══════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            "<p style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;"
            "color:#F1F5F9;margin:1rem 0 .25rem'>📊 NEPSE Analytics</p>"
            "<p style='font-size:11px;color:#334155;font-family:IBM Plex Mono,monospace;"
            "margin-bottom:1.25rem'>Nepal Stock Exchange · AI-Powered</p>",
            unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        search  = st.text_input("🔍 Search by symbol",
                                placeholder="NABIL, ADBL, NHPC…",
                                label_visibility="collapsed")
        search  = search.strip().upper()
        matches = [t for t in tickers if search in t] if search else tickers
        if not matches:
            st.warning("No stocks found.")
            st.stop()
        selected = st.selectbox("Select Stock", matches,
                                label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='sec-hdr' style='margin-top:.5rem'>Chart Settings</div>",
                    unsafe_allow_html=True)
        days  = st.slider("History", 30, 365, 180, 10,
                          format="%d days", label_visibility="visible")
        fdays = st.select_slider("Forecast", [1, 7, 14, 30], value=7,
                                 label_visibility="visible")
        chart_type = st.radio("Chart Type", ["Line", "Candlestick"],
                              horizontal=True, label_visibility="visible")

        st.markdown("<hr>", unsafe_allow_html=True)
        

    # ══════════════════════════════════════════════════════════════
    #  LOAD DATA & TRAIN
    # ══════════════════════════════════════════════════════════════
    stock_df = get_stock(selected)
    if stock_df.empty:
        st.error(f"No data found for **{selected}**."); st.stop()

    n_rows       = len(stock_df)
    latest_date  = stock_df["Date"].iloc[-1].date()
    latest_close = float(stock_df["Close"].iloc[-1])
    prev_close   = float(stock_df["Prev_Close"].iloc[-1]) if n_rows > 1 else latest_close
    high_52w     = float(stock_df["High"].max())
    low_52w      = float(stock_df["Low"].min())
    avg_52w      = float(stock_df["Close"].mean())
    avg_vol      = float(stock_df["Volume"].mean())
    chg          = latest_close - prev_close
    chg_pct      = (chg / prev_close * 100) if prev_close else 0
    arrow        = "▲" if chg >= 0 else "▼"
    d_cls        = "up" if chg >= 0 else "dn"
    days_stale   = (today - latest_date).days
    next_trade   = next_n_trading_days(latest_date, 1)[0]
    pos_52w      = (latest_close - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5

    ver = db_version(selected)
    with st.spinner(f"Analysing {selected}…"):
        df_feat, results, preds, dates_te, yte, rf_full = train(stock_df, ver)

    if rf_full is None:
        st.error(
            f"**{selected}** has only **{n_rows}** trading session(s) in the database. "
            f"At least {MIN_ROWS} are needed. Add more daily CSVs to the New_Data folder."
        )
        st.stop()

    fcast      = forecast(rf_full, stock_df, fdays) if rf_full else []
    price_color = "#10B981" if chg > 0 else ("#F87171" if chg < 0 else "#94A3B8")
    tom_price  = fcast[0]["price"] if fcast else latest_close
    tom_chg    = tom_price - latest_close
    tom_pct    = (tom_chg / latest_close * 100) if latest_close else 0
    tom_arrow  = "▲" if tom_chg >= 0 else "▼"
    tom_color  = "#10B981" if tom_chg >= 0 else "#F87171"

    # Technical indicators
    ind    = make_features(stock_df.tail(60))
    rsi14  = float(ind["RSI_14"].iloc[-1]) if not ind.empty else 50
    macd_v = float(ind["MACD"].iloc[-1])   if not ind.empty else 0
    ma20   = float(ind["MA_20"].iloc[-1])   if not ind.empty else latest_close
    std20  = float(ind["Std_20"].iloc[-1])  if not ind.empty else 0
    bb_up  = ma20 + 2*std20
    bb_dn  = ma20 - 2*std20

    # Investability score
    latest_extra = {"high_52w": high_52w, "low_52w": low_52w}
    inv = compute_score(stock_df, latest_extra)
    score = inv["score"]
    s_label, s_color, s_emoji = score_label(score)
    best_model = min(results, key=lambda k: results[k]["RMSE"]) if results else "Random Forest"

    # ── Stale data warning ────────────────────────────────────────
    if days_stale > 3:
        st.markdown(
            f"<div class='stale-warn'>⚠ Data is <b>{days_stale} days old</b> "
            f"(last: {latest_date}). Drop a new CSV in <code>New_Data/</code> "
            f"and refresh for updated predictions.</div>",
            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  HEADER ROW
    # ══════════════════════════════════════════════════════════════
    col_hdr, col_score = st.columns([3, 1])

    with col_hdr:
        st.markdown(
            f"<p class='page-title'>Nepal Stock Exchange</p>"
            f"<p class='stock-name'>{selected}</p>"
            f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:.5rem'>"
            f"<span class='stock-price' style='color:{price_color}'>"
            f"NPR {latest_close:,.2f}</span>"
            f"<span class='tag {d_cls}'>{arrow} NPR {abs(chg):,.2f} ({abs(chg_pct):.2f}%)</span>"
            f"<span class='tag bl'>As of {latest_date}</span>"
            f"<span class='tag nt'>{n_rows} sessions</span>"
            f"</div>",
            unsafe_allow_html=True)

    with col_score:
        # ── Animated SVG Speedometer ──────────────────────────────
        # Arc goes from -180° (left = 0) to 0° (right = 100)
        # Needle sweeps from left to current score position
        import math

        cx, cy, r = 110, 105, 80   # center, radius

        def polar(angle_deg, radius=r):
            """Convert angle (0=left, 180=right) to SVG x,y."""
            rad = math.radians(180 - angle_deg)
            return cx + radius * math.cos(rad), cy - radius * math.sin(rad)

        # Score → angle (0 = left end, 180 = right end)
        needle_angle = score / 100 * 180

        # Needle tip and base points
        nx, ny   = polar(needle_angle, r - 10)
        bl_x, bl_y = polar(needle_angle - 90, 8)
        br_x, br_y = polar(needle_angle + 90, 8)

        # Arc segments: Poor 0-35, Weak 35-50, Moderate 50-65, Good 65-80, Excellent 80-100
        def arc_path(start_pct, end_pct, inner=58, outer=r):
            a0 = start_pct / 100 * 180
            a1 = end_pct   / 100 * 180
            ox0, oy0 = polar(a0, outer)
            ox1, oy1 = polar(a1, outer)
            ix0, iy0 = polar(a0, inner)
            ix1, iy1 = polar(a1, inner)
            large = 1 if (a1 - a0) > 180 else 0
            return (f"M {ox0:.1f} {oy0:.1f} "
                    f"A {outer} {outer} 0 {large} 1 {ox1:.1f} {oy1:.1f} "
                    f"L {ix1:.1f} {iy1:.1f} "
                    f"A {inner} {inner} 0 {large} 0 {ix0:.1f} {iy0:.1f} Z")

        segments = [
            (0,  35,  "#7F1D1D", "#EF4444"),   # Poor — red
            (35, 50,  "#7C2D12", "#F97316"),   # Weak — orange
            (50, 65,  "#713F12", "#F59E0B"),   # Moderate — amber
            (65, 80,  "#14532D", "#22C55E"),   # Good — green
            (80, 100, "#052E16", "#10B981"),   # Excellent — emerald
        ]

        seg_paths = ""
        for s0, s1, fill_dark, fill_bright in segments:
            seg_paths += f'<path d="{arc_path(s0, s1)}" fill="{fill_dark}" stroke="{fill_bright}" stroke-width="0.5" opacity="0.9"/>'

        # Tick marks
        ticks_svg = ""
        for pct in [0, 25, 50, 75, 100]:
            ang = pct / 100 * 180
            tx0, ty0 = polar(ang, r + 4)
            tx1, ty1 = polar(ang, r - 3)
            ticks_svg += f'<line x1="{tx0:.1f}" y1="{ty0:.1f}" x2="{tx1:.1f}" y2="{ty1:.1f}" stroke="#334155" stroke-width="1.5"/>'

        # Labels
        label_data = [("Poor", 0), ("Weak", 25), ("Mod", 50), ("Good", 75), ("Best", 100)]
        labels_svg = ""
        for lbl, pct in label_data:
            ang = pct / 100 * 180
            lx, ly = polar(ang, r + 16)
            labels_svg += (f'<text x="{lx:.1f}" y="{ly:.1f}" '
                           f'text-anchor="middle" dominant-baseline="middle" '
                           f'font-size="7" fill="#475569" '
                           f'font-family="IBM Plex Mono,monospace">{lbl}</text>')

        speedometer_svg = f"""
        <div style='background:#0D1117;border:1px solid #1E2A3A;border-radius:16px;
             padding:1rem 1rem .75rem;text-align:center'>
            <div style='font-size:10px;color:#475569;font-family:IBM Plex Mono,monospace;
                 text-transform:uppercase;letter-spacing:.15em;margin-bottom:.25rem'>
                Investability Score
            </div>
            <svg viewBox="30 20 160 110" width="100%" height="160"
                 xmlns="http://www.w3.org/2000/svg">

                <!-- Background arc -->
                <path d="{arc_path(0, 100, 58, r)}" fill="#0F172A" stroke="#1E2A3A" stroke-width="0.5"/>

                <!-- Colored segments -->
                {seg_paths}

                <!-- Tick marks -->
                {ticks_svg}

                <!-- Labels -->
                {labels_svg}

                <!-- Needle: CSS animation via style tag -->
                <style>
                  @keyframes sweepNeedle {{
                    from {{ transform: rotate(-{needle_angle:.1f}deg); transform-origin: {cx}px {cy}px; }}
                    to   {{ transform: rotate(0deg);                  transform-origin: {cx}px {cy}px; }}
                  }}
                  #needle {{
                    transform-origin: {cx}px {cy}px;
                    animation: sweepNeedle 1.2s cubic-bezier(0.25,0.46,0.45,0.94) 0.1s both;
                  }}
                </style>
                <polygon id="needle"
                    points="{nx:.1f},{ny:.1f} {bl_x:.1f},{bl_y:.1f} {br_x:.1f},{br_y:.1f}"
                    fill="{s_color}" opacity="0.95"/>

                <!-- Center hub -->
                <circle cx="{cx}" cy="{cy}" r="7" fill="{s_color}" opacity="0.9"/>
                <circle cx="{cx}" cy="{cy}" r="3" fill="#070B14"/>

                <!-- Score text -->
                <text x="{cx}" y="{cy + 22}" text-anchor="middle"
                      font-size="22" font-weight="700" fill="{s_color}"
                      font-family="Syne,sans-serif">{score}</text>
                <text x="{cx}" y="{cy + 34}" text-anchor="middle"
                      font-size="8" fill="{s_color}" opacity="0.8"
                      font-family="IBM Plex Mono,monospace"
                      letter-spacing="2">{s_label.upper()}</text>
            </svg>

            <!-- Straight bar below gauge -->
            <div style='margin:.25rem .5rem 0;background:#1E2A3A;border-radius:4px;
                 height:6px;overflow:hidden'>
                <div style='height:6px;width:{score}%;border-radius:4px;
                     background:linear-gradient(90deg,#EF4444,#F59E0B,#10B981);
                     transition:width 1.2s ease'></div>
            </div>
            <div style='display:flex;justify-content:space-between;margin-top:3px;
                 font-size:8px;color:#334155;font-family:IBM Plex Mono,monospace;
                 padding:0 .5rem'>
                <span>Poor</span><span>Moderate</span><span>Excellent</span>
            </div>
        </div>"""
        # Wrap in full HTML doc for components.html iframe renderer
        speedometer_html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{ margin:0; padding:0; background:#0D1117; font-family:'IBM Plex Mono',monospace; }}
</style>
</head><body>{speedometer_svg}</body></html>"""
        components.html(speedometer_html, height=260, scrolling=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  KEY METRICS ROW
    # ══════════════════════════════════════════════════════════════
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1: st.metric("Close (NPR)",    f"{latest_close:,.1f}",
                        f"{arrow} {chg_pct:+.2f}%")
    with m2: st.metric("52W High (NPR)", f"{high_52w:,.1f}")
    with m3: st.metric("52W Low (NPR)",  f"{low_52w:,.1f}")
    with m4: st.metric("52W Avg (NPR)",  f"{avg_52w:,.1f}")
    with m5: st.metric("RSI (14)",       f"{rsi14:.1f}",
                        "Overbought" if rsi14>70 else ("Oversold" if rsi14<30 else "Neutral"))
    with m6: st.metric("MACD",           f"{macd_v:+.2f}",
                        "Bullish" if macd_v>0 else "Bearish")
    with m7: st.metric("Avg Vol",        f"{avg_vol/1000:.1f}K" if avg_vol>=1000 else f"{avg_vol:.0f}")

    # ══════════════════════════════════════════════════════════════
    #  52-WEEK RANGE VISUAL
    # ══════════════════════════════════════════════════════════════
    pct = min(max(pos_52w * 100, 0), 100)
    st.markdown(f"""
    <div class='wk52-wrap'>
        <div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>
            <span style='font-size:11px;color:#64748B;font-family:IBM Plex Mono,monospace'>
                52-Week Range
            </span>
            <span style='font-size:11px;color:#94A3B8;font-family:IBM Plex Mono,monospace'>
                Current price at <b style='color:#38BDF8'>{pct:.0f}th percentile</b>
            </span>
        </div>
        <div style='position:relative;background:#1E2A3A;border-radius:4px;height:8px;margin:.5rem 0'>
            <div style='height:8px;width:100%;border-radius:4px;
                 background:linear-gradient(90deg,#EF4444 0%,#F59E0B 50%,#10B981 100%)'></div>
            <div style='position:absolute;top:-5px;left:{pct}%;transform:translateX(-50%);
                 width:18px;height:18px;background:#070B14;border:2px solid #38BDF8;
                 border-radius:50%'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:.25rem'>
            <span style='font-size:11px;color:#EF4444;font-family:IBM Plex Mono,monospace'>
                Low: NPR {low_52w:,.2f}
            </span>
            <span style='font-size:11px;color:#94A3B8;font-family:IBM Plex Mono,monospace'>
                NPR {latest_close:,.2f}
            </span>
            <span style='font-size:11px;color:#10B981;font-family:IBM Plex Mono,monospace'>
                High: NPR {high_52w:,.2f}
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  PREDICTION BOX
    # ══════════════════════════════════════════════════════════════
    pred_color = "#10B981" if tom_chg >= 0 else "#F87171"
    st.markdown(f"""
    <div class='pred-wrap'>
        <div class='pred-lbl'>
            🤖 AI Price Prediction &nbsp;·&nbsp; Next Trading Day:
            {next_trade.strftime('%A, %d %b %Y')}
        </div>
        <div style='display:flex;align-items:baseline;gap:16px;flex-wrap:wrap'>
            <span class='pred-val'>NPR {tom_price:,.2f}</span>
            <span style='font-size:22px;font-family:IBM Plex Mono,monospace;
                  color:{pred_color}'>
                {tom_arrow} {abs(tom_pct):.2f}%
            </span>
        </div>
        <div class='pred-sub'>
            Expected change: <b>NPR {tom_chg:+.2f}</b> from today's close of
            NPR {latest_close:,.2f} &nbsp;·&nbsp;
            Model: {best_model} &nbsp;·&nbsp;
            Accuracy: ±NPR {results.get(best_model,{}).get('RMSE',0):.2f} RMSE
        </div>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  FORECAST CARDS
    # ══════════════════════════════════════════════════════════════
    if fdays > 1 and fcast:
        st.markdown(f"<div class='sec-hdr'>{fdays}-Day Price Forecast</div>",
                    unsafe_allow_html=True)
        cards = "<div class='fc-row'>"
        for i, f in enumerate(fcast):
            pp = fcast[i-1]["price"] if i > 0 else latest_close
            d  = f["price"] - pp
            dp = (d / pp * 100) if pp else 0
            dc = "fc-up" if d >= 0 else "fc-dn"
            da = "▲" if d >= 0 else "▼"
            cards += (f"<div class='fc-card'>"
                      f"<div class='fc-d'>{f['date'].strftime('%a %d %b')}</div>"
                      f"<div class='fc-p'>NPR {f['price']:,.0f}</div>"
                      f"<div class='{dc}'>{da} {abs(dp):.1f}%</div>"
                      f"</div>")
        cards += "</div>"
        st.markdown(cards, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  MAIN TABS
    # ══════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈  Price & Forecast",
        "📊  Technical Analysis",
        "🏦  Market Insights",
        "🤖  AI Model",
        "📋  Statistics",
    ])

    # ── Tab 1: Price & Forecast ───────────────────────────────────
    with tab1:
        st.markdown("<div class='sec-hdr'>Price Chart</div>", unsafe_allow_html=True)
        if chart_type == "Candlestick":
            st.pyplot(candlestick_chart(stock_df, min(days, 120)), use_container_width=True)
            st.caption("Candlestick view limited to 120 days for readability.")
        else:
            st.pyplot(price_chart(stock_df, days), use_container_width=True)

        if fcast:
            st.markdown("<div class='sec-hdr'>AI Forecast Chart</div>", unsafe_allow_html=True)
            st.pyplot(forecast_chart(stock_df, fcast, latest_close, latest_date),
                      use_container_width=True)
            st.markdown(
                "<p style='font-size:11px;color:#334155;font-family:IBM Plex Mono,monospace'>"
                "⚠ Forecast applies NEPSE ±10% circuit breaker on each day. "
                "Shaded band shows expanding uncertainty. Not financial advice.</p>",
                unsafe_allow_html=True)

    # ── Tab 2: Technical Analysis ─────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec-hdr'>RSI Indicator</div>", unsafe_allow_html=True)
            st.pyplot(rsi_chart(stock_df, days), use_container_width=True)
            st.markdown(
                "<div style='font-size:11px;color:#475569;font-family:IBM Plex Mono,monospace;"
                "background:#0D1117;border:1px solid #1E2A3A;border-radius:8px;padding:.6rem .9rem'>"
                "<b style='color:#94A3B8'>RSI Guide</b><br>"
                "Below 30 → Oversold, potential buy opportunity<br>"
                "30–70 → Normal trading range<br>"
                "Above 70 → Overbought, exercise caution"
                "</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='sec-hdr'>MACD</div>", unsafe_allow_html=True)
            st.pyplot(macd_chart(stock_df, days), use_container_width=True)
            st.markdown(
                "<div style='font-size:11px;color:#475569;font-family:IBM Plex Mono,monospace;"
                "background:#0D1117;border:1px solid #1E2A3A;border-radius:8px;padding:.6rem .9rem'>"
                "<b style='color:#94A3B8'>MACD Guide</b><br>"
                "Green (above 0) → Bullish momentum<br>"
                "Red (below 0) → Bearish momentum<br>"
                "Crossover at 0 → Potential trend reversal"
                "</div>", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Bollinger Bands Summary</div>", unsafe_allow_html=True)
        bb_pct = min(max((latest_close - bb_dn) / (bb_up - bb_dn + 1e-9) * 100, 0), 100)
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1: st.metric("BB Upper",  f"NPR {bb_up:,.2f}")
        with bcol2: st.metric("BB Middle", f"NPR {ma20:,.2f}")
        with bcol3: st.metric("BB Lower",  f"NPR {bb_dn:,.2f}")
        with bcol4: st.metric("BB Width",  f"{((bb_up-bb_dn)/ma20*100):.2f}%",
                               "Wide (volatile)" if (bb_up-bb_dn)/ma20 > 0.1 else "Narrow (calm)")

    # ── Tab 3: Market Insights ────────────────────────────────────
    with tab3:
        left, right = st.columns([1, 2])
        with left:
            # Signal breakdown
            st.markdown("<div class='sec-hdr'>Signal Analysis</div>", unsafe_allow_html=True)
            signals_html = ""
            for lbl, val, sent in inv["signals"]:
                dot_cls = "sig-bull" if sent=="bullish" else ("sig-bear" if sent=="bearish" else "sig-neut")
                signals_html += (
                    f"<div class='signal-row'>"
                    f"<div class='sig-dot {dot_cls}'></div>"
                    f"<span class='sig-lbl'>{lbl}</span>"
                    f"<span class='sig-val'>{val}</span>"
                    f"</div>"
                )
            st.markdown(signals_html, unsafe_allow_html=True)

            # Sub-score breakdown
            st.markdown("<div class='sec-hdr'>Score Breakdown</div>", unsafe_allow_html=True)
            for name, val in inv["breakdown"].items():
                pct_bar = int(val / 25 * 100)
                bar_color = "#10B981" if pct_bar >= 60 else ("#F59E0B" if pct_bar >= 35 else "#EF4444")
                st.markdown(
                    f"<div style='margin:.35rem 0'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:11px;color:#64748B;font-family:IBM Plex Mono,monospace;"
                    f"margin-bottom:3px'><span>{name}</span><span>{val}/25</span></div>"
                    f"<div style='background:#1E2A3A;border-radius:3px;height:5px'>"
                    f"<div style='height:5px;width:{pct_bar}%;border-radius:3px;"
                    f"background:{bar_color}'></div></div></div>",
                    unsafe_allow_html=True)

        with right:
            st.markdown("<div class='sec-hdr'>Volume Analysis</div>", unsafe_allow_html=True)
            st.pyplot(volume_analysis_chart(stock_df, days), use_container_width=True)

        st.markdown("<div class='sec-hdr'>Daily Returns Distribution</div>",
                    unsafe_allow_html=True)
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            st.pyplot(returns_distribution_chart(stock_df), use_container_width=True)
        with rc2:
            df_r = make_features(stock_df)
            rets = df_r["Return_1d"].dropna() * 100
            pos_days  = int((rets > 0).sum())
            neg_days  = int((rets < 0).sum())
            circ_up   = int((rets >= 9.5).sum())
            circ_dn   = int((rets <= -9.5).sum())
            win_rate  = pos_days / max(len(rets), 1) * 100
            st.markdown(f"""
            <div class='stat-grid'>
                <div class='stat-item'>
                    <div class='stat-lbl'>Up Days</div>
                    <div class='stat-val' style='color:#10B981'>{pos_days}</div>
                </div>
                <div class='stat-item'>
                    <div class='stat-lbl'>Down Days</div>
                    <div class='stat-val' style='color:#F87171'>{neg_days}</div>
                </div>
                <div class='stat-item'>
                    <div class='stat-lbl'>Win Rate</div>
                    <div class='stat-val'>{win_rate:.1f}%</div>
                </div>
                <div class='stat-item'>
                    <div class='stat-lbl'>Avg Return</div>
                    <div class='stat-val'>{rets.mean():.2f}%</div>
                </div>
                <div class='stat-item'>
                    <div class='stat-lbl'>Upper Circuit</div>
                    <div class='stat-val' style='color:#10B981'>{circ_up}×</div>
                </div>
                <div class='stat-item'>
                    <div class='stat-lbl'>Lower Circuit</div>
                    <div class='stat-val' style='color:#F87171'>{circ_dn}×</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 4: AI Model ───────────────────────────────────────────
    with tab4:
        if results:
            st.markdown("<div class='sec-hdr'>Model Accuracy (Historical Test Period)</div>",
                        unsafe_allow_html=True)
            st.pyplot(model_comparison_chart(dates_te, yte, {
                k: v for k, v in preds.items()
                if (k=="Ridge" and True) or (k=="Random Forest" and True) or (k=="XGBoost" and XGB_OK)
            }), use_container_width=True)

            st.markdown("<div class='sec-hdr'>Performance Metrics</div>",
                        unsafe_allow_html=True)
            mc = st.columns(len(results))
            for col, (name, m) in zip(mc, results.items()):
                with col:
                    best_flag = "⭐ " if name == best_model else ""
                    st.markdown(
                        f"<div style='background:#0D1117;border:1px solid "
                        f"{'#1D4ED8' if name==best_model else '#1E2A3A'};"
                        f"border-radius:12px;padding:1rem 1.25rem;text-align:center'>"
                        f"<p style='font-family:IBM Plex Mono,monospace;font-size:12px;"
                        f"color:#94A3B8;margin-bottom:.75rem'>{best_flag}{name}</p>",
                        unsafe_allow_html=True)
                    st.metric("MAE",  f"NPR {m['MAE']:.2f}")
                    st.metric("RMSE", f"NPR {m['RMSE']:.2f}")
                    st.metric("R²",   f"{m['R²']:.4f}")
                    st.metric("MAPE", f"{m['MAPE']:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

            # Feature importance
            if rf_full:
                st.markdown("<div class='sec-hdr'>What Drives the Prediction</div>",
                            unsafe_allow_html=True)
                imp = pd.Series(rf_full.feature_importances_, index=FEATURES).sort_values()
                top = imp.tail(10)
                fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
                bars = ax.barh(top.index, top.values, color=BL, alpha=0.8, height=0.55)
                for bar, val in zip(bars, top.values):
                    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                            f"{val:.3f}", va="center", fontsize=8, color=TXT)
                ax.set_xlabel("Importance Score", fontsize=9)
                for sp in ax.spines.values(): sp.set_edgecolor(GRID)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                st.markdown(
                    "<p style='font-size:11px;color:#334155;font-family:IBM Plex Mono,monospace'>"
                    "Higher score = more influence on the prediction. "
                    "Lag_1 (yesterday's price) is typically most important.</p>",
                    unsafe_allow_html=True)

    # ── Tab 5: Statistics ─────────────────────────────────────────
    with tab5:
        df_s = make_features(stock_df)
        rets = df_s["Return_1d"].dropna() * 100

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("<div class='sec-hdr'>Price Statistics</div>", unsafe_allow_html=True)
            st.metric("All-Time High (in DB)",  f"NPR {stock_df['High'].max():,.2f}")
            st.metric("All-Time Low (in DB)",   f"NPR {stock_df['Low'].min():,.2f}")
            st.metric("Average Close",          f"NPR {stock_df['Close'].mean():,.2f}")
            st.metric("Price Std Dev",          f"NPR {stock_df['Close'].std():,.2f}")
            st.metric("Total Sessions",         f"{n_rows}")
        with sc2:
            st.markdown("<div class='sec-hdr'>Return Statistics</div>", unsafe_allow_html=True)
            st.metric("Best Day",    f"+{rets.max():.2f}%")
            st.metric("Worst Day",   f"{rets.min():.2f}%")
            st.metric("Avg Daily",   f"{rets.mean():.3f}%")
            st.metric("Volatility",  f"{rets.std():.2f}%",
                      "Daily std deviation")
            st.metric("Sharpe-like", f"{(rets.mean()/rets.std()):.3f}" if rets.std()>0 else "—")
        with sc3:
            st.markdown("<div class='sec-hdr'>Volume Statistics</div>", unsafe_allow_html=True)
            st.metric("Latest Volume",  f"{int(stock_df['Volume'].iloc[-1]):,}")
            st.metric("Average Volume", f"{int(avg_vol):,}")
            st.metric("Max Volume",     f"{int(stock_df['Volume'].max()):,}")
            st.metric("Min Volume",     f"{int(stock_df['Volume'].min()):,}")
            st.metric("Data Since",     str(stock_df["Date"].min().date()))

        # Raw data table
        st.markdown("<div class='sec-hdr'>Price History</div>", unsafe_allow_html=True)
        show_cols = [c for c in ["Date","Open","High","Low","Close","Volume","Prev_Close"]
                     if c in stock_df.columns]
        st.dataframe(
            stock_df[show_cols].sort_values("Date", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=350,
        )

    # Disclaimer
    st.markdown(
        "<div class='disclaimer'>"
        "⚠ NEPSE Analytics is for informational purposes only. "
        "AI predictions are based on historical price patterns and do not constitute financial advice. "
        "Always consult a qualified financial advisor before making investment decisions. "
        "Past performance is not indicative of future results."
        "</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()