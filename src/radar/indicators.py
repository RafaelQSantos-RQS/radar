"""Indicadores técnicos para análise de minicontratos.

Todas as funções são **puras** — recebem um DataFrame e retornam
uma Series/DataFrame, sem efeitos colaterais e sem I/O.
"""

import pandas as pd

__all__ = [
    "compute_all",
    "compute_atr",
    "compute_bollinger",
    "compute_ema",
    "compute_macd",
    "compute_rsi",
    "compute_sma",
]


# ── Moving Averages ──────────────────────────────────────


def compute_sma(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
) -> pd.Series:
    """Simple Moving Average."""
    return df[column].rolling(window=period).mean()


def compute_ema(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
) -> pd.Series:
    """Exponential Moving Average (span-based)."""
    return df[column].ewm(span=period, adjust=False).mean()


# ── Oscillators ──────────────────────────────────────────


def compute_rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
) -> pd.Series:
    """Relative Strength Index com suavização de Wilder (vetorizada)."""
    delta = df[column].diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """MACD. Retorna DataFrame com ``macd``, ``signal`` e ``histogram``."""
    ema_fast = compute_ema(df, period=fast, column=column)
    ema_slow = compute_ema(df, period=slow, column=column)

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


# ── Volatility ───────────────────────────────────────────


def compute_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std: int = 2,
    column: str = "close",
) -> pd.DataFrame:
    """Bollinger Bands. Retorna ``bb_upper``, ``bb_middle``, ``bb_lower``."""
    middle = compute_sma(df, period=period, column=column)
    rolling_std = df[column].rolling(window=period).std()

    return pd.DataFrame({
        "bb_upper": middle + (rolling_std * std),
        "bb_middle": middle,
        "bb_lower": middle - (rolling_std * std),
    })


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Average True Range com suavização de Wilder (vetorizada)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ── Combined ─────────────────────────────────────────────


def compute_all(
    df: pd.DataFrame,
    sma_periods: list[int] | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Calcula o conjunto completo de indicadores e retorna cópia com colunas.

    Os nomes das colunas seguem o contrato usado pelas estratégias
    built-in (``rsi_14``, ``macd_signal``, ``macd_hist``, ``atr_14``).
    """
    result = df.copy()

    for p in sma_periods or [9, 20, 50]:
        result[f"sma_{p}"] = compute_sma(result, period=p)

    result["ema_9"] = compute_ema(result, period=9)
    result["rsi_14"] = compute_rsi(result, period=rsi_period)

    macd = compute_macd(result, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    result = pd.concat([result, macd.rename(columns={
        "signal": "macd_signal",
        "histogram": "macd_hist",
    })], axis=1)

    bb = compute_bollinger(result, period=bb_period)
    result = pd.concat([result, bb], axis=1)

    result["atr_14"] = compute_atr(result, period=atr_period)

    return result
