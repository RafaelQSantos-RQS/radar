"""Estratégia QuantScore — score composto 0–100 para apoio à decisão.

Portado do projeto ``b3-alpha-strategy`` (AlphaSimples_QuantScore_V2.0),
adaptado para o Radar (DataFrame OHLCV + TradeSim).

O score combina:
    - momentum (variação % no período)
    - força relativa vs IBOV (momentum do ativo − momentum do IBOV)
    - RSI(14)
    - preço vs EMA20

Regras de entrada:
    - COMPRA: ângulo de tendência em [ang_min_compra, ang_max_compra]
      e score >= min_score_compra
    - VENDA: ângulo em [-ang_max_venda, -ang_min_venda]
      e score <= max_score_venda

Filtro anti-evento corporativo: variação diária > 35% → score -1.0
(impede entrada em base não ajustada com split/desdobramento).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_PARAMS", "QuantScoreParams", "compute_score", "quantscore_strategy"]


@dataclass
class QuantScoreParams:
    """Parâmetros da estratégia QuantScore."""

    stop_inicial: float = 1.0
    take_inicial: float = 2.0
    min_score_compra: float = 60.0
    max_score_venda: float = 40.0
    capital_por_ordem: float = 1000.0
    ang_min_compra: float = 60.0
    ang_max_compra: float = 75.0
    ang_min_venda: float = 60.0
    ang_max_venda: float = 75.0
    lookback: int = 20
    ibov_symbols: tuple[str, ...] = ("IBOV", "IBOV11")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QuantScoreParams:
        """Constrói a partir de dict parcial (valores ausentes usam default)."""
        if not data:
            return cls()
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


DEFAULT_PARAMS = QuantScoreParams()


# ── Cálculo do score ────────────────────────────────────────


def _compute_angle(closes: np.ndarray) -> float:
    """Ângulo da regressão linear dos últimos 3 preços normalizados (graus)."""
    if len(closes) < 2:
        return 0.0
    window = closes[-3:] if len(closes) >= 3 else closes
    base = window[0]
    if base != 0:
        normalized = (window / base) * 100.0
    else:
        normalized = window.copy()
    x = np.arange(len(normalized))
    a, _ = np.polyfit(x, normalized, 1)
    return float(np.degrees(np.arctan(a)))


def _score_components(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    ibov_closes: np.ndarray | None,
) -> dict[str, float]:
    """Calcula os componentes do score para a última barra do array."""
    last_close = float(closes[-1]) if len(closes) > 0 else 1.0

    # ATR simples (média dos True Ranges)
    tr_list = []
    for i in range(1, len(closes)):
        h, l, cp = highs[i], lows[i], closes[i - 1]
        tr_list.append(max(h - l, abs(h - cp), abs(l - cp)))
    atr = float(np.mean(tr_list)) if tr_list else 0.0
    atr_pct = (atr / last_close) * 100.0 if last_close > 0 else 0.0

    avg_volume = float(np.mean(volumes)) if len(volumes) > 0 else 0.0
    liquidez_atr = (avg_volume * (last_close / 10.0)) / (atr_pct + 1e-5)

    # RSI simples (média dos últimos `period` ganhos/perdas)
    period = 14
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else (np.mean(gains) if len(gains) else 0.0)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else (np.mean(losses) if len(losses) else 0.0)
    if avg_loss == 0:
        rsi = 100.0
    else:
        rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    # EMA20 simples
    alpha = 2 / (20 + 1)
    ema20 = float(closes[0])
    for price in closes[1:]:
        ema20 = (price * alpha) + (ema20 * (1 - alpha))
    preco_vs_ema = ((last_close - ema20) / ema20) * 100.0 if ema20 > 0 else 0.0

    first_close = closes[0] if len(closes) > 0 else last_close
    momentum = ((last_close - first_close) / first_close) * 100.0 if first_close > 0 else 0.0

    retornos = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])
    volatilidade = float(np.std(retornos) * 100.0)

    # Filtro anti-evento corporativo (split/desdobramento em base não ajustada)
    salto_maximo = float(np.max(np.abs(retornos)) * 100.0) if len(retornos) > 0 else 0.0
    teve_evento_corporativo = salto_maximo > 35.0

    fr_ibov = 0.0
    if ibov_closes is not None and len(ibov_closes) >= len(closes):
        ret_ibov = (
            ((ibov_closes[-1] - ibov_closes[0]) / ibov_closes[0]) * 100.0
            if ibov_closes[0] > 0
            else 0.0
        )
        fr_ibov = momentum - ret_ibov

    score = 50.0
    if teve_evento_corporativo:
        score = -1.0
    else:
        if momentum > 0:
            score += min(momentum * 2, 15.0)
        else:
            score -= min(abs(momentum) * 2, 15.0)

        if fr_ibov > 0:
            score += min(fr_ibov * 1.5, 15.0)
        else:
            score -= min(abs(fr_ibov) * 1.5, 15.0)

        if 45 <= rsi <= 65:
            score += 10.0
        elif rsi > 70:
            score -= 10.0
        elif rsi < 30:
            score += 5.0

        if preco_vs_ema > 0:
            score += 10.0
        else:
            score -= 10.0

        score = max(0.0, min(100.0, score))

    return {
        "atr_pct": round(atr_pct, 4),
        "avg_volume": round(avg_volume, 2),
        "liquidez_atr": round(liquidez_atr, 2),
        "rsi": round(rsi, 2),
        "ema20": round(ema20, 4),
        "preco_vs_ema": round(preco_vs_ema, 2),
        "momentum": round(momentum, 2),
        "volatilidade": round(volatilidade, 2),
        "fr_ibov": round(fr_ibov, 2),
        "salto_maximo": round(salto_maximo, 2),
        "teve_evento_corporativo": teve_evento_corporativo,
        "score_final": round(score, 2),
    }


def compute_score(
    df: pd.DataFrame,
    ibov_df: pd.DataFrame | None = None,
    lookback: int = 20,
) -> dict[str, Any]:
    """Calcula o score QuantScore para a última barra do DataFrame.

    Args:
        df: DataFrame OHLCV (colunas ``open, high, low, close, volume``).
        ibov_df: DataFrame OHLCV do IBOV (opcional, para força relativa).
        lookback: Número de barras da janela de cálculo.

    Returns:
        Dict com componentes do score e ``score_final`` (0–100 ou -1.0
        se evento corporativo).
    """
    if df.empty:
        raise ValueError("DataFrame vazio")

    window = df.tail(lookback)
    closes = window["close"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    volumes = window["volume"].to_numpy(dtype=float)

    ibov_closes = None
    if ibov_df is not None and not ibov_df.empty:
        ibov_closes = ibov_df.tail(lookback)["close"].to_numpy(dtype=float)

    components = _score_components(closes, highs, lows, volumes, ibov_closes)
    components["angulo"] = round(_compute_angle(closes), 4)
    return components


# ── Strategy_fn para o TradeSim ─────────────────────────────


def quantscore_strategy(
    params: QuantScoreParams | dict[str, Any] | None = None,
    ibov_df: pd.DataFrame | None = None,
) -> Callable[[int, pd.DataFrame], dict]:
    """Cria a função de estratégia QuantScore compatível com ``TradeSim.run``.

    A função avalia o score na barra ``idx`` e retorna:
        - ``{"direction": "buy"|"sell", "sl", "tp", "volume"}`` quando a
          regra de entrada é satisfeita
        - ``{"direction": "hold"}`` caso contrário

    SL/TP são percentuais sobre o preço de entrada (open da barra).
    """
    p = params if isinstance(params, QuantScoreParams) else QuantScoreParams.from_dict(params)
    p = p or DEFAULT_PARAMS

    def strategy_fn(idx: int, df: pd.DataFrame) -> dict:
        if idx < 1 or idx >= len(df):
            return {"direction": "hold"}

        window = df.iloc[max(0, idx - p.lookback + 1): idx + 1]
        closes = window["close"].to_numpy(dtype=float)
        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)
        volumes = window["volume"].to_numpy(dtype=float)

        ibov_closes = None
        if ibov_df is not None and not ibov_df.empty:
            ibov_closes = ibov_df.iloc[max(0, idx - p.lookback + 1): idx + 1]["close"].to_numpy(dtype=float)

        comp = _score_components(closes, highs, lows, volumes, ibov_closes)
        score = comp["score_final"]
        angulo = _compute_angle(closes)

        if score == -1.0:
            return {"direction": "hold"}

        bar = df.iloc[idx]
        open_price = float(bar["open"])
        if open_price <= 0:
            return {"direction": "hold"}

        volume = max(1, int(p.capital_por_ordem / open_price))

        if p.ang_min_compra <= angulo <= p.ang_max_compra and score >= p.min_score_compra:
            sl = round(open_price * (1 - (p.stop_inicial / 100.0)), 2)
            tp = round(open_price * (1 + (p.take_inicial / 100.0)), 2)
            return {"direction": "buy", "sl": sl, "tp": tp, "volume": volume}

        if -p.ang_max_venda <= angulo <= -p.ang_min_venda and score <= p.max_score_venda:
            sl = round(open_price * (1 + (p.stop_inicial / 100.0)), 2)
            tp = round(open_price * (1 - (p.take_inicial / 100.0)), 2)
            return {"direction": "sell", "sl": sl, "tp": tp, "volume": volume}

        return {"direction": "hold"}

    return strategy_fn