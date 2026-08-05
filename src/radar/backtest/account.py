"""AccountInfo — estado da conta durante o backtest.

Mantém saldo, equity, margem e patrimônio histórico para cálculo
de métricas como drawdown e Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

__all__ = ["AccountInfo"]


@dataclass
class AccountInfo:
    """Estado completo da conta de trading.

    Attributes:
        initial_balance: Saldo inicial (R$).
        balance: Saldo atual (cash + P&L realizados).
        equity: Patrimônio atual (balance + P&L não realizado).
        margin: Margem utilizada em posições abertas.
        free_margin: Margem livre (equity - margin).
        profit: P&L flutuante atual.
        total_commission: Comissão acumulada.
        total_b3_fee: Taxas B3 (emolumentos + registro) acumuladas.
        total_slippage: Slippage acumulado.
        equity_curve: Série temporal do equity (para métricas).
    """

    initial_balance: float = 10_000.0
    balance: float = 10_000.0
    equity: float = 10_000.0
    margin: float = 0.0
    free_margin: float = 10_000.0
    profit: float = 0.0
    total_commission: float = 0.0
    total_b3_fee: float = 0.0
    total_slippage: float = 0.0
    equity_curve: list[dict] = field(default_factory=list)

    def snapshot(self, timestamp: datetime) -> None:
        """Registra o estado atual na equity_curve."""
        self.equity_curve.append({
            "time": timestamp,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "profit": round(self.profit, 2),
            "margin": round(self.margin, 2),
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Converte a equity_curve para DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)

    @property
    def total_costs(self) -> float:
        """Custos totais acumulados (R$)."""
        return self.total_commission + self.total_b3_fee + self.total_slippage

    @property
    def net_profit(self) -> float:
        """Lucro líquido total."""
        return self.balance - self.initial_balance

    @property
    def total_return_pct(self) -> float:
        """Retorno total percentual."""
        if self.initial_balance == 0:
            return 0.0
        return self.net_profit / self.initial_balance
