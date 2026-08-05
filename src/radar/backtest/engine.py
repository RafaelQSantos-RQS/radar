"""TradeSim — simulador de trades barra a barra.

O TradeSim:

- Abre/fecha posições com preços reais
- Gerencia Stop Loss e Take Profit fixos
- Atualiza trailing stop e break-even
- Deduz custos B3 (corretagem, taxas B3, slippage)
- Mantém AccountInfo (saldo, equity, margem, drawdown)
- Gera equity_curve para cálculo de Sharpe, MDD, etc.

Uso:
    sim = TradeSim(initial_balance=10000, symbol="WDO$")
    result = sim.run(df, strategy_fn)
    print(result["summary"])
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from radar.backtest.account import AccountInfo
from radar.backtest.costs import COST_MODELS, compute_costs

__all__ = ["Order", "OrderStatus", "OrderType", "TradeSim"]


class OrderType(Enum):
    """Direção da ordem."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Estado da ordem."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Representa uma posição/ordem no backtest.

    Attributes:
        ticket: Número identificador único.
        symbol: Símbolo do ativo.
        order_type: BUY ou SELL.
        volume: Quantidade de contratos.
        open_price: Preço de entrada (pontos).
        open_time: Data/hora da entrada.
        sl: Stop Loss (preço em pontos, 0 = não definido).
        tp: Take Profit (preço em pontos, 0 = não definido).
        close_price: Preço de saída.
        close_time: Data/hora da saída.
        close_reason: Motivo do fechamento ("sl", "tp", "signal", "trailing", "eod").
        gross_pnl: P&L bruto em R$ (sem custos).
        commission: Corretagem paga (R$) — entrada + saída.
        b3_fee: Taxas B3 (emolumentos + registro) pagas (R$) — entrada + saída.
        slippage: Custo de slippage (R$) — entrada + saída.
        status: Estado atual da ordem.
    """
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float = 1.0
    open_price: float = 0.0
    open_time: datetime | None = None
    sl: float = 0.0
    tp: float = 0.0
    trailing_activation: float = 0.0
    close_price: float | None = None
    close_time: datetime | None = None
    close_reason: str | None = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    b3_fee: float = 0.0
    slippage: float = 0.0
    status: OrderStatus = OrderStatus.OPEN

    @property
    def net_pnl(self) -> float:
        """P&L líquido (já descontando custos de entrada e saída)."""
        return self.gross_pnl - self.commission - self.b3_fee - self.slippage

    @property
    def is_open(self) -> bool:
        return self.status == OrderStatus.OPEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "direction": self.order_type.value,
            "volume": self.volume,
            "open_price": self.open_price,
            "open_time": self.open_time,
            "close_price": self.close_price,
            "close_time": self.close_time,
            "close_reason": self.close_reason,
            "sl": self.sl,
            "tp": self.tp,
            "gross_pnl": round(self.gross_pnl, 2),
            "commission": round(self.commission, 2),
            "b3_fee": round(self.b3_fee, 2),
            "slippage": round(self.slippage, 2),
            "net_pnl": round(self.net_pnl, 2),
            "status": self.status.value,
        }


class TradeSim:
    """Simulador de trades completo.

    Args:
        initial_balance: Saldo inicial em R$.
        symbol: Símbolo do ativo (para modelo de custos).
        trailing_distance: Distância do trailing stop em pontos.
        use_costs: Ativar/desativar custos.
        max_positions: Número máximo de posições simultâneas.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        symbol: str = "WDO$",
        trailing_distance: float = 0.0,
        use_costs: bool = True,
        max_positions: int = 1,
    ):
        self.account = AccountInfo(initial_balance=initial_balance)
        self.symbol = symbol
        self.trailing_distance = trailing_distance
        self.use_costs = use_costs
        self.max_positions = max_positions
        self.tick_value = COST_MODELS.get(symbol, COST_MODELS["WDO$"]).tick_value

        self.open_orders: dict[int, Order] = {}
        self.closed_orders: list[Order] = []
        self._ticket_counter = 0
        self._bankrupt = False
        self._debug = False

    # ── Gestão de ordens ────────────────────────────────────

    def _next_ticket(self) -> int:
        self._ticket_counter += 1
        return self._ticket_counter

    def open_order(
        self,
        order_type: OrderType,
        price: float,
        timestamp: datetime,
        volume: float = 1.0,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> int | None:
        """Abre uma nova ordem.

        Returns:
            Ticket da ordem, ou None se excedeu max_positions.
        """
        if len(self.open_orders) >= self.max_positions:
            return None

        # Deduzir custos de entrada
        entry_costs = compute_costs(self.symbol, price, volume) if self.use_costs else {}

        ticket = self._next_ticket()
        order = Order(
            ticket=ticket,
            symbol=self.symbol,
            order_type=order_type,
            volume=volume,
            open_price=price,
            open_time=timestamp,
            sl=sl,
            tp=tp,
            commission=entry_costs.get("commission", 0),
            b3_fee=entry_costs.get("b3_fee", 0),
            slippage=entry_costs.get("slippage", 0),
        )

        self.open_orders[ticket] = order
        self.account.total_commission += order.commission
        self.account.total_b3_fee += order.b3_fee
        self.account.total_slippage += order.slippage

        if self._debug:
            print(f"  [{timestamp}] ABRIR {order_type.value.upper()} {volume}@ {price:.2f} "
                  f"(SL={sl:.2f}, TP={tp:.2f}) ticket={ticket}")

        return ticket

    def close_order(
        self,
        ticket: int,
        price: float,
        timestamp: datetime,
        reason: str = "signal",
    ) -> None:
        """Fecha uma ordem aberta."""
        order = self.open_orders.pop(ticket, None)
        if order is None:
            return

        order.close_price = price
        order.close_time = timestamp
        order.close_reason = reason

        # Calcular P&L bruto
        price_diff = price - order.open_price
        if order.order_type == OrderType.SELL:
            price_diff = -price_diff
        order.gross_pnl = price_diff * self.tick_value * order.volume

        # Custos de saída — TODOS debitados na ordem (commission, b3_fee,
        # slippage). Corrige bug: antes commission/b3_fee iam só para o
        # account, fazendo o net_pnl da ordem divergir do net_profit da
        # conta e inflando win rate / profit factor.
        exit_costs = compute_costs(self.symbol, price, order.volume) if self.use_costs else {}
        order.slippage += exit_costs.get("slippage", 0)
        order.commission += exit_costs.get("commission", 0)
        order.b3_fee += exit_costs.get("b3_fee", 0)
        self.account.total_slippage += exit_costs.get("slippage", 0)
        self.account.total_commission += exit_costs.get("commission", 0)
        self.account.total_b3_fee += exit_costs.get("b3_fee", 0)

        # Atualizar saldo
        self.account.balance += order.net_pnl

        order.status = OrderStatus.CLOSED
        self.closed_orders.append(order)

        if self._debug:
            direction = "▲" if order.net_pnl > 0 else "▼"
            print(f"  [{timestamp}] FECHAR ticket={ticket} "
                  f"{direction} P&L={order.gross_pnl:+.2f} custos={order.commission+order.b3_fee+order.slippage:.2f} "
                  f"líquido={order.net_pnl:+.2f} motivo={reason}")

    # ── Verificação SL/TP ───────────────────────────────────

    def _check_sl_tp(self, bar: pd.Series) -> None:
        """Verifica se alguma ordem aberta atingiu SL ou TP."""
        high = float(bar["high"])
        low = float(bar["low"])

        for ticket in list(self.open_orders.keys()):
            order = self.open_orders[ticket]

            if order.order_type == OrderType.BUY:
                # Stop Loss: low <= SL
                if order.sl > 0 and low <= order.sl:
                    self.close_order(ticket, order.sl, bar["time"], "sl")
                # Take Profit: high >= TP
                elif order.tp > 0 and high >= order.tp:
                    self.close_order(ticket, order.tp, bar["time"], "tp")
            else:  # SELL
                # Stop Loss: high >= SL
                if order.sl > 0 and high >= order.sl:
                    self.close_order(ticket, order.sl, bar["time"], "sl")
                # Take Profit: low <= TP
                elif order.tp > 0 and low <= order.tp:
                    self.close_order(ticket, order.tp, bar["time"], "tp")

    def _update_trailing_stop(self, bar: pd.Series) -> None:
        """Atualiza trailing stop e break-even para ordens abertas."""
        if self.trailing_distance <= 0:
            return

        high = float(bar["high"])
        low = float(bar["low"])

        for order in self.open_orders.values():
            break_even_buffer = self.trailing_distance * 0.3  # 30% para BE

            if order.order_type == OrderType.BUY:
                # Break-even: lucro suficiente para cobrir custos
                cost_pts = (order.commission + order.b3_fee + order.slippage) / (
                    self.tick_value * order.volume
                )
                be_level = order.open_price + cost_pts + break_even_buffer
                if high >= be_level and order.sl < order.open_price:
                    order.sl = order.open_price + cost_pts  # Sobe para break-even
                    if self._debug:
                        print(f"  [{bar['time']}] Break-even ativado ticket={order.ticket}")

                # Trailing: stop sobe acompanhando
                novo_sl = high - self.trailing_distance
                if novo_sl > order.sl:
                    order.sl = novo_sl
                    if self._debug:
                        print(f"  [{bar['time']}] Trailing atualizado ticket={order.ticket} sl={order.sl:.2f}")

            else:  # SELL
                cost_pts = (order.commission + order.b3_fee + order.slippage) / (
                    self.tick_value * order.volume
                )
                be_level = order.open_price - cost_pts - break_even_buffer
                if low <= be_level and order.sl > order.open_price:
                    order.sl = order.open_price - cost_pts

                novo_sl = low + self.trailing_distance
                if novo_sl < order.sl or order.sl == 0:
                    order.sl = novo_sl

    # ── Loop principal ──────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable[[int, pd.DataFrame], dict],
        warmup: int = 60,
    ) -> dict[str, Any]:
        """Executa a simulação barra a barra.

        Args:
            df: DataFrame com OHLCV + indicadores (ordenado por time).
            strategy_fn: Função ``(idx, df) -> dict`` que retorna:
                - direction: ``"buy"`` | ``"sell"`` | ``"hold"``
                - sl: stop loss em pontos (opcional)
                - tp: take profit em pontos (opcional)
                - volume: contratos (opcional, default 1)
            warmup: Número de barras iniciais para aquecimento
                (indicadores não estão estáveis).

        Returns:
            Dict com ``summary``, ``trades``, ``equity_curve``,
            ``account`` e ``error``.
        """
        if df.empty:
            return self._result("DataFrame vazio.")

        n_bars = len(df)
        last_idx = n_bars - 1

        for idx in range(n_bars):
            bar = df.iloc[idx]
            timestamp = bar["time"]

            if idx < warmup:
                self.account.snapshot(timestamp)
                continue

            # 1. Verificar SL/TP
            self._check_sl_tp(bar)

            # 2. Atualizar trailing stop
            self._update_trailing_stop(bar)

            # ── Proteção contra falência ────────────────────
            if self.account.balance <= 0 and not self.open_orders:
                if not self._bankrupt:
                    self._bankrupt = True
                    if self._debug:
                        print(f"  [{timestamp}] 🔴 FALÊNCIA — balance={self.account.balance:.2f}")
                direction = "hold"
                signal = {"direction": "hold"}
            else:
                # 3. Chamar estratégia
                try:
                    signal = strategy_fn(idx, df)
                except Exception as e:  # noqa: BLE001 — uma falha de estratégia não pode derrubar o backtest
                    if self._debug:
                        print(f"  [ERRO] strategy_fn falhou no índice {idx}: {e}")
                    signal = {"direction": "hold"}

                direction = signal.get("direction", "hold")

                # 4. Processar sinal (abrir posição se for buy/sell)
                if direction in ("buy", "sell") and not self.open_orders:
                    order_type = OrderType.BUY if direction == "buy" else OrderType.SELL
                    price = float(bar["close"])
                    volume = float(signal.get("volume", 1.0))
                    sl = float(signal.get("sl", 0.0))
                    tp = float(signal.get("tp", 0.0))

                    self.open_order(
                        order_type=order_type,
                        price=price,
                        timestamp=timestamp,
                        volume=volume,
                        sl=sl,
                        tp=tp,
                    )

                # 5. Sinal contrário fecha a posição aberta
                #    (corrige bug: branch antigo era dead code — `pass` —
                #     posição nunca fechava por sinal, só por SL/TP/EOD)
                elif direction in ("buy", "sell") and self.open_orders:
                    for ticket in list(self.open_orders.keys()):
                        order = self.open_orders[ticket]
                        wants = OrderType.BUY if direction == "buy" else OrderType.SELL
                        if order.order_type != wants:
                            self.close_order(ticket, float(bar["close"]), timestamp, "signal")

                # 6. Forçar fechamento no último candle
                if idx == last_idx:
                    for ticket in list(self.open_orders.keys()):
                        self.close_order(ticket, float(bar["close"]), timestamp, "eod")

            # 7. Atualizar P&L flutuante e equity
            floating = 0.0
            margin = 0.0
            for order in self.open_orders.values():
                price_diff = float(bar["close"]) - order.open_price
                if order.order_type == OrderType.SELL:
                    price_diff = -price_diff
                floating += price_diff * self.tick_value * order.volume
                margin += order.volume * order.open_price * 0.05  # ~5% margem

            self.account.profit = floating
            self.account.margin = margin
            self.account.equity = self.account.balance + floating
            self.account.free_margin = self.account.equity - margin

            # 8. Snapshot
            self.account.snapshot(timestamp)

        return self._result()

    def _result(self, error: str | None = None) -> dict[str, Any]:
        """Compila o resultado final."""
        trades = [t.to_dict() for t in self.closed_orders]

        if not trades:
            return {
                "summary": {
                    "symbol": self.symbol,
                    "initial_balance": self.account.initial_balance,
                    "final_balance": self.account.balance,
                    "net_profit": self.account.net_profit,
                    "total_return_pct": self.account.total_return_pct,
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_costs": self.account.total_costs,
                },
                "trades": [],
                "equity_curve": self.account.to_dataframe(),
                "error": error,
            }

        winners = [t for t in trades if t["net_pnl"] > 0]
        losers = [t for t in trades if t["net_pnl"] < 0]
        total_trades = len(trades)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(t["net_pnl"] for t in winners)
        gross_loss = sum(abs(t["net_pnl"]) for t in losers)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.999 if gross_profit > 0 else 0.0)

        return {
                "summary": {
                    "symbol": self.symbol,
                    "initial_balance": self.account.initial_balance,
                    "final_balance": round(self.account.balance, 2),
                    "net_profit": round(self.account.net_profit, 2),
                    "total_return_pct": round(self.account.total_return_pct, 4),
                    "total_trades": total_trades,
                    "win_rate": round(win_rate, 4),
                    "profit_factor": round(profit_factor, 4),
                    "total_costs": round(self.account.total_costs, 2),
                    "total_commission": round(self.account.total_commission, 2),
                    "total_b3_fee": round(self.account.total_b3_fee, 2),
                    "total_slippage": round(self.account.total_slippage, 2),
                    "bankrupt": self._bankrupt,
                },
                "trades": trades,
                "equity_curve": self.account.to_dataframe(),
                "error": error,
            }
