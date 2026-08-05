"""Modelo de custos operacionais da B3 para minicontratos.

Valores aproximados com base em taxas públicas da B3 e corretoras
(consulte sempre sua Nota de Corretagem para valores exatos).

┌─────────────────────────────────────────────────────────┐
│              REFERÊNCIA DE CUSTOS B3                    │
├──────────────┬──────────────┬──────────────┬────────────┤
│ Item         │ WDO$         │ WIN$         │ Fonte      │
├──────────────┼──────────────┼──────────────┼────────────┤
│ Tick mínimo  │ 0,5 pts      │ 5 pts        │ B3         │
│ Valor do tick│ R$ 5,00      │ R$ 1,00      │ B3         │
│ Corretagem*  │ R$ 0,90      │ R$ 0,90      │ Corretoras │
│ Taxas B3**   │ ~R$ 1,20/side│ ~R$ 0,30/side│ B3         │
│ Slippage***  │ 1 tick       │ 1 tick       │ Conservador│
└──────────────┴──────────────┴──────────────┴────────────┘

* Corretagem: muitas corretoras oferecem corretagem zero (Clear,
  XP, Modalmais para day trade). R$ 0,90 é estimativa conservadora.

** Taxas B3 (emolumentos + registro + liquidação):
   WDO$: ~R$ 1,20 por contrato por lado (ida ou volta)
   WIN$: ~R$ 0,30 por contrato por lado
   Fonte: b3.com.br/pt_br/produtos-e-servicos/tarifas/

*** Slippage: 1 tick é conservador para WDO$/WIN$ em M1.
    O valor do tick usado: WDO$ = R$ 5,00 / WIN$ = R$ 1,00.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["COST_MODELS", "compute_costs"]


@dataclass
class CostModel:
    """Configuração de custos para um ativo.

    Attributes:
        commission_per_order: Corretagem fixa por ordem (R$).
        b3_fee_per_contract: Taxas B3 fixas por contrato por lado (R$).
        slippage_ticks: Número de ticks de derrapagem na execução.
        tick_value: Valor monetário de 1 tick (R$).
        tick_size: Tamanho do tick em pontos.
    """

    commission_per_order: float = 0.90
    b3_fee_per_contract: float = 1.20
    slippage_ticks: int = 1
    tick_value: float = 5.00   # WDO$ = 0,5 pts = R$ 5,00
    tick_size: float = 0.5     # WDO$ = 0,5 pontos


# Modelos pré-definidos com valores aproximados da B3.
COST_MODELS: dict[str, CostModel] = {
    "WDO$": CostModel(
        commission_per_order=0.90,
        b3_fee_per_contract=1.20,  # ~R$ 1,20 por lado
        tick_value=5.00,            # 0,5 pts × R$ 5,00
        tick_size=0.5,
    ),
    "WIN$": CostModel(
        commission_per_order=0.90,
        b3_fee_per_contract=0.30,  # ~R$ 0,30 por lado
        tick_value=1.00,            # 5 pts × R$ 1,00
        tick_size=5.0,
    ),
}


def compute_costs(
    symbol: str,
    price: float,
    volume: float = 1.0,
) -> dict[str, float]:
    """Calcula custos totais para uma operação de entrada ou saída.

    Args:
        symbol: Símbolo do ativo (``"WDO$"`` ou ``"WIN$"``).
        price: Preço de execução em pontos (não usado no modelo de taxas fixas).
        volume: Quantidade de contratos.

    Returns:
        Dict com ``commission``, ``b3_fee``, ``slippage`` e ``total`` (R$).
    """
    model = COST_MODELS.get(symbol, COST_MODELS["WDO$"])

    commission = model.commission_per_order
    b3_fee = model.b3_fee_per_contract * volume
    slippage_cost = model.tick_value * model.slippage_ticks * volume
    total = commission + b3_fee + slippage_cost

    return {
        "commission": round(commission, 2),
        "b3_fee": round(b3_fee, 2),
        "slippage": round(slippage_cost, 2),
        "total": round(total, 2),
    }
