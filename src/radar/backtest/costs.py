"""Modelo de custos operacionais B3 completo.

Portado do ``order_book.py`` do b3-alpha-strategy. Cobre:

- **Abertura**: corretagem (comissão %), ISS (alíquota sobre corretagem),
  emolumentos (% do volume), registro (% do volume), custódia.
- **Fechamento**: corretagem, ISS, emolumentos, liquidação, registro,
  custódia + IRRF (retenção) e IR (imposto de renda) por day trade/swing.

Valores default seguem o projeto do usuário (b3-alpha-strategy) e são
configuráveis via ``CostParams``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "COST_MODELS",
    "DEFAULT_COST_PARAMS",
    "CostModel",
    "CostParams",
    "compute_entry_costs",
    "compute_exit_costs",
]


@dataclass
class CostModel:
    """Configuração de tick do ativo (para cálculo de P&L)."""

    tick_value: float = 5.00   # WDO$ = 0,5 pts = R$ 5,00
    tick_size: float = 0.5     # WDO$ = 0,5 pontos


COST_MODELS: dict[str, CostModel] = {
    "WDO$": CostModel(tick_value=5.00, tick_size=0.5),
    "WIN$": CostModel(tick_value=1.00, tick_size=5.0),
}


@dataclass
class CostParams:
    """Parâmetros de custos B3 (percentuais e alíquotas).

    Defaults do b3-alpha-strategy:
        comissao_pct=0.0, aliquota_iss=2.0, emolumentos_pct=0.005,
        liquidacao_pct=0.025, taxa_registro_pct=0.0, taxa_custodia=0.0,
        ir_swing_pct=15.0, ir_daytrade_pct=20.0,
        irrf_swing_pct=0.005, irrf_daytrade_pct=1.0
    """

    comissao_pct: float = 0.0
    aliquota_iss: float = 2.0
    emolumentos_pct: float = 0.005
    liquidacao_pct: float = 0.025
    taxa_registro_pct: float = 0.0
    taxa_custodia: float = 0.0
    ir_swing_pct: float = 15.0
    ir_daytrade_pct: float = 20.0
    irrf_swing_pct: float = 0.005
    irrf_daytrade_pct: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CostParams:
        """Constrói a partir de dict parcial (valores ausentes usam default)."""
        if not data:
            return cls()
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


DEFAULT_COST_PARAMS = CostParams()


def compute_entry_costs(
    price: float,
    volume: float,
    params: CostParams = DEFAULT_COST_PARAMS,
) -> dict[str, float]:
    """Custos de abertura de uma posição.

    Returns:
        Dict com ``commission``, ``iss``, ``emolumentos``, ``registro``,
        ``custodia`` e ``total`` (R$).
    """
    vol = price * volume

    commission = vol * (params.comissao_pct / 100.0)
    iss = commission * (params.aliquota_iss / 100.0) if commission > 0 else 0.0
    emolumentos = vol * (params.emolumentos_pct / 100.0)
    registro = vol * (params.taxa_registro_pct / 100.0)
    custodia = params.taxa_custodia

    total = commission + iss + emolumentos + registro + custodia
    return {
        "commission": round(commission, 2),
        "iss": round(iss, 2),
        "emolumentos": round(emolumentos, 2),
        "registro": round(registro, 2),
        "custodia": round(custodia, 2),
        "total": round(total, 2),
    }


def compute_exit_costs(
    open_price: float,
    close_price: float,
    volume: float,
    gross_pnl: float,
    is_daytrade: bool,
    vol_venda: float,
    params: CostParams = DEFAULT_COST_PARAMS,
) -> dict[str, float]:
    """Custos de fechamento de uma posição (operacionais + IRRF + IR).

    Args:
        open_price: Preço de entrada.
        close_price: Preço de saída.
        volume: Quantidade de contratos.
        gross_pnl: P&L bruto (antes de custos).
        is_daytrade: True se entrada e saída no mesmo dia.
        vol_venda: Volume financeiro da venda (close×qtd p/ compra,
            open×qtd p/ venda a descoberto).
        params: Parâmetros de custos.

    Returns:
        Dict com ``commission``, ``iss``, ``emolumentos``, ``liquidacao``,
        ``registro``, ``custodia``, ``irrf``, ``ir`` e ``total`` (R$).
    """
    vol_total = (open_price * volume) + (close_price * volume)

    commission = vol_total * (params.comissao_pct / 100.0)
    iss = commission * (params.aliquota_iss / 100.0) if commission > 0 else 0.0
    emolumentos = vol_total * (params.emolumentos_pct / 100.0)
    liquidacao = vol_total * (params.liquidacao_pct / 100.0)
    registro = vol_total * (params.taxa_registro_pct / 100.0)
    custodia = params.taxa_custodia

    operacionais = commission + iss + emolumentos + liquidacao + registro + custodia

    # IRRF (retenção na fonte)
    if is_daytrade:
        irrf = max(0.0, gross_pnl * (params.irrf_daytrade_pct / 100.0))
    else:
        irrf = vol_venda * (params.irrf_swing_pct / 100.0)

    # IR sobre o lucro após custos operacionais
    res_antes_ir = gross_pnl - operacionais
    aliquota_ir = params.ir_daytrade_pct if is_daytrade else params.ir_swing_pct
    ir = max(0.0, res_antes_ir * (aliquota_ir / 100.0))

    total = operacionais + irrf + ir
    return {
        "commission": round(commission, 2),
        "iss": round(iss, 2),
        "emolumentos": round(emolumentos, 2),
        "liquidacao": round(liquidacao, 2),
        "registro": round(registro, 2),
        "custodia": round(custodia, 2),
        "irrf": round(irrf, 2),
        "ir": round(ir, 2),
        "total": round(total, 2),
    }