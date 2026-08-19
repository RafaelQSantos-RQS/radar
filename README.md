# Radar

Motor de apoio à decisão para day trading na B3 (minicontratos WDO$/WIN$ e ações).
API-first em FastAPI, dados em DuckDB, estratégia única **QuantScore**.

## Arquitetura

```
src/radar/
├── api/            # FastAPI: schemas, service, router, app
├── backtest/       # TradeSim (barra a barra), custos B3 completos
├── benchmarks.py   # SELIC, IPCA, Ibovespa (dados seed)
├── data/           # ingestão de candles (CSV → DuckDB)
├── indicators.py   # RSI, EMA, ATR
├── metrics.py      # Sharpe, MDD, win rate, profit factor
├── optimizer/      # grid paramétrico + jobs assíncronos
└── strategy/       # QuantScore (score composto 0–100)
```

## Estratégia QuantScore

Score composto 0–100 combinando momentum, força relativa vs IBOV, RSI(14) e
preço vs EMA20, com filtro anti-evento corporativo (variação > 35% → score -1.0).

- **COMPRA**: ângulo de tendência em `[ang_min_compra, ang_max_compra]` e score ≥ `min_score_compra`
- **VENDA**: ângulo em `[-ang_max_venda, -ang_min_venda]` e score ≤ `max_score_venda`
- **Saída**: BCT (Breakout Ladder) — trailing stop dinâmico com virtual TP escalado
- **IBOV** é excluído como alvo de entrada (benchmark)

## Custos B3 completos

Corretagem, ISS, emolumentos, liquidação, registro, custódia, IRRF e IR
(day trade 20% / swing 15%), aplicados na abertura e fechamento de cada posição.
Configuráveis via `costs_params` (defaults da B3).

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| POST | `/v1/ingest` | Sobe candles (CSV) para o DuckDB |
| POST | `/v1/signal` | Avalia QuantScore na última barra |
| POST | `/v1/backtest` | Backtest QuantScore com custos + benchmarks |
| POST | `/v1/optimize` | Cria job de otimização por grid paramétrico |
| GET | `/v1/jobs/{id}` | Consulta status/resultado do job |

## Uso rápido

```bash
# Ingestão
curl -F "file=@candles.csv" -F "symbol=WDO$" -F "timeframe=M1" \
  http://localhost:8000/v1/ingest

# Sinal
curl -X POST http://localhost:8000/v1/signal \
  -H "Content-Type: application/json" \
  -d '{"symbol": "WDO$", "timeframe": "M1"}'

# Backtest
curl -X POST http://localhost:8000/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol": "WDO$", "timeframe": "M1"}'

# Otimização
curl -X POST http://localhost:8000/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{"symbol": "WDO$", "timeframe": "M1",
       "param_grid": {"stop_inicial": [1.0, 2.0], "min_score_compra": [60, 70]}}'
```

## Testes

```bash
uv run pytest
uv run ruff check src tests
```