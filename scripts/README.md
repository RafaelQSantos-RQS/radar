# scripts/

Utilitários de apoio ao Radar. Todos os scripts que envolvem o MetaTrader 5
rodam no Windows (onde o MT5 é instalado) e produzem CSVs para a pasta
`dump/` do Radar.

## ExportTicks1s.mq5 — exportar dados em grão de 1 segundo

O MT5 guarda **ticks** (menor grão disponível) e o script os agrega em
**barras OHLCV de 1 segundo**, no formato que o Radar ingere:

```
date;time;open;high;low;close;volume
2025.01.02;09:00:01;5234.5;5235.0;5234.0;5234.5;12
```

### Instalação (no Windows)

1. Abra o MetaTrader 5 → menu `Arquivo` → `Abrir pasta de dados`.
2. Copie `ExportTicks1s.mq5` para `MQL5/Scripts/`.
3. No MT5, pressione `F4` (MetaEditor) e depois `F7` para compilar o script.
4. No `Navegador` → `Scripts`, arraste `ExportTicks1s` para qualquer gráfico.

### Uso

Configure os inputs que aparecem na janela do script:

| Input | Padrão | Descrição |
|-------|--------|-----------|
| `InpSymbol` | `WDO$` | Símbolo (WDO$, WIN$, etc.) |
| `InpFrom` | `2025.01.01` | Início do período |
| `InpTo` | `2025.12.31` | Fim do período |
| `InpUseTrades` | `true` | `true` = usa apenas preço de negociação (`last`); `false` = usa bid/ask médio |
| `InpFilename` | `radar_ticks_1s.csv` | Nome do arquivo de saída |

O CSV é salvo em `<pasta de dados MT5>/MQL5/Files/`.

### Importante — histórico de ticks

- Corretoras costumam **guardar apenas ~2 semanas de ticks** no servidor.
  Para períodos maiores, deixe o terminal aberto acumulando histórico
  (o MT5 baixa ticks automaticamente enquanto conectado).
- Dados de 1s para 1 dia de WDO$ ≈ 15–20 mil barras (só horário de
  negociação). Um mês pode gerar centenas de MB em CSV — ok para o
  DuckDB, mas evite exportar anos de uma vez.
- Se `InpUseTrades=true` e a barra ficar vazia em horário sem negociação,
  ela simplesmente não é gravada (não existem gaps falsos).

### Ingestão no Radar

Depois de exportar, copie o CSV do `MQL5/Files/` para `dump/` (na máquina
onde roda o Radar) e suba:

```bash
curl -F "file=@dump/radar_ticks_1s.csv" \
     -F "symbol=WDO$" -F "timeframe=S1" \
     http://localhost:8000/v1/ingest
```

O parser do Radar já entende esse cabeçalho (`date;time;open;high;low;close;volume`).
