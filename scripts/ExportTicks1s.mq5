//+------------------------------------------------------------------+
//|                                                  ExportTicks1s.mq5 |
//|  Baixa ticks do MetaTrader 5 e exporta para CSV                  |
//|  (grão 1s OHLCV, compatível com o Radar)                         |
//+------------------------------------------------------------------+
#property copyright "Radar"
#property version   "1.00"
#property script_show_inputs

//--- inputs
input string InpSymbol    = "WDO$";     // Símbolo (ex.: WDO$, WIN$)
input datetime InpFrom    = D'2025.01.01';  // Data inicial
input datetime InpTo      = D'2025.12.31';  // Data final
input bool   InpUseTrades = true;       // true = só trades (last), false = bid/ask
input string InpFilename  = "radar_ticks_1s.csv"; // Nome do arquivo de saída

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
   if(!SymbolSelect(InpSymbol, true))
   {
      Print("Falha ao selecionar símbolo: ", InpSymbol);
      return;
   }

   // Força o MT5 a sincronizar o histórico de ticks do servidor.
   MqlTick dummy[];
   int n = CopyTicks(InpSymbol, dummy, COPY_TICKS_ALL, 0, 1);
   if(n <= 0)
   {
      Print("Sem histórico de ticks disponível para ", InpSymbol);
      return;
   }

   ulong from_msc = (ulong)InpFrom * 1000;
   ulong to_msc   = (ulong)InpTo * 1000;

   // Verifica se o terminal tem ticks no intervalo.
   MqlTick probe[];
   int probe_n = CopyTicksRange(InpSymbol, probe, COPY_TICKS_ALL, from_msc, to_msc);
   if(probe_n <= 0)
   {
      Print("Sem ticks no intervalo ", TimeToString(InpFrom), " -> ", TimeToString(InpTo));
      Print("Corretoras costumam guardar só ~2 semanas de ticks. Aumente o prazo");
      Print("ou verifique se o terminal ficou aberto acumulando histórico.");
      return;
   }

   int handle = FileOpen(InpFilename, FILE_WRITE | FILE_CSV | FILE_ANSI, ";");
   if(handle == INVALID_HANDLE)
   {
      Print("Falha ao criar arquivo: ", InpFilename, " (erro ", GetLastError(), ")");
      return;
   }

   // Cabeçalho no formato que o Radar espera: date;time;open;high;low;close;volume
   FileWrite(handle, "date", "time", "open", "high", "low", "close", "volume");

   ulong pos_msc = from_msc;
   int   total_bars = 0;
   while(pos_msc <= to_msc)
   {
      MqlTick ticks[];
      // Baixa em blocos de 1 dia para não estourar memória.
      ulong block_end = pos_msc + 86400000 - 1;
      if(block_end > to_msc) block_end = to_msc;

      int got = CopyTicksRange(InpSymbol, ticks, COPY_TICKS_ALL, pos_msc, block_end);
      if(got > 0)
         total_bars += ProcessTicks(handle, ticks, got);

      if(got < 0)
      {
         Print("Erro CopyTicksRange: ", GetLastError());
         break;
      }
      pos_msc = block_end + 1;
   }

   FileClose(handle);

   Print("Concluído: ", total_bars, " barras de 1s exportadas para ", InpFilename);
   Print("Arquivo em: <pasta de dados MT5>/MQL5/Files/", InpFilename);
}

//+------------------------------------------------------------------+
//| Agrega ticks em barras OHLCV de 1 segundo e grava no CSV         |
//+------------------------------------------------------------------+
int ProcessTicks(int handle, const MqlTick &ticks[], int count)
{
   int bars = 0;
   long last_sec = 0;
   double o = 0.0, h = 0.0, l = 0.0, c = 0.0;
   long volume = 0;
   bool in_bar = false;

   for(int i = 0; i < count; i++)
   {
      const MqlTick &t = ticks[i];

      double price;
      if(InpUseTrades)
      {
         // Só ticks de trade (last != 0). WDO$/WIN$ negociam via last.
         if(t.last == 0.0) continue;
         price = t.last;
      }
      else
      {
         price = (t.bid + t.ask) / 2.0;
         if(price <= 0.0) continue;
      }

      long sec = (long)(t.time_msc / 1000);

      if(!in_bar || sec != last_sec)
      {
         // Fecha barra anterior
         if(in_bar)
            WriteBar(handle, last_sec, o, h, l, c, volume);

         // Abre nova barra
         last_sec = sec;
         o = h = l = c = price;
         volume = 0;
         in_bar = true;
         bars++;
      }
      else
      {
         if(price > h) h = price;
         if(price < l) l = price;
         c = price;
      }

      volume += (long)t.volume_msc;
   }

   if(in_bar)
      WriteBar(handle, last_sec, o, h, l, c, volume);

   return bars;
}

//+------------------------------------------------------------------+
//| Grava uma barra de 1s como linha do CSV                          |
//+------------------------------------------------------------------+
void WriteBar(int handle, long sec, double o, double h, double l, double c, long volume)
{
   datetime dt = (datetime)(sec);
   FileWrite(
      handle,
      TimeToString(dt, TIME_DATE),
      TimeToString(dt, TIME_MINUTES | TIME_SECONDS),
      DoubleToString(o, _Digits),
      DoubleToString(h, _Digits),
      DoubleToString(l, _Digits),
      DoubleToString(c, _Digits),
      (long)volume
   );
}
