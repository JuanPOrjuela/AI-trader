# AI-trader

# Sistema Autonomo de Trading Cuantitativo con IA

Bot de trading modular para **Criptomonedas** (Binance / Bybit / Coinbase via CCXT) y
**Acciones y ETFs** (Alpaca / Interactive Brokers), disenado para operar **24/7 sin
supervision humana** con gestion de riesgo de nivel institucional.

Arranca en **Paper Trading** y funciona de inmediato: sin claves de API, sin
configuracion previa y usando **precios publicos reales** cuando hay conexion.

```bash
pip install -r requirements.txt
python main.py selftest      # 20 comprobaciones de diagnostico
python main.py paper         # arranca la operativa simulada 24/7
```

---

## 1. Que hace, exactamente

En cada ciclo, para cada activo vigilado:

```
  Feed multi-timeframe          1m / 5m / 15m / 1h / 4h / 1d sincronizados
        v
  Ingenieria de caracteristicas 30+ indicadores, estructura de mercado (SMC),
        v                       perfil de volumen, Hurst, volatilidad realizada
  Clasificador de regimen       HMM gaussiano propio (Baum-Welch)
        v                       Bull / Bear / Rango / Choque de volatilidad
  Confluencia multi-factor      >= 3 FAMILIAS independientes de acuerdo
        v
  Stop y objetivos              SL por estructura o 1.5*ATR; TP1 1.5R / TP2 3R / runner
        v
  Valor esperado                EV = P(win)*TP - P(loss)*SL, con P(win) bayesiana
        v
  Comite de riesgo              sizing, correlacion, riesgo agregado, drawdown
        v
  Control de ejecucion          spread, slippage por profundidad, impacto
        v
  Orden + gestion de posicion   parciales, break-even, trailing dinamico
```

Una operacion solo se abre si **todas** las etapas la aprueban.

---

## 2. Arranque rapido

### Requisitos
Python 3.10 o superior. El nucleo obligatorio son 5 paquetes; todo lo demas es opcional.

### Instalacion

```bash
cd ai_trader_system
pip install -r requirements.txt
cp .env.example .env          # opcional: solo si va a usar claves o notificaciones
```

### Comandos

```bash
# Diagnostico completo del sistema (recomendado antes de nada)
python main.py selftest

# Paper trading 24/7 (modo por defecto)
python main.py paper

# Paper acotado a 20 ciclos, solo cripto, con log detallado
python main.py paper --cycles 20 --no-stocks --log-level DEBUG

# Backtest con datos reales descargados del exchange
python main.py backtest --symbol BTC/USDT --timeframe 15m --bars 8000 --save-report

# Backtest con datos sinteticos (sin red)
python main.py backtest --symbol BTC/USDT --synthetic --bars 6000

# Estado persistido de la ultima sesion
python main.py status

# Operativa real: exige claves validas y confirmacion escrita
python main.py live --no-dry-run
```

---

## 3. Gestion de riesgo (motor innegociable)

Cada limite del mandato esta implementado y cubierto por tests.

| Regla | Valor por defecto | Donde |
|---|---|---|
| Riesgo por operacion | 1% (techo 2%) | `risk/risk_manager.py` |
| Formula de sizing | `(Capital * Riesgo%) / abs(Entrada - Stop)` | `position_size()` |
| Kelly fraccional | 0.25 x Kelly, **solo puede reducir** | `_size_and_finalize()` |
| Stop Loss obligatorio | estructura o 1.5 x ATR | `risk/stop_target_calc.py` |
| TP1 | 1.5R, cierra 45%, mueve stop a break-even | `build_targets()` |
| TP2 | 3R, cierra 30% | `build_targets()` |
| Runner | 25%, trailing tipo Chandelier | `update_trailing_stop()` |
| Riesgo abierto agregado | 6% maximo | `RiskManager` |
| Drawdown diario | -3% congela 24h | `risk/circuit_breaker.py` |
| Drawdown total | -15% detiene el sistema | `CircuitBreaker._halt()` |
| Correlacion | veta pares con corr > 0.80 | `correlation_with_open()` |
| R:R minimo | 1:2 | `PredictiveEngine` |

**Nunca se abre una posicion sin stop loss.** Si la entrada se ejecuta pero la
proteccion no puede colocarse, la posicion se cierra de inmediato.

### Decisiones de diseno defensivas

- **El stop nunca retrocede.** `update_trailing_stop()` solo acepta movimientos
  que reduzcan el riesgo. Permitir lo contrario convierte un riesgo definido en
  riesgo ilimitado.
- **La congelacion sobrevive al reinicio.** El estado del circuit breaker se
  persiste en SQLite: reiniciar el proceso no puede ser la via para saltarse un
  limite. Hay un test que lo verifica.
- **Kelly solo reduce.** Si `P(win)` esta sobreestimada (y casi siempre lo esta),
  dejar que Kelly amplie el tamano es la via rapida a la ruina.
- **Sin datos de correlacion se asume lo peor.** Dos activos del mismo grupo se
  tratan como correlacionados aunque falte historico. Asumir independencia seria
  el error caro.
- **Al descongelar se rebasa el dia contable.** Sin esto, una congelacion que
  expira dentro del mismo dia se vuelve a disparar al instante con la misma
  perdida, y la operativa no se reanudaria jamas.

---

## 4. Los algoritmos cuantitativos

### 4.1 Clasificador de regimenes: HMM gaussiano propio

`research/market_regime.py` implementa un **Modelo Oculto de Markov gaussiano de
covarianza diagonal** entrenado con **Baum-Welch** (forward-backward escalado,
paso M cerrado y Viterbi), escrito directamente sobre numpy.

No se usa `hmmlearn` a proposito: el modelo es el corazon del sistema y debe ser
auditable, reproducible y no romperse por la version de una libreria externa.

Cuatro caracteristicas ortogonales alimentan el modelo:

| # | Caracteristica | Que captura |
|---|---|---|
| 0 | deriva suavizada / volatilidad | direccion con buena relacion senal/ruido |
| 1 | volatilidad realizada / mediana | estado de riesgo |
| 2 | pendiente de la EMA / volatilidad | fuerza de la tendencia |
| 3 | magnitud del retorno suavizada | choque frente a calma |

Los estados latentes **no tienen etiqueta a priori**: se nombran despues, segun
las estadisticas de cada estado, con umbrales **relativos** al conjunto ajustado.
Un umbral absoluto etiquetaria mal: lo que en BTC es tendencia fuerte, en SPY es
ruido. Ademas, un choque de volatilidad se reparte entre varios estados latentes
(los tramos al alza y a la baja del propio caos), asi que se marcan **todos** los
estados con volatilidad extrema, no solo el mayor.

### 4.2 Confluencia multi-factor

Cinco **familias independientes** votan por separado:

| Familia | Peso | Ejemplos de factores |
|---|---|---|
| `TECHNICAL` | 1.00 | alineacion de EMAs, BOS/CHoCH, ADX, MACD, barridas de liquidez |
| `FLOW` | 0.90 | Order Book Imbalance ponderado, microprecio, OBV, volumen |
| `REGIME` | 0.85 | estado del HMM, persistencia esperada |
| `VOLATILITY` | 0.60 | compresion (squeeze), extremos de banda, Hurst |
| `SENTIMENT` | 0.45 | noticias con lexico financiero, riesgo macro |

La confluencia se cuenta **por familias, no por indicadores**. Contar tres
indicadores tecnicos correlacionados como "tres factores" es el error clasico que
produce sistemas que parecen robustos y no lo son.

### 4.3 Microestructura

`data/orderbook_watcher.py` calcula OBI simple y ponderado por distancia al medio
(la liquidez lejana pesa menos: casi nunca se ejecuta y es la mas facil de
falsear), microprecio, muros de liquidez, un proxy de **VPIN** para detectar flujo
toxico, y **slippage esperado recorriendo el libro nivel a nivel** — la unica
forma de saber lo que costaria realmente una orden de mercado.

### 4.4 Valor esperado bayesiano

```
EV_R = P(win) x RR_efectivo - (1 - P(win)) x 1.0
```

`P(win)` mezcla dos fuentes deliberadamente distintas:

1. **Posterior Beta** sobre trades reales del mismo tipo (estrategia + regimen).
   Prior `Beta(6, 9)` = 40% de acierto, deliberadamente pesimista.
2. **Modelo logistico** sobre las caracteristicas de la senal, entrenado online.
   Calibrado para que una senal excelente de ~0.52, no ~0.86.

El peso se desplaza hacia la evidencia empirica conforme se acumulan trades
(`shrinkage = n / (n + 12)`). El sistema arranca prudente y converge a lo que el
mercado demuestra.

**`RR_efectivo` pondera el plan de salidas escalonadas**, no usa el objetivo final:
cerrar 45% en 1.5R y 30% en 3R rinde 2.325R, no 3R. Usar el objetivo final
sobreestimaria el EV de forma sistematica; asi se construyen los backtests
optimistas que luego no se replican.

---

## 5. Estructura del proyecto

```
ai_trader_system/
├── config/
│   ├── settings.py              carga de .env, validacion y switches globales
│   ├── trading_universe.yaml    activos vigilados, ticks, lotes, horarios
│   └── strategy_params.yaml     parametros por activo (herencia en 3 niveles)
├── core/
│   ├── types.py                 dataclasses de dominio compartidas
│   ├── engine.py                loop principal 24/7 y orquestacion
│   ├── scheduler.py             tareas periodicas aisladas con auto-desactivacion
│   └── portfolio_state.py       balance, posiciones, exposicion y R-multiplos
├── connectors/
│   ├── base.py                  interfaz abstracta, rate limiter, reintentos
│   ├── paper_broker.py          motor de emparejamiento y mercado sintetico
│   ├── crypto_exchange.py       CCXT + REST publico real + paper
│   └── stock_broker.py          Alpaca/IBKR + calendario de mercado con DST
├── data/
│   ├── market_feed.py           ingesta y agregacion multi-timeframe
│   ├── orderbook_watcher.py     microestructura, OBI, VPIN, slippage
│   ├── sentiment_feed.py        noticias, lexico financiero, riesgo macro
│   └── database.py              SQLite (WAL) con auditoria completa
├── research/
│   ├── feature_engineering.py   indicadores, SMC, perfil de volumen, Hurst
│   ├── market_regime.py         HMM gaussiano propio (Baum-Welch + Viterbi)
│   ├── alpha_signals.py         familias de factores y motor de confluencia
│   └── predictive_engine.py     posterior Beta, modelo logistico y EV
├── risk/
│   ├── risk_manager.py          sizing, correlacion, limites agregados
│   ├── stop_target_calc.py      SL/TP dinamicos, break-even y trailing
│   └── circuit_breaker.py       kill-switches persistentes
├── execution/
│   ├── order_router.py          enrutado bracket/OCO, idempotencia, reintentos
│   ├── position_manager.py      parciales, break-even, trailing, salidas
│   └── slippage_guard.py        spread, slippage e impacto antes de ejecutar
├── backtesting/
│   ├── event_simulator.py       motor event-driven con costes y latencia
│   └── performance_report.py    Sharpe, Sortino, Calmar, drawdown, desgloses
├── telemetry/
│   ├── logger.py                logger estructurado con rotacion
│   └── notifier.py              Telegram / Discord, best-effort
├── tests/                       108 pruebas
├── main.py                      CLI: paper | live | backtest | selftest | status
├── requirements.txt
└── .env.example
```

---

## 6. Modos de operacion y degradacion

El sistema **nunca se cae por una dependencia ausente**. Degrada hacia abajo:

| Situacion | Comportamiento |
|---|---|
| Con claves + CCXT + `MODE=LIVE --no-dry-run` | operativa real |
| Sin claves, con red | **paper con precios publicos REALES** de Binance |
| Sin red | paper con mercado sintetico realista |
| Sin `loguru` | logger de la stdlib con rotacion equivalente |
| Sin `hmmlearn` | HMM propio (siempre; no es una dependencia) |
| Sin feeds RSS | sentimiento neutro, se opera con las otras 4 familias |

El **mercado sintetico** no es un paseo aleatorio: reproduce cambios de regimen
persistentes, agrupamiento de volatilidad y saltos por noticias. Los tests
verifican que exhibe **curtosis > 3** (colas gruesas) y autocorrelacion positiva
de la magnitud de los retornos. Sin esas propiedades, el clasificador de
regimenes nunca veria en paper las condiciones que encontrara en real.

---

## 7. Tests

```bash
python -m pytest              # 108 pruebas
python -m pytest -m "not slow"
python -m pytest tests/test_risk_manager.py -v
```

| Fichero | Pruebas | Cubre |
|---|---|---|
| `test_risk_manager.py` | 36 | sizing, SL/TP, trailing, drawdown, kill-switches, contabilidad |
| `test_signals.py` | 45 | indicadores, microestructura, regimenes, confluencia, EV, sentimiento |
| `test_execution.py` | 17 | slippage, paper broker, mercado sintetico, agregacion |
| `test_integration.py` | 10 | ciclo de vida de posicion, backtest, persistencia, motor |

Incluye pruebas explicitas de **ausencia de sesgo de anticipacion**: se altera el
futuro de la serie y se comprueba que ningun indicador ni caracteristica del HMM
calculada sobre el pasado cambia.

---

## 8. Validacion honesta sobre datos reales

Backtest sobre **33 dias reales de BTC/USDT** descargados de Binance, comparando
configuraciones (mismo codigo, mismos limites de riesgo):

| Configuracion | Trades | Acierto | Profit Factor | Expectancy | Retorno | Comisiones |
|---|---|---|---|---|---|---|
| Gatillo 5m (por defecto original) | 110 | 42.7% | 0.88 | -0.414R | **-3.10%** | 1.026 |
| Gatillo 5m + 4 familias | 105 | 41.0% | 0.69 | -0.393R | -8.59% | 952 |
| **Gatillo 15m (por defecto actual)** | 28 | 50.0% | 1.62 | +0.240R | **+5.76%** | 263 |
| Gatillo 15m + 4 familias | 18 | 72.2% | 3.77 | +0.847R | +12.34% | 174 |
| Gatillo 1h | 5 | 20.0% | 0.66 | -0.283R | -1.47% | 31 |
| *Comprar y mantener* | - | - | - | - | *+17.38%* | 0 |

**Lo que esto significa, sin adornos:**

- A 5 minutos el sistema **paga 1.026 USDT de comisiones sobre 10.000 de capital**
  (mas del 10%) y convierte una senal decente en expectancy negativa. La
  sobreoperativa, no la senal, era el problema. Por eso el gatillo por defecto de
  cripto se cambio a **15m**, y el motivo esta documentado en
  `config/strategy_params.yaml`.
- **Ninguna configuracion bate a comprar y mantener** en esta muestra concreta,
  que fue un mercado alcista del +17%. Un sistema que opera en ambos sentidos y
  esta fuera del mercado la mayor parte del tiempo no deberia batir a un
  buy & hold en un tramo asi; lo relevante seria su comportamiento en mercados
  bajistas y laterales, que esta muestra no contiene.
- **33 dias y 18-28 operaciones no permiten concluir nada.** El propio informe
  emite el veredicto `SIN CONCLUSION` por debajo de 30 operaciones o 30
  observaciones diarias, y marca CAGR y Calmar como `n/d` por debajo de 60 dias
  de muestra. Anualizar dos semanas produce cifras de cuatro digitos que no
  significan nada.

**Antes de arriesgar capital real**: ejecute backtests de varios anos, en varios
activos, incluyendo 2022 (bajista) y periodos laterales, y despues opere en paper
al menos un mes. Los numeros de arriba validan que **la maquinaria funciona**, no
que la estrategia tenga ventaja.

### Metodologia del informe

- **Sharpe y Sortino se calculan sobre retornos DIARIOS**, no barra a barra. Con
  velas de 5 minutos la mayoria de barras tiene retorno cero porque no hay
  posicion abierta; esos ceros hunden la desviacion tipica y disparan
  artificialmente el ratio. Muestrear a diario es lo que hace cualquier fondo y lo
  unico comparable con la industria.
- **El PnL de cada operacion es NETO de comisiones.** Reportar el bruto hace que
  la suma de los trades no cuadre con la variacion real del equity, que es
  justamente el error que hace parecer rentable a un sistema que se lo come todo
  en comisiones.
- **Prioridad intrabarra: el stop antes que el objetivo.** Si una vela toca ambos,
  se asume el stop. Es el supuesto conservador y el unico honesto sin datos tick.
- **Latencia modelada:** la senal nace al cierre de una barra y se ejecuta en la
  apertura de la SIGUIENTE. Nunca al precio que genero la senal.

---

## 9. Operativa 24/7

- **Reconexion automatica** con backoff exponencial **y jitter**. Sin jitter,
  todos los simbolos que fallan a la vez reintentan en el mismo instante y
  provocan una estampida que el exchange vuelve a rechazar.
- **Rate limiting proactivo** por ventana deslizante: se limita antes de recibir
  un 429, porque cuando el exchange lo devuelve la ventana de ejecucion ya se
  perdio.
- **Aislamiento de fallos**: un simbolo que falla no detiene el ciclo de los
  demas; una tarea de mantenimiento rota se desactiva sola tras 5 fallos y avisa.
- **Las posiciones abiertas se gestionan siempre**, incluso con el circuit breaker
  disparado. Dejar de vigilar un stop porque el sistema esta congelado seria el
  peor momento posible para dejar de mirar.
- **Apagado ordenado** ante SIGINT/SIGTERM, con persistencia de estado y modelo.
- **Horarios de mercado con `zoneinfo`**, no con offsets fijos: el horario de
  verano de Nueva York cambia el equivalente UTC de la apertura dos veces al ano,
  y un offset fijo operaria una hora fuera de sesion durante meses.

### Tareas periodicas

| Tarea | Intervalo |
|---|---|
| Persistencia de equity | 60 s |
| Refresco de sentimiento | 15 min |
| Recalculo de correlaciones | 15 min |
| Salud de conectores y reconexion | 2 min |
| Latido por Telegram/Discord | 60 min |
| Guardado del modelo predictivo | 30 min |
| Mantenimiento y poda de la base de datos | 24 h |

---

## 10. Configuracion

`config/strategy_params.yaml` aplica herencia en tres niveles:

```
default  <-  (crypto | stock)  <-  overrides[SIMBOLO]
```

Asi, `SOL/USDT` hereda todo lo generico, luego lo especifico de cripto y por
ultimo su propio `atr_stop_mult: 1.9`, mas holgado por su mayor volatilidad.

`config/trading_universe.yaml` define activos, `tick`/`lot` para el redondeo de
ordenes, grupos de correlacion a priori y el calendario bursatil.

---

## 11. Paso a operativa real

1. Ejecute `python main.py selftest` y confirme 20/20.
2. Opere en **paper un mes** como minimo y revise `python main.py status`.
3. Haga backtests plurianuales incluyendo mercados bajistas y laterales.
4. Instale el conector: `pip install ccxt` y/o `pip install alpaca-py`.
5. Rellene las claves en `.env`. **Empiece con las claves de paper del broker.**
6. Restrinja los permisos de la API: **habilite trading, NUNCA retiradas**, y
   limite por IP.
7. Baje el riesgo: `RISK_PER_TRADE_PCT=0.25` durante las primeras semanas.
8. Arranque con `python main.py live --no-dry-run` y escriba `CONFIRMO`.

> **Aviso.** Este software se entrega con fines educativos y de investigacion. El
> trading algoritmico puede producir perdidas superiores al capital depositado.
> Nada en este repositorio es asesoramiento financiero. Los resultados de los
> backtests **no** predicen resultados futuros. Opere unicamente con capital que
> pueda permitirse perder por completo.

---

## 12. Limitaciones conocidas

- **Sin WebSockets reales todavia.** El feed usa REST con polling y reconexion.
  La estructura esta preparada en `market_feed.py`, pero la ingesta por streaming
  no esta implementada; a intervalos de 5 s no es limitante, para HFT lo seria.
- **El libro de acciones es sintetico.** Alpaca no expone profundidad completa en
  su plan basico, asi que las metricas de microestructura de renta variable son
  aproximadas.
- **Sin cointegracion ni pares.** El filtro de correlacion evita redundancia,
  pero no hay estrategia de arbitraje estadistico.
- **El walk-forward esta especificado pero no automatizado.** La seccion
  `walk_forward` de `strategy_params.yaml` define ventanas y objetivo; la
  reoptimizacion periodica no se ejecuta sola todavia.
- **El sentimiento usa lexico, no un transformer.** Es determinista, auditable y
  sin latencia de GPU, pero no entiende ironia ni contexto largo.
- **IBKR esta contemplado en la configuracion** pero el conector implementado a
  fondo es el de Alpaca.
