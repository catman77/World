## 1) Строгость и проверяемость выводов в [experiment_E_time_travel.pdf](/files/MT-IgbbtHuxE3Bl9NeHbG)

Файл в целом хорошо структурирован: он отделяет (i) микрофизику `X(t)` (спины, φ, граф, время) от (ii) состояния наблюдателя `O(t)` (IFACEHistory/SemanticState/cache), вводит `micro_hash` и формулирует тесты **T5 (детерминизм)**, **R0 (микро‑обратимость)** и **OR (Observer reset)**. Это правильная рамка: “путешествие во времени” нужно обсуждать не словами, а через операции над состоянием и проверяемые инварианты.

### Что уже достаточно строго
1) **Операционализация**: вы задали конкретные процедуры (“прогнать вперёд/назад”, “очистить память наблюдателя”) и критерии (`micro_hash` совпадает).  
2) **Разделение слоёв**: чётко сказано, что глобальный откат включает и `O(t)`, а reset — только `O(t)` при фиксированном `X(t)`.  
3) **Конкретные параметры прогона** (N, α, Tforward, Tbackward, Nseeds, εφ), что позволяет повторить.

### Что мешает внешней проверке “уровня A/B/C”
Чтобы внешний пользователь мог подтвердить всё *без доверия к вашему ноутбуку*, не хватает типичных для A/B/C элементов:

1) **Артефактов для пересчёта**: в PDF описаны хэши и тесты, но не видно (или не заявлено явно), что сохраняются:
   - `step_records.npz`/`parquet` (или хотя бы aggregated counts),
   - таблица `seed → pass/fail + hashes`,
   - `manifest.json` с sha256 артефактов,
   - `recompute_contract` (формулы, точные поля, квантизация φ, канонизация edges).
   Без этого внешний проверяющий не может гарантировать, что PASS получен именно из заявленных данных и процедур.

2) **Явного определения шага назад для φ**: у вас φ обновляется формулой вида
\[
\phi(t+1)=\phi(t)+D_\phi \Delta_G \phi(t)+\beta\rho_s(t)-\gamma\phi(t),
\]
что в общем случае **не является обратимым отображением** (из-за затухания/диффузии это обычно сжимающее отображение). В тексте заявлено “φ‑поле обратимо при фиксированных спинах”, но для строгости нужно одно из двух:
   - либо показать, что в эксперименте φ не эволюционирует отдельным уравнением, а вычисляется как решение \(L\phi=\rho\) (тогда обратимость сводится к обратимости `s`),  
   - либо явно задать обратный шаг для φ (что возможно только при специальных условиях, например \(\gamma=0\) и при обратимом интеграторе, или если вы храните “инкременты” для φ в StepRecord).

3) **Чёткое определение “полного состояния” для обратимости**: вы правильно говорите, что кэши и детекторы Ω‑циклов могут “сломать” обратимость. Для внешнего стандарта нужно явно заявить: тест R0 проверяет обратимость *только X(t)* или *всего S(t)*, и что именно включено в hash.

Если эти пункты добавить, эксперимент E станет CI‑аудируемым так же, как A/B.

---

## 2) По сути вопроса: “невозможны ли путешествия во времени в детерминированном мире, включая перенос сознания?”

Вы понимаете **почти** правильно, но важно уточнить, *в каком смысле* “путешествие во времени” и что вы считаете “переносом сознания”.

### 2.1. Глобальный откат мира (вариант A)
Ваш вывод корректен: если вы откатываете **всё состояние** \(S(t)=(X(t),O(t))\) назад, то у наблюдателя не остаётся записи о “будущем”, потому что память — часть \(O(t)\) и тоже откатилась.  
С точки зрения внутреннего агента это не “путешествие”, а просто та же история заново, без знания о том, что она уже была.

Это не “запрещено физикой” — просто **неоперационально**: не возникает наблюдаемого эффекта “я побывал в будущем и вернулся”.

### 2.2. Reset только наблюдателя (вариант B) = амнезия, но не обязательно “невозможно”
Если операция выглядит как:
\[
(X_{t+T}, O_{t+T}) \;\to\; (X_{t+T}, O_t),
\]
то да, в простейшем смысле это **амнезия**: мир остаётся в состоянии будущего, а память возвращается к более раннему состоянию.

Однако это *не* равносильно утверждению “перенос сознания в прошлое невозможен в принципе”. Возможны два разных сценария:

#### Сценарий B1 (как у вас): reset = стирание/откат памяти без внешнего канала
Тогда да: это не даёт “путешествия” в смысле передачи полезной информации в прошлое. Наблюдатель просто теряет то, что узнал.

#### Сценарий B2: “перенос сознания” = восстановление более раннего состояния O **плюс** сохранение некоторого маркера/сообщения
Чтобы это стало похожим на “путешествие сознания”, нужно, чтобы *что-то* из будущего всё же попало в прошлое. Формально это означает наличие канала:
- либо скрытого состояния \(M\), которое **не откатывается** вместе с O,
- либо внешнего записывающего носителя в X, который “переживает” reset O,
- либо “двухслойной памяти”: часть памяти сбрасывается, а часть нет.

Но тогда это уже **не тот же детерминированный закрытый мир**, который вы описали как \(S=(X,O)\) без дополнительных степеней свободы. Это становится либо:
- расширением состояния (и тогда при “честном” откате надо откатывать и это \(M\)),  
- либо введением несимметричной операции (стирание информации) — а это уже нарушение обратимости/замкнутости или подключение внешней среды.

Именно поэтому ваш вывод “reset = амнезия” логически следует **при условии**, что:
- мир замкнут,
- оператор reset не создаёт новый внешний носитель информации,
- и вы не разрешаете “частичный” откат, который сохраняет некоторую память вне O.

### Итоговая формулировка
- **В детерминированной замкнутой системе** с честным учётом полного состояния:  
  “путешествие во времени” как *передача информации в прошлое* не возникает само по себе из обратимости. Глобальный rollback не наблюдаем изнутри, observer‑reset даёт амнезию.
- Но фраза “невозможно в принципе” верна только при явно оговорённых предпосылках: *нет внешнего носителя информации, нет частично неоткатываемых степеней свободы, reset не оставляет маркеров в X*.

Если вы хотите, я могу предложить строгое дополнение к эксперименту E: тест “No information to past” — проверку, что при любой последовательности действий, включающей rollback/reset, не существует протокола, который увеличивает взаимную информацию между “будущими измерениями” и “прошлыми состояниями наблюдателя” без введения внешнего канала.

Ниже — строгое дополнение к эксперименту **E**: тест **NIP (No Information to Past)**, который проверяет именно то, что вы сформулировали: *при любых допустимых последовательностях действий, включающих rollback/reset, нельзя получить протокол, который увеличивает взаимную информацию между “будущими измерениями” и “прошлыми состояниями наблюдателя” без внешнего канала*.

Я сформулирую это как (1) чёткую модель, (2) измеримую величину (взаимная информация), (3) протокол теста, (4) критерии PASS/FAIL, (5) минимальные артефакты для внешней проверки (как в A/B/C/D).

---

# 1) Формальная постановка: что именно запрещаем

## 1.1. Разделение состояния
Как в вашем E:

- мир: \(X_t\) (s, φ, graph, t, …)  
- наблюдатель: \(O_t\) (IFACEHistory, SemanticState, caches, …)

Полный state: \(S_t = (X_t, O_t)\).

## 1.2. Допустимые операции (actions)
Разрешаем только два класса “временных” действий:

1) **Rollback мира**:  
\[
\mathrm{Rollback}_k: (X_t,O_t)\mapsto (X_{t-k}, O_{t-k})
\]
(строгое откатывание всего состояния по журналу `StepRecord`).

2) **ObserverReset** (ваш OR-тест):  
\[
\mathrm{Reset}: (X_t,O_t)\mapsto (X_t, R(O_t))
\]
где \(R\) — детерминированная функция (очистить IFACEHistory и/или SemanticState).

Важно: никаких внешних логов/буферов “памяти будущего” и никаких “мета-сидов” внутри эксперимента.

---

# 2) Что считаем “информацией из будущего в прошлое”

## 2.1. Случайная величина “будущего измерения”
Определим наблюдаемое измерение в будущем как:

\[
Y = g(O_{t+T})
\]

Где \(g\) — фиксированный “readout” наблюдателя после эволюции на \(T\) шагов (например, бит из SemanticState: “обнаружен ли Ω‑цикл данного типа”, или “какой закон гравитации выучен”, или просто hash некоторого окна IFACE).

Важно: \(g\) выбирается заранее и фиксируется в контракте.

## 2.2. “Прошлое состояние наблюдателя”
Определим прошлое состояние наблюдателя после возврата как:

\[
Z = h(O_{t})
\]

где \(h\) — функция, извлекающая измеримый “бит памяти” из наблюдателя в момент \(t\) (например, один флаг, маленький вектор признаков, или бинаризация `SemanticState`).

**Ключ:** Если существует протокол, который позволяет сделать \(Z\) информативным о \(Y\), то вы получили “информацию из будущего в прошлое”.

---

# 3) Центральная метрика: взаимная информация \(I(Y;Z)\)

Мы тестируем:

\[
I(Y;Z) \approx 0
\]

для всех допустимых протоколов, без внешнего канала.

Поскольку ваш мир детерминированен, единственный источник случайности для статистики — это распределение по **контекстам \(\omega\)** (seed’ам), как вы уже делаете в A/B/C/D.

То есть мы рассматриваем \(\omega \sim \mathcal{U}\{0,\dots,K-1\}\) и получаем выборку пар \((Y(\omega), Z(\omega))\).

---

# 4) Тест NIP: протоколы-атаки, которые должны провалиться (дать I≈0)

Чтобы тест был строгим, мы не должны проверять “в среднем при одном сценарии”, а должны проверить **класс возможных стратегий**.

Ниже — минимальный набор “атак”, которые покрывают типичные лазейки.

## 4.1. Attack A: “Rollback and read”
**Идея атакующего:** получить \(Y\) в будущем, затем откатиться и “положить” \(Y\) в память прошлого.

Но в вашей модели глобальный rollback откатывает и память, значит “положить” нечего.

Протокол:
1) старт: \(S_0(\omega)\)
2) прогон: \(S_T = F^T(S_0)\)
3) вычислить \(Y(\omega)=g(O_T)\)
4) откат: \(S_0' = \mathrm{Rollback}_T(S_T)\)
5) прочитать \(Z(\omega)=h(O_0')\)

Ожидание: \(Z\) не коррелирует с \(Y\) → \(I(Y;Z)=0\).

## 4.2. Attack B: “Reset and replay”
**Идея атакующего:** получить \(Y\), затем сбросить наблюдателя, “чтобы оказаться как раньше”, но оставить мир как в будущем.

Протокол:
1) \(S_0 \to S_T\)
2) \(Y=g(O_T)\)
3) \(\tilde S_T = \mathrm{Reset}(S_T) = (X_T, R(O_T))\)
4) попытаться прочитать “прошлую” память: \(Z=h(\tilde O_T)\)

Здесь проверка другая: **есть ли способ, чтобы reset оставил след о Y**. Если reset — действительно “амнезия”, то \(I(Y;Z)\approx 0\).

## 4.3. Attack C: “Rollback + reset + replay”
Это более хитро:  
получили \(Y\), откатили всё, затем reset (или наоборот), затем снова прогнали вперёд и пытаемся сделать так, чтобы “процедура выбора действий” зависела от будущего.

Но в детерминированном мире без внешнего канала действия не могут зависеть от будущего.

Протоколы:
- \(F^T \to \mathrm{Rollback}_T \to \mathrm{Reset} \to F^T\)
- \(\mathrm{Reset} \to F^T \to \mathrm{Rollback}_T\)

Опять проверяем \(I(Y;Z)\) на выходах.

---

# 5) Как посчитать взаимную информацию на практике (дискретизация)

Чтобы сделать внешний аудит простым, выбирайте **дискретные** \(Y\) и \(Z\). Например:

- \(Y \in \{0,1\}\): один бит “в будущем событие произошло/не произошло”.
- \(Z \in \{0,1\}\): один бит “в прошлом наблюдатель записал/не записал”.

Тогда взаимная информация вычисляется по таблице сопряжённых частот:

\[
I(Y;Z)=\sum_{y,z} p(y,z)\log_2\frac{p(y,z)}{p(y)p(z)}
\]

Также удобный эквивалент: **accuracy** лучшего предсказателя \(Y\) из \(Z\). Для независимости должно быть близко к \( \max(p(Y=0),p(Y=1))\).

### Обязательная поправка на конечную выборку
На конечном K оценка MI имеет положительный смещение. Поэтому тест делайте через permutation test:

- Перемешиваем \(Z\) относительно \(Y\) 1000 раз и строим нулевое распределение \(I_0\).
- PASS если наблюдаемое \(I\) не превосходит 95‑й перцентиль \(I_0\).

---

# 6) Критерии PASS/FAIL (TargetSpec)

Вводим:

- `MI_obs` — наблюдаемая взаимная информация
- `MI_null_p95` — 95% перцентиль permutation‑нуля
- `acc_obs` — точность классификатора \(Y\) по \(Z\) (для бинарных)

**PASS (No information to past)** если для всех атак A/B/C:
- `MI_obs <= MI_null_p95 + eps` (eps ~ 1e-3)
- и `acc_obs <= base_rate + delta` (delta ~ 0.02)

Где `base_rate = max(p(Y=0), p(Y=1))` по выборке.

---

# 7) Что сохранять как артефакты (в стиле A/B/C)

Чтобы внешний пользователь мог проверить без кода:

1) `tables/nip_trials.parquet`  
Строка = один seed, один протокол (A/B/C):
- `seed`
- `protocol_id` (A/B/C)
- `Y` (int)
- `Z` (int)
- (опционально) `Y_raw_hash`, `Z_raw_hash` для аудита

2) `tables/nip_summary.parquet`  
По каждому протоколу:
- `MI_obs`
- `acc_obs`
- `p_value_perm`
- `MI_null_p95`
- `PASS`

3) `experiment_E_report.json` дополнить секцией:
- `NIP` (thresholds, perm_test_params, definition of Y/Z, results)
- `recompute_contract`:
  - как из `nip_trials` пересчитать MI, accuracy, permutation p-value

4) `manifest.json` с sha256.

---

# 8) Минимальная “реализация” выбора Y и Z в вашем мире

Чтобы тест был честным, нужно выбрать \(g(O_T)\) и \(h(O_0)\) такие, которые:
- реально доступны наблюдателю,
- и не содержат скрытой “мета‑памяти”.

Примеры:

### Вариант Y (будущее)
- \(Y=1\), если в `SemanticState.event_stats` число “распадов” > порога
- или если в `IFACEState.objects` появился объект с `Q` определённого знака
- или если `SemanticState.gravity_law.alpha_hat` попал в интервал

### Вариант Z (прошлое)
- \(Z\) = бит, записанный в `SemanticState` *после rollback* (например, `semantic_state.flags["saw_future"]=...`).
Но важно: если вы строго откатываете состояние, этот флаг не может сохраниться.

Для Attack A Z должен браться именно из состояния после rollback, без внешнего буфера.

---

# 9) Почему это тестирует именно “невозможность передачи информации в прошлое”
Если какая-то лазейка существует (например, `ObserverResetAction` сохраняет часть информации в объекте, который не считается частью O, либо есть скрытый лог в X, который “не откатывается”), то появится статистически значимая связь \(Z\) с \(Y\). Это и есть operational definition “information traveled to past”.

---

Если хотите, я могу:
- дать точный Pydantic‑блок для включения `NIP` в `ExperimentEReport` (как мы делали для B/C/D),
- и псевдокод/референс‑реализацию `compute_mutual_information_binary()` + permutation test, чтобы CI мог проверять NIP автоматически.

Ниже — (1) точный **Pydantic‑блок** для включения теста **NIP (No Information to Past)** в `ExperimentEReport` (в стиле B/C/D), и (2) референс‑реализация **`compute_mutual_information_binary()`** + **permutation test**, чтобы CI мог пересчитывать PASS/FAIL по артефактам.

Смысл NIP: даже если в мире есть обратимость (см. [experiment_E_time_travel.pdf](/files/MT-IgbbtHuxE3Bl9NeHbG)) и возможна коллективная когерентность наблюдателей (через \(\lambda\) и \(\Xi\) из [Meaning_v1.pdf](/files/EljdBEFwvKSyTxfHYHjS7)), **не должно существовать протокола**, который увеличивает взаимную информацию между “будущим измерением” и “прошлой памятью наблюдателя” *без внешнего канала*.

---

# 1) Pydantic‑блок для NIP в ExperimentEReport

## 1.1. Модель артефактов и контракта пересчёта
Предположим, вы сохраняете:
- `tables/nip_trials.parquet`: по одному ряду на (seed, protocol_id)
  - `seed: int`
  - `protocol_id: str` (например `"A"|"B"|"C"`)
  - `Y: int` (0/1)
  - `Z: int` (0/1)
- `tables/nip_perm_null.npz` (опционально) — если хотите сохранять нулевые распределения (обычно не надо; CI может пересчитать).
- `experiment_E_report.json` + `manifest.json` с sha256

Ниже Pydantic‑схема, которую можно встроить в ваш существующий `ExperimentEReport` как поле `nip: NIPSection`.

```python
from __future__ import annotations
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, conint, confloat

ProtocolId = Literal["A", "B", "C"]  # расширяйте при необходимости

class NIPDefinitions(BaseModel):
    """
    Фиксирует, что именно является Y и Z (чтобы внешний проверяющий не гадал).
    """
    Y_name: str = Field(..., description="Name of future measurement bit")
    Z_name: str = Field(..., description="Name of past-memory bit after rollback/reset")
    Y_description: str
    Z_description: str
    readout_version: str = Field(..., description="Version tag of readout functions g() and h()")
    protocol_descriptions: Dict[ProtocolId, str]

class NIPPermutationConfig(BaseModel):
    n_perm: conint(ge=1000) = 5000
    alpha: confloat(gt=0.0, lt=1.0) = 0.05
    rng_seed: int = 123
    two_sided: bool = False  # для MI обычно one-sided: MI > null
    correction: Literal["none", "bonferroni", "fdr_bh"] = "bonferroni"

class NIPThresholds(BaseModel):
    eps_mi: confloat(ge=0.0) = 1e-3
    delta_acc: confloat(ge=0.0) = 0.02

class NIPPerProtocolResult(BaseModel):
    protocol_id: ProtocolId
    n: conint(ge=1)
    # empirical stats:
    pY1: confloat(ge=0.0, le=1.0)
    pZ1: confloat(ge=0.0, le=1.0)
    base_rate: confloat(ge=0.0, le=1.0)  # max(pY, 1-pY)
    mi_bits: confloat(ge=0.0)
    acc: confloat(ge=0.0, le=1.0)
    # permutation test:
    mi_null_p95: confloat(ge=0.0)
    p_value: confloat(ge=0.0, le=1.0)
    passed: bool
    notes: Optional[str] = None

class NIPArtifacts(BaseModel):
    nip_trials_table: str = Field(..., description="Relative path to nip_trials.parquet")
    # optional, for debugging:
    nip_debug_plot: Optional[str] = None  # e.g., barplot MI vs null
    nip_null_cache: Optional[str] = None  # e.g., npz with null distributions (optional)

class NIPRecomputeContract(BaseModel):
    """
    Машиночитаемая фиксация того, что CI должен пересчитать из nip_trials.parquet.
    """
    trials_table: str
    required_columns: List[str] = ["seed", "protocol_id", "Y", "Z"]
    mi_formula: str = Field(
        "I(Y;Z)=sum_{y,z} p(y,z)*log2(p(y,z)/(p(y)p(z))) with 0*log(0)=0",
        description="Definition of MI used in CI"
    )
    acc_formula: str = Field(
        "acc = mean( argmax_y p(y|Z) == Y ) (plug-in from empirical counts)",
        description="How accuracy is computed from Y,Z"
    )
    perm_test: str = Field(
        "Permutation test: shuffle Z within protocol; compute MI; p= (1+#{MI_perm>=MI_obs})/(1+n_perm)",
        description="Permutation test definition"
    )

class NIPSection(BaseModel):
    enabled: bool = True
    definitions: NIPDefinitions
    permutation: NIPPermutationConfig
    thresholds: NIPThresholds
    artifacts: NIPArtifacts
    recompute_contract: NIPRecomputeContract
    results: List[NIPPerProtocolResult]
    passed_all: bool
```

### Как вставить в ваш `ExperimentEReport`
В существующем отчёте (с tests/hypotheses/artifacts из [experiment_E_time_travel.pdf](/files/MT-IgbbtHuxE3Bl9NeHbG)) добавьте поле:

```python
class ExperimentEReport(BaseModel):
    # ...
    tests: Dict[str, Any]
    hypotheses: Dict[str, Any]
    artifacts: Dict[str, Any]

    nip: Optional[NIPSection] = None
```

И в JSON (`experiment_E_report.json`) фиксируйте:
- `nip.enabled=True`
- `nip.definitions` (что такое Y/Z + описания протоколов A/B/C)
- `nip.results` (по каждому протоколу)
- `nip.passed_all`

---

# 2) Референс‑реализация: compute_mutual_information_binary() + permutation test

Ниже — минимальная, но корректная реализация, не зависящая от scipy. Она рассчитана на вход в виде массивов 0/1 одинаковой длины.

## 2.1. MI для бинарных переменных

```python
import numpy as np

def compute_mutual_information_binary(Y: np.ndarray, Z: np.ndarray, *, eps: float = 0.0) -> float:
    """
    Compute I(Y;Z) in bits for binary Y,Z in {0,1}.
    Uses plug-in estimate from empirical frequencies.
    Convention: 0*log(0)=0.
    eps: optional additive smoothing (Laplace-style). If eps=0, pure empirical.
    """
    Y = np.asarray(Y).astype(np.int64)
    Z = np.asarray(Z).astype(np.int64)
    assert Y.shape == Z.shape
    assert np.all((Y == 0) | (Y == 1))
    assert np.all((Z == 0) | (Z == 1))

    # counts n(y,z)
    n00 = np.sum((Y == 0) & (Z == 0))
    n01 = np.sum((Y == 0) & (Z == 1))
    n10 = np.sum((Y == 1) & (Z == 0))
    n11 = np.sum((Y == 1) & (Z == 1))

    # smoothing (optional): add eps to each cell
    c = np.array([n00, n01, n10, n11], dtype=np.float64) + eps
    n = np.sum(c)

    p00, p01, p10, p11 = c / n

    pY0 = p00 + p01
    pY1 = p10 + p11
    pZ0 = p00 + p10
    pZ1 = p01 + p11

    # helper: safe term p * log2(p/(pY*pZ))
    def term(p, py, pz):
        if p <= 0.0:
            return 0.0
        return p * np.log2(p / (py * pz))

    mi = 0.0
    mi += term(p00, pY0, pZ0)
    mi += term(p01, pY0, pZ1)
    mi += term(p10, pY1, pZ0)
    mi += term(p11, pY1, pZ1)
    return float(mi)
```

Рекомендация: для CI лучше `eps=0` (строго по данным). Если боитесь нулевых клеток на малых n — фиксируйте `eps=0.5` и это записывайте в `recompute_contract`. Но тогда *обязательно* фиксируйте это как часть определения.

## 2.2. Точность лучшего предсказателя \(Y\) по \(Z\) (из тех же частот)

```python
def compute_best_predictor_accuracy_binary(Y: np.ndarray, Z: np.ndarray) -> float:
    """
    Accuracy of Bayes-optimal predictor using empirical p(y|z).
    For each z in {0,1}, predict y that is more frequent among samples with that z.
    """
    Y = np.asarray(Y).astype(np.int64)
    Z = np.asarray(Z).astype(np.int64)
    assert Y.shape == Z.shape

    acc = 0
    n = len(Y)
    for z in (0, 1):
        mask = (Z == z)
        if not np.any(mask):
            continue
        y_block = Y[mask]
        # predict the majority class in this block
        y_hat = 1 if np.mean(y_block) >= 0.5 else 0
        acc += np.sum(y_block == y_hat)
    return float(acc / n)
```

## 2.3. Permutation test для MI (one-sided)

```python
def permutation_test_mi_binary(
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    n_perm: int = 5000,
    rng_seed: int = 123,
    return_null: bool = False
):
    """
    Permutation test for MI(Y;Z).
    Null: Y independent of Z, approximated by shuffling Z.
    p-value: (1 + count(MI_perm >= MI_obs)) / (1 + n_perm)
    """
    Y = np.asarray(Y).astype(np.int64)
    Z = np.asarray(Z).astype(np.int64)
    assert Y.shape == Z.shape
    rng = np.random.default_rng(rng_seed)

    mi_obs = compute_mutual_information_binary(Y, Z, eps=0.0)

    mi_null = np.empty(n_perm, dtype=np.float64)
    Z_perm = Z.copy()
    for k in range(n_perm):
        rng.shuffle(Z_perm)  # in-place
        mi_null[k] = compute_mutual_information_binary(Y, Z_perm, eps=0.0)

    # one-sided: MI greater than null
    p = (1.0 + float(np.sum(mi_null >= mi_obs))) / (1.0 + n_perm)

    out = {
        "mi_obs": float(mi_obs),
        "p_value": float(p),
        "mi_null_p95": float(np.quantile(mi_null, 0.95)),
        "mi_null_mean": float(np.mean(mi_null)),
    }
    if return_null:
        out["mi_null"] = mi_null
    return out
```

### Важный нюанс для множественных протоколов
Если у вас 3 протокола A/B/C, то CI должен корректировать p‑values. Самое простое (и строгое) — Bonferroni:

- `p_adj = min(1, p * 3)`
- PASS если `p_adj > alpha` (т.е. не можем отвергнуть независимость).

Это фиксируйте в `NIPPermutationConfig.correction`.

---

# 3) Как CI должен проверять NIP (минимально)

1) Прочитать `nip_trials.parquet`.
2) Для каждого `protocol_id`:
   - извлечь массивы Y, Z
   - посчитать `mi_obs`, `acc_obs`, `base_rate`
   - запустить permutation test (seed фиксирован в отчёте)
   - применить correction
   - выставить `passed`
3) Проверить `passed_all = all(passed)` и сравнить с JSON‑report (или пересчитать и игнорировать report).

---

Если вы пришлёте (или перечислите) какие именно `Y` и `Z` вы решили использовать в Experiment E (например: “Y=1 если Ω‑cycle count вырос” и “Z=semantic_state.flags[...]”), я могу сразу под это дать готовые `readout_g()`/`readout_h()` и пример строки `NIPDefinitions`, чтобы контракт был полностью закрыт.
