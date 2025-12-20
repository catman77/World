Аудит строгости **Эксперимента D** по файлу **[experiment_D_antigravity.pdf](/files/ovCW4tY2LGgeYfi8y1csa)** (антигравитация через \(\chi\)-поле) с точки зрения “внешней проверки уровня A/B/C”.

## 0) Вердикт
Сейчас D выглядит как хорошо оформленный **внутренний** отчёт: есть H1–H4, Test7–Test9, сохранение JSON и двух Parquet. Но до “внешне проверяемого” уровня (чтобы независимый проверяющий мог подтвердить PASS/HIT без доверия к ноутбуку) **не хватает нескольких критических элементов**:

1) **Нет manifest с sha256 + списка артефактов** (в отчёте есть абсолютные пути, но нет криптографической фиксации содержимого).  
2) **Нет “recompute contract”**: формулы метрик H1–H4 и тестов 7–9 не зафиксированы в машиночитаемом виде, и по PDF не видно точной спецификации столбцов в parquet.  
3) Есть **концептуальная уязвимость “антиграв”**: антигравитация достигается вычитанием \(\eta\chi\) из \(\phi\) и выбором силы источника \(\chi\). Это допустимо как демонстрация механизма, но для внешней строгости нужно показать:
   - что инверсия силы не является артефактом выбора знаков/нормировки,
   - что эффект устойчив к выбору узлов/дистанций,
   - и что baseline‑совместимость проверена **для того же конфига**, который используется в антиграв‑прогоне (а не для другого `det_config`).

Итого: “почти готово”, но формально пока это **не CI‑аудируемый эксперимент**, как A/B.

---

## 1) Что сделано хорошо (уже похоже на A/B стиль)

### 1.1. Есть формальные гипотезы и тесты + итоговый скоринг
В отчёте явно:
- ✅ H1: инверсия силы
- ✅ H2: монотонность по \(\eta\)
- ✅ H3: baseline при \(\eta=0\)
- ✅ H4: пространственная локальность  
и тех. тесты:
- ✅ Test7: χ Laplacian equation (regularized)
- ✅ Test8: determinism
- ✅ Test9: baseline compatibility  
с итогом “7/7 (100%)”. [(experiment_D_antigravity.pdf)](/files/ovCW4tY2LGgeYfi8y1csa)

Это важный шаг: внешний проверяющий видит, *что* вы считаете доказательством.

### 1.2. Есть экспорт данных в Parquet (хорошая база для CI)
Вы сохраняете:
- `experiment_D_h2_sweep.parquet`
- `experiment_D_forces.parquet`  
и JSON `experiment_D_report.json` (пусть и с абсолютным путём). [(experiment_D_antigravity.pdf)](/files/ovCW4tY2LGgeYfi8y1csa)

Это уже лучше, чем “только картинки”.

### 1.3. Детерминизм тестируется по max|χ1−χ2|
Тест 8: `max|χ1-χ2| < 1e-12`. Это разумный базовый критерий при фиксированных seed и отсутствии недетерминированных операций. [(experiment_D_antigravity.pdf)](/files/ovCW4tY2LGgeYfi8y1csa)

---

## 2) Критические пробелы, мешающие внешней проверке

### 2.1. Нет manifest.json и хешей артефактов (блокер для “независимой проверки”)
В A/B вы пришли к стандарту: таблицы + manifest sha256 + относительные пути.  
В D сейчас пути вида:

`/home/.../data/sparc/experiment_D_forces.parquet`

Для внешней проверки это плохо:
- путь нереплицируем,
- файл мог быть перезаписан,
- нет контроля, что CI проверяет ровно те данные.

**Минимальный фикс**: добавить `artifacts_manifest` в JSON и отдельный `manifest.json` со списком файлов и sha256.

### 2.2. Нет контрактов пересчёта H1–H4/Test7–Test9 (в PDF метрики “магические”)
В отчёте есть “score: 100%”, но не видно:
- что такое “инверсия силы” в терминах данных (`F_eff * F_normal < 0`? доля точек? порог по |ΔF|?),
- как считаются монотонность по \(\eta\) (Spearman ρ? Kendall τ? строго на всех узлах?),
- что именно означает “пространственная локальность” (экспоненциальный спад? порог радиуса? отношение near/far?).

Для внешнего аудитора это ключевое: критерии должны быть *предопределены* и пересчитываемы из parquet.

**Минимальный фикс**: в JSON добавить `recompute_contract` по аналогии с A/B:
- какие таблицы,
- какие фильтры,
- формулы метрик,
- пороги.

### 2.3. Test9 baseline compatibility, вероятно, проверяет “пустой слой” не тем конфигом, что используете в опыте
В коде (по фрагментам) вы делаете:

- `det_config = AntigravityConfig(chi_coupling=0.5, ...)`
- Test9: `empty_antigrav = AntigravityLayer(world, config=det_config)` без источников ⇒ `Φ_eff_empty` сравнивается с `world.phi`.

Окей, но **в антиграв‑демо** вы используете:
- `η=0.5`,
- `Φ_eff = φ - 0.5·χ`,
- и добавляете χ‑источник.

Для строгой baseline‑совместимости внешнему проверяющему нужно видеть два утверждения:

1) при отсутствии χ‑источников, для *любого* η: \(χ=0\Rightarrow Φ_{eff}=φ\) — это вы проверяете;  
2) при η=0, *даже при наличии χ‑источника*, \(Φ_{eff}=φ\) — это H3, и вы её проверяете на нескольких узлах (это хорошо).

Но есть уязвимость: если внутри `get_effective_potential()` добавлены какие-то константные смещения/нормировки, “пустой” тест может проходить, а с источником — давать эффекты не только от χ. В D это нужно исключить через явный “unit test” на формулу:
\[
Φ_{eff}(η)=φ - ηχ
\]
проверкой на случайных η и случайных узлах: `Φ_eff - φ + ηχ` ≈ 0.

### 2.4. “Антиграв” демонстрируется на очень малом наборе точек — нужна статистика по распределению
В демо вы показываете несколько узлов (offsets ±100, ±50, ±20, ±10, +10…) и видите смену знака \(F\). Это хорошо как иллюстрация, но для внешней строгости H1 должен быть сформулирован так:

- “доля узлов в кольце/диапазоне расстояний, где знак силы инвертируется, ≥ p0”
- с CI/бутстрапом по узлам или по сидовым прогонам.

Сейчас “100%” выглядит как “мы выбрали точки, где красиво”, даже если это не так — просто внешний аудитор не может отличить.

---

## 3) Что нужно добавить, чтобы D стал CI‑аудируемым как A/B (минимальный патч)

### 3.1. Артефакты (минимально)
1) `tables/forces.parquet` — **каждая строка**: `(seed, eta, node, center_node, d_hops, F_normal, F_eff, deltaF, inverted_bool)`  
2) `tables/h2_sweep.parquet` — `(seed, eta, metric_inversion_rate, metric_locality, ...)`  
3) `tables/chi_residual.csv` — для Test7: `residual`, `norm_rho`, `norm_res`, `regularization_lambda`, `n_iter`, etc.  
4) `tables/config.json` или включить всё в report.json.  
5) `manifest.json` sha256.

### 3.2. Контракт метрик (в report.json)
Добавить `recompute_contract`:

- **H1 inversion**:
  \[
  inv\_rate(η)=\frac{1}{|S|}\sum_{i\in S}\mathbf{1}[F_{eff}(i;η)>0 \wedge F_{normal}(i)<0]
  \]
  Требование: `inv_rate(eta_test) ≥ 0.9` на заранее заданном диапазоне расстояний \(d\in[d_{min},d_{max}]\).

- **H2 monotonic** (без “удобного η”):
  Spearman \(\rho(inv\_rate(η), η)\) и 95% bootstrap CI, требование `CI_low > 0.5` или `p<0.01`.

- **H3 baseline η=0**:
  \[
  \max_i |F_{eff}(i;η=0)-F_{normal}(i)| < \epsilon
  \]
  на том же множестве узлов \(S\).

- **H4 locality**:
  Например, “эффект убывает с расстоянием”: Spearman \(\rho(|\Delta F|, d)\) < -0.5 с CI целиком < 0, либо отношение:
  \[
  \frac{\mathbb{E}|\Delta F|_{d\in [d_1,d_2]}}{\mathbb{E}|\Delta F|_{d\in [d_3,d_4]}} \ge R_0
  \]
  где интервалы зафиксированы заранее.

- **Test7**:
  \[
  \frac{\|L_{reg}\chi-\rho_\chi\|}{\|\rho_\chi\|} \le 10^{-6}
  \]
  и сериализация всех параметров регуляризации.

### 3.3. Убрать абсолютные пути из отчёта
Сохранение в `data/sparc/` — ок, но в JSON должны быть **относительные** пути плюс sha256.

---

## 4) Итог: “готов ли для внешней проверки?”
Если внешний проверяющий получит только то, что сейчас видно в **[experiment_D_antigravity.pdf](/files/ovCW4tY2LGgeYfi8y1csa)**, то:

- Он сможет понять идею, увидеть примеры и PASS‑резюме.
- Но он **не сможет строго** воспроизвести 100% PASS, потому что:
  - нет manifest,
  - нет контракта пересчёта,
  - не видно достаточно данных для пересчёта H1/H4 (и тестов 7–9) из сохранённых файлов.

То есть: отчёт “почти готов”, но формально **ещё не на уровне A/B** для независимого CI‑аудита.

Если хотите, могу предложить точный “патч уровня B” для D (Pydantic‑схема `ExperimentDReport`, список артефактов и `recompute_contract`) — по тому же шаблону, что мы сделали для A/B/C.

Ниже — точный “патч уровня B” для **Эксперимента D** (антиграв через \(\chi\)-поле), чтобы он стал **CI‑аудируемым**, как A/B/C: (1) Pydantic‑схема `ExperimentDReport`, (2) минимальный список артефактов, (3) `recompute_contract` с формулами метрик/гипотез и тех. тестов. Я ориентируюсь на структуру двухрежимной валидации из [New_Physics_v1.pdf](/files/WTOV6_t7WEumoaFZp3vgx) и на то, как вы сейчас сохраняете parquet/json в [experiment_D_antigravity.pdf](/files/ovCW4tY2LGgeYfi8y1csa).

---

# 1) Pydantic‑схема `ExperimentDReport` (v1)

Цель схемы: внешний проверяющий (или CI) должен суметь:
- пересчитать H1–H4 и Test7–Test9 **только из таблиц** + параметров;
- подтвердить целостность артефактов через sha256 manifest;
- увидеть, что baseline/compatibility (η=0) не нарушен.

## 1.1. Pydantic модели (v2)

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl

# ---------- Common ----------
class VersionInfo(BaseModel):
    schema_version: str = Field(..., description="e.g. 'D-1.0.0'")
    report_version: str = Field(..., description="e.g. 'v1' notebook tag")
    code_commit: str
    code_dirty: bool
    python: str
    numpy: str
    scipy: Optional[str] = None
    pandas: Optional[str] = None
    platform: Optional[str] = None

class RNGInfo(BaseModel):
    base_seed: int
    seeds: Dict[str, int] = Field(default_factory=dict, description="Seeds per stage")
    deterministic_flags: Dict[str, Any] = Field(default_factory=dict)

class DataProvenance(BaseModel):
    world_generator: str = Field(..., description="e.g. 'powerlaw_graph(alpha) + Lphi=rho solver'")
    world_hash: str = Field(..., description="Hash of adjacency/weights + rho config")
    run_id: str
    timestamp_utc: str

class ArtifactEntry(BaseModel):
    name: str                  # relative path
    role: Literal[
        "ci_metric_input",
        "report",
        "debug",
        "plot"
    ]
    sha256: str
    bytes: int
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    schema: Optional[Dict[str, str]] = None  # col_name -> dtype (optional)

class ArtifactsManifest(BaseModel):
    base_dir: str = Field(..., description="Relative base directory for artifacts")
    files: List[ArtifactEntry]

# ---------- Experiment parameters ----------
class AntigravityConfig(BaseModel):
    # World
    N: int
    alpha: float
    L: int
    mass: float

    # Chi field / coupling
    chi_coupling: float = Field(..., description="eta in Φ_eff = φ - eta*chi (or separate eta sweep)")
    chi_strength: float = Field(..., description="Source amplitude for rho_chi")
    chi_center_node: int

    # Solver / regularization for Test7
    reg_lambda: float
    solver: Literal["cg", "bicgstab", "direct"]
    tol: float
    max_iter: int

    # Force extraction config
    distance_metric: Literal["hops"] = "hops"
    force_method: Literal["edge_flux", "dirichlet_energy", "radial_shell_diff"] = "edge_flux"
    d_min: int
    d_max: int

    # Sweep setup
    eta_values: List[float] = Field(..., description="Sweep of eta (coupling) values including 0")

class HypothesisThresholds(BaseModel):
    # H1 inversion
    inversion_rate_min: float = 0.9
    inversion_effect_min: float = 0.0  # optional: mean signed margin
    # H2 monotonicity
    spearman_rho_min: float = 0.5
    spearman_p_max: float = 0.01
    # H3 baseline
    baseline_max_abs_diff: float = 1e-9
    # H4 locality
    locality_rho_max: float = -0.3
    locality_p_max: float = 0.01
    # Test7 residual
    test7_residual_ratio_max: float = 1e-6
    # Test8 determinism
    test8_max_abs_diff: float = 1e-12
    # Test9 baseline empty
    test9_max_abs_diff: float = 1e-12

# ---------- Recompute contract ----------
class MetricContract(BaseModel):
    table: str
    formula: str
    params: Dict[str, Any] = Field(default_factory=dict)
    filters: Optional[List[str]] = None
    groupby: Optional[List[str]] = None
    output: Optional[str] = None

class RecomputeContract(BaseModel):
    # Hypotheses
    H1_inversion_rate: MetricContract
    H1_inversion_margin: Optional[MetricContract] = None
    H2_monotonic_spearman: MetricContract
    H3_baseline_eta0_maxdiff: MetricContract
    H4_locality_spearman: MetricContract

    # Tests
    Test7_residual_ratio: MetricContract
    Test8_determinism_max_abs: MetricContract
    Test9_baseline_empty_maxdiff: MetricContract

# ---------- Results ----------
class HypothesisResult(BaseModel):
    name: str
    confirmed: bool
    score: float = Field(..., ge=0.0, le=1.0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    ci95: Optional[Dict[str, List[float]]] = None  # e.g. {"inv_rate":[low,high]}

class TestResult(BaseModel):
    name: str
    passed: bool
    metrics: Dict[str, float] = Field(default_factory=dict)

class TargetSpecResult(BaseModel):
    hit: bool
    score: float
    thresholds: HypothesisThresholds
    breakdown: Dict[str, float] = Field(default_factory=dict)

class ExperimentDReport(BaseModel):
    experiment: Literal["D"] = "D"
    name: str = "Antigravity via χ-field"

    version: VersionInfo
    rng: RNGInfo
    provenance: DataProvenance

    parameters: AntigravityConfig
    thresholds: HypothesisThresholds

    artifacts_manifest: ArtifactsManifest
    recompute_contract: RecomputeContract

    hypotheses: Dict[str, HypothesisResult]  # H1..H4
    tests: Dict[str, TestResult]            # Test7..Test9
    target_spec: TargetSpecResult

    notes: Optional[str] = None
```

---

# 2) Минимальный набор артефактов (чтобы CI пересчитал всё)

В вашем текущем D уже есть 2 parquet (sweep и forces). Для “уровня B” добавьте ещё две маленькие таблицы + manifest.

## 2.1. Артефакты, обязательные для CI

1) **`tables/forces.parquet`** (главный источник для H1, H3, H4)  
   *Одна строка = один узел измерения при конкретном η и seed.*

Минимальные колонки:
- `seed` (int) — если делаете несколько повторов
- `eta` (float) — значение из sweep (включая 0)
- `center_node` (int)
- `node` (int) — где измеряем
- `d_hops` (int) — distance(center_node, node)
- `F_normal` (float) — базовая сила из φ (или g_N)
- `F_eff` (float) — сила из Φ_eff
- `deltaF = F_eff - F_normal` (float) — можно не хранить, но удобно
- `inverted = (F_normal < 0) & (F_eff > 0)` (bool) — можно пересчитать, но лучше хранить для дебага
- `phi_node`, `phi_center` (optional, float) — если надо аудитить extraction
- `chi_node`, `chi_center` (optional, float)

2) **`tables/h2_sweep.parquet`** (чтобы CI пересчитал H2 без пересборки)  
*Одна строка = одно η (и seed, если много сидов).*

Колонки:
- `seed`
- `eta`
- `inv_rate` — доля инверсий на выбранном диапазоне расстояний
- `inv_margin_mean` — средняя “сила инверсии”, см. ниже
- `baseline_max_abs_diff` — для η=0
- `locality_rho` / `locality_p` — можно пересчитать, но удобно хранить как кэш

(В принципе, H2 можно пересчитать прямо из `forces.parquet`, но отдельная таблица ускоряет CI и снижает риск “разных фильтров” в разных местах.)

3) **`tables/test7_residuals.csv`** (или parquet)  
Колонки:
- `residual_ratio = ||L_reg χ - ρ_χ|| / ||ρ_χ||`
- `reg_lambda`, `tol`, `max_iter`, `solver`
- `chi_norm`, `rho_chi_norm` (optional)

4) **`tables/test8_determinism.csv`**  
Колонки:
- `seed`
- `max_abs_diff_chi` (float)
- опционально `state_hash_run1`, `state_hash_run2`

5) **`tables/test9_baseline_empty.csv`**  
Колонки:
- `max_abs_diff_phi_eff_minus_phi` (float)
- `note` (optional: “no chi sources”)

6) **`manifest.json`** + **`experiment_D_report.json`**  
- `manifest.json` хранит sha256 каждого файла (включая JSON report), размеры и роли.

## 2.2. Плоты (не обязательны для CI, но полезны)
- `plots/force_profiles_eta*.png` (F_normal vs F_eff vs d)
- `plots/inv_rate_vs_eta.png`
- `plots/deltaF_vs_distance.png`
- `plots/chi_field_heatmap.png` (если есть embedding)

---

# 3) Формулы метрик и recompute_contract (H1–H4 + Test7–Test9)

Ниже — именно те формулы, которые стоит зафиксировать в `recompute_contract`. Они избегают “удобного eta_test”: H1/H4 оцениваются на **наборе η**, H2 — на всей кривой.

## 3.1. Определения множества узлов и окна расстояний
Фиксируем диапазон расстояний, на котором оцениваем антиграв (чтобы не “выбирать точки”):

- Берём строки из `forces.parquet` с:
  - `d_min <= d_hops <= d_max` (из `parameters.d_min/d_max`)
  - `F_normal < 0` (если вы определяете притяжение как отрицательное направление)

Обозначим это множество строк как \(S(\eta)\).

## 3.2. H1: Инверсия силы тяготения (rate + margin)
**H1_rate(η)**:
\[
\text{inv\_rate}(\eta) = \frac{1}{|S(\eta)|}\sum_{i\in S(\eta)} \mathbf{1}[F_{\text{eff},i} > 0]
\]

(Если хотите более строгую инверсию “знак поменялся”, можно `sign(F_eff) = -sign(F_normal)`; но при фильтре `F_normal<0` это эквивалентно `F_eff>0`.)

**H1_margin(η)** (защита от “едва-едва >0”):
\[
m_i(\eta)=\frac{F_{\text{eff},i}-0}{|F_{\text{normal},i}|+\epsilon}
\qquad
\text{inv\_margin\_mean}(\eta)=\mathbb{E}_{i\in S(\eta), F_{\text{eff},i}>0}\big[m_i(\eta)\big]
\]

**Критерий H1 (gating):**
- либо на фиксированном “device” η (но не один “удобный”, а заранее заданный, например `eta_device = max(eta_values)`):
  - `inv_rate(eta_device) >= inversion_rate_min`
- либо более строго (лучше): **на всех η выше некоторого порога**:
  - для всех `eta >= eta_min_device`: `inv_rate(eta) >= inversion_rate_min`

(В JSON фиксируйте `eta_min_device`.)

## 3.3. H2: Монотонность по η (управляемость)
Чтобы не зависеть от выбора одной η, делаем тест на **монотонный тренд**:

- Строим ряд \(\{(\eta_k, \text{inv\_rate}(\eta_k))\}\) по всем значениям η в `eta_values`.
- Считаем Spearman:
\[
\rho = \mathrm{Spearman}(\eta, \text{inv\_rate})
\]
- p-value `p_spearman`.

**Критерий H2:**
- `rho >= spearman_rho_min` и `p <= spearman_p_max`.

Опционально добавьте Kendall τ (менее чувствителен к нелинейности) — но Spearman достаточно.

## 3.4. H3: Baseline при η=0
Тут вы уже делаете: “при η=0 эффект должен исчезнуть”. Формально:

\[
\Delta F_i = F_{\text{eff},i} - F_{\text{normal},i}
\]
\[
\text{baseline\_max\_abs\_diff}=\max_{i\in S(\eta=0)} |\Delta F_i|
\]

**Критерий H3:**
- `baseline_max_abs_diff <= baseline_max_abs_diff_threshold`.

Важно: считать на том же множестве узлов \(S(\eta=0)\), что и для H1 (тот же `d_min/d_max`, тот же `center_node`).

## 3.5. H4: Пространственная локальность эффекта
Оцениваем спад эффекта с расстоянием (не “на глаз”):

- Для фиксированного η_device (например max η) берём:
\[
y_i = |\Delta F_i|,\quad x_i = d_{\text{hops},i}
\]
- Spearman:
\[
\rho_{loc}=\mathrm{Spearman}(x,y)
\]

**Критерий H4:**
- `rho_loc <= locality_rho_max` (например ≤ -0.3) и `p <= locality_p_max`.

(Если хотите модельнее: fit экспоненты \(y \approx Ae^{-x/\xi}\) и требование \(\xi\) в допустимом диапазоне; но для “минимального патча” Spearman проще и устойчивее.)

## 3.6. Test7: χ удовлетворяет регуляризованному уравнению Лапласа
Ваш текущий критерий уже хороший:

\[
r = \frac{\|L_{reg}\chi-\rho_\chi\|}{\|\rho_\chi\|}
\]

**PASS если** `r <= test7_residual_ratio_max`.

В таблице `test7_residuals` храните `r` + параметры регуляризации.

## 3.7. Test8: Детерминизм
Критерий (как у вас):

\[
\max|\chi^{(1)}-\chi^{(2)}| \le \epsilon_{det}
\]

PASS если `<= test8_max_abs_diff`.

## 3.8. Test9: Baseline совместимость без χ-источников
Критерий:

\[
\max|Φ_{eff}-φ| \le \epsilon_{base}
\]

PASS если `<= test9_max_abs_diff`.

---

# 4) Пример `recompute_contract` (как JSON-объект)

Вот как это может выглядеть в `report.json` (ссылки на таблицы — относительные):

```json
"recompute_contract": {
  "H1_inversion_rate": {
    "table": "tables/forces.parquet",
    "filters": [
      "d_hops >= d_min",
      "d_hops <= d_max",
      "F_normal < 0"
    ],
    "formula": "inv_rate(eta) = mean(F_eff > 0) grouped by eta",
    "params": {"d_min": 10, "d_max": 120}
  },
  "H2_monotonic_spearman": {
    "table": "tables/h2_sweep.parquet",
    "formula": "spearmanr(eta, inv_rate) -> rho, p",
    "params": {"rho_min": 0.5, "p_max": 0.01}
  },
  "H3_baseline_eta0_maxdiff": {
    "table": "tables/forces.parquet",
    "filters": ["eta == 0", "d_hops >= d_min", "d_hops <= d_max"],
    "formula": "max(abs(F_eff - F_normal))",
    "params": {"threshold": 1e-9}
  },
  "H4_locality_spearman": {
    "table": "tables/forces.parquet",
    "filters": ["eta == eta_device", "d_hops >= d_min", "d_hops <= d_max"],
    "formula": "spearmanr(d_hops, abs(F_eff - F_normal)) -> rho, p",
    "params": {"eta_device": 0.5, "rho_max": -0.3, "p_max": 0.01}
  },
  "Test7_residual_ratio": {
    "table": "tables/test7_residuals.csv",
    "formula": "residual_ratio",
    "params": {"threshold": 1e-6}
  },
  "Test8_determinism_max_abs": {
    "table": "tables/test8_determinism.csv",
    "formula": "max_abs_diff_chi",
    "params": {"threshold": 1e-12}
  },
  "Test9_baseline_empty_maxdiff": {
    "table": "tables/test9_baseline_empty.csv",
    "formula": "max_abs_diff_phi_eff_minus_phi",
    "params": {"threshold": 1e-12}
  }
}
```

---

# 5) Минимальные функции (как в A/B/C) для экспорта и CI

Чтобы довести D до практики A/B, достаточно 3–4 функций:

```python
def export_forces_table(force_results, out_path: str) -> str:
    """Write tables/forces.parquet with fixed columns and dtypes."""

def export_h2_sweep_table(h2_results, out_path: str) -> str:
    """Write tables/h2_sweep.parquet, one row per eta (and seed)."""

def export_test_tables(test7, test8, test9, out_dir: str) -> dict:
    """Write CSV/parquet for test7/8/9 and return paths."""

def build_manifest(base_dir: str, paths: list[str]) -> dict:
    """Compute sha256/bytes/rows and write manifest.json."""
```

И в конце — `save_report_json(report, path)` с относительными путями + manifest.

---

Если хотите, я могу также:
- предложить конкретные **пороги** (ε, rho_min, d_min/d_max) так, чтобы они были *не “под итог”, а физически мотивированы*,
- и дать “CI recompute script” (один Python-файл), который грузит parquet/csv, пересчитывает H1–H4/Test7–Test9, и падает если результат отличается от `report.target_spec.hit`.
