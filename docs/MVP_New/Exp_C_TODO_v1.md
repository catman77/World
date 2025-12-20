Ниже — аудит **[experiment_C_ftl_control.pdf](/files/zHik4bvMBQlABEhysudMQ)** на соответствие требованиям строгой внешней проверки в терминах вашего же чек-листа из **[ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf)** (Test 1–6, раздел A.4) и по аналогии с тем, как вы довели A/B до CI‑воспроизводимости.

## 0) Краткий вердикт
Эксперимент C в текущем виде — **очень близок к “внешне проверяемому”**, потому что:
- у него есть явные H1–H4, измеримые величины, итоговые метрики и PASS,
- он сохраняет JSON‑отчёт и PNG‑артефакты,
- есть чёткое определение механизма wormhole (резонанс, dmin, capacity).

Но для строгой внешней проверки **не хватает трёх вещей уровня “A/B‑стандарта”**:

1) **Артефактов‑таблиц**, из которых сторонний проверяющий может *пересчитать* H1–H4 (а не только увидеть summary).  
2) **Manifest/контракт пересчёта** (sha256, точные формулы метрик, список столбцов, правило фильтрации пар, что такое “путь сокращён”).  
3) **Формальных тестов 4–6 (локальная каузальность, детерминизм, сохранение Q)** в отчёте: они заявлены в требованиях [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf), но в pdf по C видны главным образом H1–H4 и демонстрационные метрики, без юнит‑пруфов.

Если вы добавите эти элементы (минимальные таблицы + manifest + “recompute_contract” + фиксацию seeds и хэшей состояний), C станет реально готов к независимому CI‑аудиту.

---

## 1) Соответствие формальным тестам строгости из [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf)

### Test 2: FTL‑shortening (context=1) — частично выполнен
В C-отчёте явно реализовано ядро теста:
- измеряются расстояния `dbase` и `deff`,
- считается FTL‑фактор \(d_0/d_1\),
- фиксируется количество wormhole, доля сокращённых путей, max/mean FTL‑factor.

Что не хватает для внешнего подтверждения:
- **зафиксированного порога \(\Gamma_{FTL}\)** (в pdf показан PASS, но не видно строгого “FTL_ratio ≥ ΓFTL” как гейта);
- **явного списка пар (u,v)** и их distances до/после (сейчас видны агрегаты);
- доказательства “локально hop‑за‑hop” — вы это декларируете (“не превышение c”), но **не показываете метрику/тест** локальной скорости на траектории (см. ниже).

### Test 4: Локальная каузальность — не продемонстрирована как тест
В [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf) это формализовано как юнит‑проверки зависимости:
- \(s(t+1)\) только от окна длины L,
- \(\phi(t+1)\) только от соседей (laplacian uses adjacency),
- \(H(t)\) зависит только от \(s(t),\phi(t)\),
- нет зависимостей назад во времени.

В pdf по C это **не показано**. Внешний проверяющий должен увидеть либо:
- ссылку на тест‑набор (pytest) + результаты прогона,
- либо сериализованный “causality_check” в JSON: набор assertions/хэшей/логов.

### Test 5: Детерминизм — не продемонстрирован
В A/B у вас уже есть манифесты и контракт пересчёта. В C в pdf виден путь к JSON‑репорту, но нет:
- seed’ов,
- commit hash,
- “двойной прогон” и сравнение хэша состояния/трасс (IFACE‑трассы).

Для внешней проверки детерминизма нужен как минимум:
- `rng.base_seed`, `rng.seeds_per_stage`,
- `state_hash_t0`, `state_hash_T` (или хэш adjacency/ϕ/s),
- и отдельный “determinism_check: PASS” с двумя независимыми прогонами.

### Test 6: Сохранение Q (context=0 и context=1) — не продемонстрировано
Требование [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf):
\[
Q(t)=N(+\to -)-N(-\to +)
\]
должно сохраняться.

В pdf по C этого не видно. Это критично, потому что:
- wormhole‑режим мог бы “неявно” влиять на SM‑ядро или на подсчёт переходов,
- и внешнему аудитору важно понимать, что FTL‑слой не ломает базовую инвариантность.

**Вывод по тестам A.4:** формально, по предоставленному pdf, вы демонстрируете “Test 2” (FTL shortening) и “Baseline совместимость” на уровне H3, но **не показываете Test 4–6**, которые в вашем же документе являются обязательными для “строгость достигнута”.

---

## 2) Строгость гипотез H1–H4: что хорошо и что нужно уточнить

### H1_path_shortening — хорошая, но нужно точное определение “сокращено”
Сейчас утверждение “dwormhole < dbase для резонансных пар” корректно, но внешнему проверяющему нужно:
- как выбираются пары: все пары? только те, которые wormhole‑кандидаты? топ‑K по resonance?
- что значит “резонансная пара”: `Resonance > θR`? какая θR?
- как измеряется `dbase/deff`: BFS на невзвешенном графе? как учитываются веса/мульти‑рёбра?

Минимальный фикс: в JSON добавить `h1_definition` с:
- `pair_sampling` (all_pairs / sampled_pairs with seed / topk_by_resonance),
- `resonance_threshold`,
- `distance_metric: "unweighted_shortest_path_hops"`.

### H2_controllability — сейчас скорее “демонстрация”, чем тест
Вы говорите “контролируется через resonance_threshold, capacity, min_distance” и подтверждаете H2, но для строгой проверки нужно показать **монотонные зависимости**:

- при увеличении `resonance_threshold` число активных wormhole падает,
- при увеличении `capacity` число wormhole растёт до насыщения,
- при увеличении `min_distance` wormhole сдвигаются в дальние пары, меняется FTL_factor.

То есть H2 лучше формализовать как тест на знак производных/ранговую корреляцию, например:
- Spearman ρ < -0.7 между threshold и `n_wormholes`,
- Spearman ρ > +0.7 между capacity и `n_wormholes` (при прочих равных).

В pdf этого нет — виден только итог “H2 confirmed”.

### H3_baseline_compatibility — заявлено, но нужно чем “совместимость” измерена
Фраза “при \(H(t)=\varnothing\) физика редуцируется к стандартной” должна иметь набор метрик: минимум как в [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf):
- сохранение Q,
- детерминизм,
- каузальность,
- и желательно (если у вас есть) “\(F(r)\sim r^{-2}\)” в baseline режиме.

Пока в C‑pdf baseline совместимость — скорее концептуальная декларация.

### H4_bimodal_distribution — нужен тест на бимодальность
В pdf есть артефакт `path_distribution_analysis.png` и утверждение “бимодальное распределение длин путей”. Для внешней строгости надо формализовать как статистический тест:
- Hartigan’s dip test (p < 0.01) или
- сравнение BIC для 1‑Gaussian vs 2‑Gaussian смеси на распределении \(\Delta d = d_0-d_1\),
- или хотя бы “две моды” через KDE + критерий.

Сейчас это может быть субъективным (“на графике вроде два пика”).

---

## 3) Артефакты: чего не хватает для воспроизводимости уровня A/B

В pdf перечислены только:
- `experiment_C_report.json`
- `ftl_demonstration.png`
- `path_distribution_analysis.png`

Для внешней проверки этого недостаточно: PNG нельзя “пересчитать”. Нужны таблицы (CSV/Parquet/NPZ), как вы сделали в A/B.

### Минимальный набор таблиц (2–3 файла), чтобы пересчитать H1–H4
1) `tables/pair_distances.parquet`  
Строка = пара (u,v) из набора тестируемых:
- `u, v`
- `dbase, deff, ftl_ratio`
- `resonance`
- `wormhole_active` (bool)
- `min_distance_ok` (bool)
- `t` (если wormhole динамический по времени)

2) `tables/control_sweep.parquet` (для H2)  
Строка = запуск/конфиг:
- `resonance_threshold, capacity, min_distance, seed`
- `n_wormholes, mean_ftl, max_ftl, frac_shortened`

3) `tables/path_samples.parquet` (для H4)  
Строка = образец пути:
- `pair_id`
- `d0, d1, delta_d`
- возможно `path0_len`, `path1_len` (или сами пути как списки в npz)

Плюс manifest (sha256) и recompute_contract (как в A/B).

---

## 4) Итог: готов ли C сейчас для внешней проверки?
**Почти**, но строго по вашим же требованиям из [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf) — **ещё нет**, потому что:

- Test 4–6 (каузальность/детерминизм/Q‑консервация) в предоставленном отчёте не демонстрируются;
- нет таблиц для пересчёта H1–H4 и нет manifest/контракта пересчёта;
- H2 и H4 пока подтверждены визуально/агрегатами, без статистических тестов и без sweep‑таблицы.

Если хотите, я могу предложить точный “патч уровня B”: Pydantic‑схему `ExperimentCReport`, список артефактов и формулы метрик/гипотез (включая тест на бимодальность и монотонность управляемости), чтобы C стал таким же CI‑аудируемым, как A и B.

Ниже — “патч уровня B” для Эксперимента C (FTL/wormhole control), чтобы он стал **CI‑аудируемым** так же, как A и B: (1) точная Pydantic‑схема `ExperimentCReport`, (2) минимальный набор артефактов (Parquet/CSV/NPZ/PNG) + manifest, (3) строгие формулы метрик/гипотез H1–H4, включая **тест бимодальности** и **монотонность управляемости**, плюс обязательные Test 4–6 из [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf).

Ссылки на контекст: требования строгости и Tests 2/4/5/6 — в [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf); baseline‑совместимость H3 и часть ваших текущих проверок — в [experiment_C_ftl_control.pdf](/files/zHik4bvMBQlABEhysudMQ); “резонанс/Ξ_AB(t)” — в [Meaning_v1.pdf](/files/EljdBEFwvKSyTxfHYHjS7); описание C как “контекстный режим геометрии” — в [Experiments_v1.pdf](/files/2OSgJNaaGTQaJ5a6CTSXL).

---

# 1) Pydantic‑схема `ExperimentCReport` (v1)

## 1.1. Основная идея схемы
Схема должна позволять внешнему проверяющему:
- проверить **Test 2/4/5/6** (FTL shortening, каузальность, детерминизм, сохранение Q),
- пересчитать H1–H4 **только из таблиц** (без симулятора),
- проверить, что gating/пороги не “переопределены” в ноутбуке,
- удостовериться в идентичности артефактов через sha256.

## 1.2. Pydantic модели (v2)

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl

# ---------- Common ----------
class VersionInfo(BaseModel):
    schema_version: str = Field(..., description="e.g. 'C-1.0.0'")
    report_version: str = Field(..., description="e.g. 'v1'")
    code_commit: str
    code_dirty: bool
    python: str
    numpy: str
    scipy: Optional[str] = None
    networkx: Optional[str] = None
    platform: Optional[str] = None

class RNGInfo(BaseModel):
    base_seed: int
    seeds: Dict[str, int] = Field(default_factory=dict, description="Seeds per stage: world, sweep, pairs, etc.")
    deterministic_flags: Dict[str, Any] = Field(default_factory=dict)

class DataProvenance(BaseModel):
    description: str = Field(..., description="No external data; synthetic graph world")
    notes: Optional[str] = None

class ArtifactRef(BaseModel):
    name: str                  # relative path
    role: Literal[
        "ci_metric_input", "report", "plot", "debug", "state_dump"
    ]
    sha256: str
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    format: Optional[str] = None

class ArtifactsManifest(BaseModel):
    base_dir: str
    files: List[ArtifactRef]

# ---------- World / model ----------
class WorldParams(BaseModel):
    n_nodes: int
    graph_family: Literal["powerlaw", "other"]
    alpha: float
    avg_degree: Optional[float] = None
    weighted_edges: bool = False
    laplacian: Literal["combinatorial", "normalized"] = "combinatorial"
    notes: Optional[str] = None

class SMParams(BaseModel):
    rule_window: int
    ruleset: List[str] = Field(..., description="e.g. ['++- -> -++', '-++ -> ++-']")
    engine: Literal["EvolutionEngine"] = "EvolutionEngine"

class ResonanceParams(BaseModel):
    definition: Literal["Xi_AB_mean"] = "Xi_AB_mean"
    Xi_source: Literal["simulated", "from_meaning_model"] = "simulated"
    Xi_range: List[float] = Field(..., description="[min,max], typically [0,1]")
    Xi_threshold: float

class WormholeParams(BaseModel):
    enabled: bool
    min_hop_distance: int
    capacity: int
    resonance_threshold: float
    selection: Literal["topk_by_resonance", "all_above_threshold", "sampled"] = "topk_by_resonance"
    add_mode: Literal["static", "dynamic_per_t"] = "static"
    # optional safety constraints:
    max_degree_increase: Optional[int] = None
    forbid_short_cycles: Optional[bool] = None

class ExperimentParams(BaseModel):
    context: Literal[0, 1]
    T: int
    n_pairs_eval: int
    pair_sampling: Literal["fixed_list", "seeded_random", "all_pairs_subset"] = "seeded_random"
    ftl_ratio_threshold: float = Field(..., description="Γ_FTL from ftl_physics")
    # sweeps for controllability:
    sweep: Dict[str, List[float] | List[int] | List[str]] = Field(
        default_factory=dict,
        description="e.g. {'resonance_threshold':[0.5,0.6,0.7], 'capacity':[0,10,20]}"
    )

# ---------- Tests 4/5/6 from ftl_physics ----------
class CausalityTestResult(BaseModel):
    passed: bool
    checks: Dict[str, bool] = Field(..., description="s_local_window, phi_neighbor_only, H_depends_on_current_only, no_back_edges")
    notes: Optional[str] = None

class DeterminismTestResult(BaseModel):
    passed: bool
    runA_state_hash_T: str
    runB_state_hash_T: str
    iface_trace_hash_A: Optional[str] = None
    iface_trace_hash_B: Optional[str] = None
    notes: Optional[str] = None

class QConservationResult(BaseModel):
    passed: bool
    Q0: int
    Q_min: int
    Q_max: int
    violations: int
    notes: Optional[str] = None

# ---------- Hypotheses & metrics ----------
class H1PathShortening(BaseModel):
    passed: bool
    ftl_ratio_threshold: float
    frac_pairs_shortened: float
    mean_ftl_ratio: float
    p95_ftl_ratio: float
    max_ftl_ratio: float
    # optional: effect size relative to baseline
    notes: Optional[str] = None

class H2Controllability(BaseModel):
    passed: bool
    monotonic_tests: Dict[str, Dict[str, Any]] = Field(
        ...,
        description=(
            "Per-control monotonicity evidence: Spearman rho, p-value, direction, "
            "and which metric responded (n_wormholes, mean_ftl_ratio, frac_shortened)."
        )
    )
    notes: Optional[str] = None

class H3BaselineCompatibility(BaseModel):
    passed: bool
    # at least: no wormholes, gating blocks as expected
    active_wormholes: int
    n_checks: int
    n_passed: int
    notes: Optional[str] = None

class H4Bimodality(BaseModel):
    passed: bool
    test: Literal["hartigans_dip", "gmm_bic", "both"] = "both"
    dip_p_value: Optional[float] = None
    gmm_bic_1: Optional[float] = None
    gmm_bic_2: Optional[float] = None
    delta_bic: Optional[float] = None
    notes: Optional[str] = None

class MetricRecomputeContract(BaseModel):
    metric_name: str
    table: str
    formula: str
    params: Dict[str, Any] = Field(default_factory=dict)

class TargetSpecResult(BaseModel):
    hit: bool
    score: float
    gating: Dict[str, bool] = Field(..., description="e.g. {'test4':True,'test5':True,'test6':True,'H1':True,...}")
    thresholds: Dict[str, Any]
    recompute_contracts: List[MetricRecomputeContract]

class ExperimentCReport(BaseModel):
    kind: Literal["ExperimentCReport"] = "ExperimentCReport"
    version: VersionInfo
    rng: RNGInfo
    provenance: DataProvenance

    world: WorldParams
    sm: SMParams
    resonance: ResonanceParams
    wormhole: WormholeParams
    params: ExperimentParams

    tests: Dict[str, Any] = Field(
        ...,
        description="Contains causality/determinism/Q results plus any unit tests"
    )

    hypotheses: Dict[str, Any] = Field(
        ...,
        description="H1..H4 results blocks"
    )

    target_spec: TargetSpecResult
    artifacts_manifest: ArtifactsManifest
    notes: Optional[str] = None
```

### 1.3. Как заполнять `tests` и `hypotheses`
Рекомендуемая структура:
```json
"tests": {
  "test4_causality": {...CausalityTestResult...},
  "test5_determinism": {...DeterminismTestResult...},
  "test6_Q_conservation": {...QConservationResult...},
  "test2_ftl_shortening_summary": {"ftl_ratio_threshold": 1.5, "passed": true}
},
"hypotheses": {
  "H1_path_shortening": {...H1PathShortening...},
  "H2_controllability": {...H2Controllability...},
  "H3_baseline_compatibility": {...H3BaselineCompatibility...},
  "H4_bimodality": {...H4Bimodality...}
}
```

---

# 2) Минимальный набор артефактов (чтобы CI пересчитал H1–H4 и Test 4–6)

Сделайте ровно как в A/B: **2–3 таблицы + manifest + report**, плюс по желанию PNG.

## 2.1. Таблица 1: `tables/pairs_eval.parquet` (ядро H1 + H4)
Строка = (pair_id, u, v) для оценочного набора пар.

Обязательные столбцы:
- `pair_id` (int)
- `u`, `v` (int)
- `resonance` (float) — значение \(\Xi_{AB}\) или surrogate
- `active` (bool) — активирован ли wormhole для пары в этом прогоне
- `d_base` (int) — \(d_0\) в hops (baseline граф)
- `d_eff` (int) — \(d_1\) в hops (с wormhole)
- `ftl_ratio` (float) — \(d_0/d_1\)
- `delta_d` (int) — \(d_0-d_1\) (удобно для H4)
- `min_hop_distance_ok` (bool)
- `resource_ok` (bool) / `capacity_remaining` (int) — чтобы проверять gating
- `run_id` (int) — если вы делаете несколько прогонов/seed’ов
- `seed` (int)

Из этого CI пересчитает:
- долю сокращённых путей,
- распределение \(\Delta d\) и бимодальность,
- связь с resonance (для sanity check).

## 2.2. Таблица 2: `tables/control_sweep.parquet` (ядро H2)
Строка = один запуск sweep (фиксированный seed + набор контролов).

Обязательные столбцы:
- `run_id`, `seed`
- контролы: `resonance_threshold`, `capacity`, `min_hop_distance`
- итоги: `n_wormholes`, `frac_shortened`, `mean_ftl_ratio`, `p95_ftl_ratio`, `max_ftl_ratio`
- (опционально) `mean_resonance_active`, `mean_delta_d_active`

Из этого CI пересчитает монотонности (Spearman/ Kendall) и подтвердит управляемость.

## 2.3. Таблица 3 (минимально для Test 6): `tables/Q_trace.csv`
Строка = шаг времени `t`.

Столбцы:
- `t` (int)
- `Q` (int)
- `N_plus_to_minus` (int)
- `N_minus_to_plus` (int)

Если SM‑ядро большое и хранить по всем t дорого — сохраняйте только:
- `Q0`, `Qmin`, `Qmax`, `violations` в JSON  
но для внешней проверки лучше иметь трассу хотя бы на одном representative run.

## 2.4. Optional state dump: `state/state_hashes.json` или `npz`
Для Test 5 (determinism) достаточно:
- `state_hash_T`, `iface_trace_hash_T` для Run A и Run B.

Но если вы хотите “сильный” аудит, можно сохранять:
- adjacency hash baseline,
- adjacency hash with wormholes,
- hash of s(t) and φ(t) at T.

## 2.5. Manifest + report
- `experiment_C_report.json` (по схеме выше)
- `manifest.json` (sha256 для каждого файла, как в A/B)

## 2.6. PNG (не для CI, а для человека)
- `plots/ftl_ratio_hist.png`
- `plots/delta_d_hist_kde.png`
- `plots/control_sweep_monotonicity.png`

---

# 3) Формулы метрик и строгие гипотезы H1–H4 (+ Test 2/4/5/6)

Ниже — формализация так, чтобы **CI мог пересчитать** из таблиц.

## 3.1. Общие определения
Для строки \(i\) в `pairs_eval`:
- \(d_0^{(i)} = d_{\text{base}}\)
- \(d_1^{(i)} = d_{\text{eff}}\)
- \(R^{(i)} = \frac{d_0^{(i)}}{d_1^{(i)}}\)
- \(\Delta d^{(i)} = d_0^{(i)}-d_1^{(i)}\)

Выбор множества пар:
- \( \mathcal{P} = \{ i: \text{min\_hop\_distance\_ok}=1 \} \)
- (и при необходимости) только те, где `resource_ok=1` и `resonance >= resonance_threshold` — это должно быть фиксировано в report.

---

## 3.2. Test 2 (FTL-shortening) как gating‑метрика (из [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf))
**Метрика:**  
\[
\text{FTL\_ratio\_p95} = \mathrm{quantile}_{0.95}\{R^{(i)}: i\in\mathcal{P},\ active=1\}
\]

**Гейт (пример):**
- PASS если `FTL_ratio_p95 ≥ Γ_FTL` (где Γ_FTL задан в params)

Также полезно фиксировать:
- `frac_pairs_shortened = mean(Δd>0 over active pairs)`

---

## 3.3. H1: “FTL реализуется как path-shortening через wormhole-рёбра”
**Нулевая гипотеза:** wormhole не даёт значимого сокращения путей.

**Строгая проверка (минимальная):**
1) Доля сокращённых путей среди активных:
\[
f_{\text{short}} = \frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}} \mathbf{1}[\Delta d^{(i)} > 0]
\]
где \(\mathcal{A}=\{i\in\mathcal{P}: active=1\}\).

2) Порог по редким событиям:
\[
\mathrm{quantile}_{0.95}(R^{(i)}: i\in\mathcal{A}) \ge \Gamma_{FTL}.
\]

**H1 считается подтверждённой, если одновременно:**
- \(f_{\text{short}} \ge 0.8\) (или другой фиксированный порог),
- `FTL_ratio_p95 ≥ Γ_FTL`,
- и `mean(d_eff) < mean(d_base)` на активных парах.

(Пороговые числа вы задаёте в `target_spec.thresholds`, фиксируя их заранее.)

---

## 3.4. H2: “Эффект контролируем параметрами (threshold/capacity/min_distance)” + тест монотонности
Делаем sweep и оцениваем монотонность без подбора “удобного” параметра.

### Контролы и ожидаемые направления
Из вашей логики в [experiment_C_ftl_control.pdf](/files/zHik4bvMBQlABEhysudMQ):
- увеличение `resonance_threshold` ⇒ **уменьшает** `n_wormholes`, `frac_shortened`
- увеличение `capacity` ⇒ **увеличивает** `n_wormholes`, `frac_shortened`, `mean_ftl_ratio` (до насыщения)
- увеличение `min_hop_distance` ⇒ **уменьшает** `n_wormholes` (если пар мало) и/или сдвигает активации в дальние пары (можно тестировать через `mean(d_base)` активных)

### Статистический тест (Spearman)
Для каждой пары (“контрол”, “ответная метрика”) вычисляем Spearman \(\rho\) по строкам `control_sweep` при фиксированных прочих контролах или на всей сетке (лучше — частный вариант: группировать по двум контролам, считать \(\rho\) по третьему, затем агрегировать медианой).

Минимальный, но строгий вариант “в один проход”:
- считать Spearman \(\rho\) по всей таблице и требовать знак + величину.

Пример гейта:
- \(\rho(\text{resonance_threshold}, n_{wh}) \le -0.7\) и p-value < 0.01
- \(\rho(\text{capacity}, n_{wh}) \ge +0.7\) и p-value < 0.01
- \(\rho(\text{capacity}, \text{mean_ftl_ratio}) \ge +0.5\) и p-value < 0.01

**H2 подтверждена**, если все требуемые монотонности проходят.

---

## 3.5. H3: baseline compatibility (context=0 / H(t)=∅) — привести к Tests 4–6
То, что у вас сейчас в [experiment_C_ftl_control.pdf](/files/zHik4bvMBQlABEhysudMQ) (4.1–4.4) — хорошая “smoke‑проверка”. Но внешний стандарт по [ftl_physics(3).pdf](/files/RiFu_bq20Eu_J2UvoOaVf) требует:

- **Test 4 (каузальность)**: формальная проверка зависимости обновлений (юнит‑тест + PASS в JSON)
- **Test 5 (детерминизм)**: два прогона → одинаковый state hash и iface trace hash
- **Test 6 (Q conservation)**: \(Q(t)=Q(0)\) для всех t (или violations=0)

Поэтому **H3** в отчёте лучше определять как:
> H3 PASSED iff Test4=PASS and Test5=PASS and Test6=PASS and also “no-wormhole distance equality checks” passed.

---

## 3.6. H4: “Бимодальность” распределения сокращений (tail events)
Ваша идея из Experiments_v1: редкие резкие события (tail events). Это можно формализовать на \(\Delta d\) или на \(R\).

### Два взаимодополняющих теста (рекомендую “both”)
1) **Hartigan’s dip test** на выборке \(\{\Delta d^{(i)}: i\in\mathcal{A}\}\) (или на \(\log R\))  
Гейт: p-value < 0.01 ⇒ отвергаем “одномодальность”.

2) **GMM 1 vs 2 компоненты** (на \(\Delta d\) или \(\log R\)), сравнение BIC:
- обучаем 1‑Gaussian и 2‑Gaussian смеси
- считаем \(\Delta \mathrm{BIC} = \mathrm{BIC}_1 - \mathrm{BIC}_2\)
Гейт: \(\Delta \mathrm{BIC} > 10\) (сильное свидетельство в пользу 2 компонент)

**H4 PASSED**, если выполняется хотя бы одно (или оба — вы фиксируете в thresholds).

---

# 4) Recompute contracts (как в A/B) — для CI
В `target_spec.recompute_contracts` зафиксируйте ключевые метрики, например:

1) `frac_pairs_shortened`:
- table: `tables/pairs_eval.parquet`
- formula: `mean(delta_d > 0 for active==True & min_hop_distance_ok==True)`

2) `FTL_ratio_p95`:
- formula: `quantile(ftl_ratio, 0.95) over same filter`

3) H2 Spearman:
- table: `tables/control_sweep.parquet`
- formula: `spearmanr(resonance_threshold, n_wormholes).rho` etc.

4) H4:
- table: `tables/pairs_eval.parquet`
- formula: dip test p-value and/or ΔBIC.

5) Test 6:
- table: `tables/Q_trace.csv`
- formula: `max(Q)-min(Q)==0`

---

# 5) Минимальный TargetSpec для HIT (уровень B)
Пример gating:

- context=0 (если вы его тоже прогоняете в C‑pipeline):
  - Test4 ✅, Test5 ✅, Test6 ✅

- context=1:
  - Test4 ✅, Test5 ✅, Test6 ✅
  - H1 ✅ (ftl_ratio_p95 ≥ Γ_FTL и frac_shortened ≥ f_min)
  - H2 ✅ (монотонности)
  - H4 ✅ (бимодальность)

`score` можно держать вторичным; внешний аудит обычно проверяет HIT и таблицы.

---

Если хотите, могу дополнить “патч” конкретными сигнатурами функций экспорта (как вы делали для B: `export_pairs_eval`, `export_control_sweep`, `export_Q_trace`, `build_manifest`) и примером pytest‑тестов для Test 4/5/6, чтобы всё автоматически проходило в CI.
