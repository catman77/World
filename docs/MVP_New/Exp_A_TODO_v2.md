Ниже — разбор нового варианта ноутбука **[experiment_A_sparc(1).pdf](/files/CZS6TdQK1oL4xjVLrarQG)** (вы пометили “нормализация”) как результата Эксперимента A. Я разделю анализ на: (i) что именно добавилось/изменилось по сравнению с предыдущей версией, (ii) что означает “мост RSL↔MOND” в строгих терминах, (iii) качество по вашим TargetSpec‑метрикам и что они реально гарантируют, (iv) критические места (где результат можно “сломать” и что стоит усилить), (v) что нужно для репо‑автоматизации.

---

## 1) Что нового в версии “нормализация” (по содержимому pdf)

### 1.1. Явно оформлен “мост” между графовой частью и MOND‑феноменологией
Ключевой блок в начале/сводке:

- **RSL граф:** power‑law рёбра, \(\alpha=2.0\)  
- **эффективная размерность:** \(D_{\mathrm{eff}}(r)\approx 1.5\)–\(1.8\)  
- **поправка:** \(\delta(r)=g_{\mathrm{eff}}/g_N - 1\)

и параллельно:

- **MOND phenom:** \(\mu(x)=1-\exp(-\sqrt{x})\)  
- **масштаб:** \(a_0\approx 1.0\times 10^{-10}\ \mathrm{m/s^2}\)

и самое важное равенство‑мэппинг:

\[
\mu \;\;\Longleftrightarrow\;\; \frac{1}{1+\delta}.
\]

Это — уже не просто “мы подогнали μ(x) к RAR”, а заявка, что **δ(r), полученная из графа**, задаёт μ(x) почти без свободы. Это качественно правильный шаг по логике Experiments_v1: “graph‑origin justification”.

### 1.2. Появился формальный “TargetSpec (MVP)” со статусом HIT
В сводной таблице/блоке:

- \(f_{\mathrm{good}}(\chi^2<5) \ge 50\%\) → ~54% ✅  
- \(\sigma_{\mathrm{RAR}} \le 0.20\ \mathrm{dex}\) → 0.183 dex ✅  
- \(E_\mu(\text{shape}) \le 0.10\) → ~0.05 ✅  
- HIT = ✅

Это важно: теперь результат выражен в **машинно проверяемых критериях**, а не в “картинках”.

### 1.3. Улучшена “нормализация” и отчётность (JSON report)
В конце видно, что вы:
- собираете `EXPERIMENT_A_REPORT` со структурой `metadata / rsl_parameters / sparc_data / calibration / phenomenology_results ... / target_spec ...`,
- печатаете финальный блок с HIT/SCORE,
- сохраняете JSON с конвертацией numpy‑типов.

Это практически уже “репо‑готовый” артефакт: можно подцепить CI/регрессии.

---

## 2) “МОСТ МЕЖДУ RSL И MOND”: что строго следует из μ = 1/(1+δ), и что ещё надо доказать

### 2.1. Связь между δ и μ корректна как алгебра (но важен выбор определения δ)
Вы используете:

\[
\delta(r)=\frac{g_{\mathrm{eff}}(r)}{g_N(r)}-1
\quad\Rightarrow\quad
g_{\mathrm{eff}}(r)=g_N(r)\,(1+\delta(r)).
\]

В MOND‑языке обычно пишут:
\[
g_{\mathrm{obs}} = \frac{g_{\mathrm{bar}}}{\mu(x)},\quad x=\frac{g_{\mathrm{bar}}}{a_0}.
\]

Если вы отождествляете \(g_N \equiv g_{\mathrm{bar}}\) (ньютоновское барионное ускорение) и \(g_{\mathrm{eff}}\equiv g_{\mathrm{obs}}\), то:

\[
\frac{g_{\mathrm{obs}}}{g_{\mathrm{bar}}}=1+\delta
\quad\Rightarrow\quad
\mu = \frac{g_{\mathrm{bar}}}{g_{\mathrm{obs}}}=\frac{1}{1+\delta}.
\]

То есть равенство \(\mu=1/(1+\delta)\) — логически безупречно **при одном условии**: что ваше \(g_N\) действительно соответствует \(g_{\mathrm{bar}}\) (ньютоновскому барионному) и вычисляется тем же способом/в тех же единицах, что и \(g_{\mathrm{eff}}\).

### 2.2. Где “мост” пока может быть слишком свободным
Чтобы это не превратилось в переобозначение (“назвали δ тем, что нужно для μ”), критично показать:

1) **Как именно вычисляется \(g_{\mathrm{eff}}(r)\) из графа** (и не зависит ли это от нормализаций/сглаживания — см. мой прошлый ответ про поток через \(\partial B_r\)).  
2) **Как задаётся \(g_N(r)\)**: это “идеальный \(1/r^2\)” в hops? или это решение на “гладком” графе? или это “барионное ньютоновское” из SPARC?  
3) Как вы сопоставляете \(r\) на графе и \(r\) в kpc (калибровка). В новой версии вы явно указываете “нормализация”; хочется видеть, что нормировка одна и не подгоняется индивидуально.

Если эти три пункта зафиксированы, тогда δ становится настоящим выходом симулятора, а μ — его производной.

---

## 3) Феноменологическая часть: что показывают цифры и графики

### 3.1. Лучшее μ(x): \(1-e^{-\sqrt{x}}\) и \(a_0 \approx 1.05\times10^{-10}\)
В таблице сравнения:

- **MOND: \(\mu(x)=1-e^{-\sqrt{x}}\)**  
  \(a_0 = 1.05\times10^{-10}\ \mathrm{m/s^2}\), \(\chi^2_{\rm red}\approx 43.5\) (в таблице помечено как “лучшая”)  
- “RSL simple” \(\mu=x/(1+x)\): \(\chi^2_{\rm red}\approx 92.6\) — плохо  
- “standard” \(\mu=x/\sqrt{1+x^2}\): \(\chi^2_{\rm red}\approx 139.9\) — плохо

Тут важный момент: у вас \(\chi^2_{\rm red}\) огромные (43–140). Это может быть:
- потому что вы считаете \(\chi^2\) в RAR‑пространстве с очень маленькими ошибками (тогда \(\chi^2_{\rm red}\) почти всегда большой),
- или потому что берёте ошибки не так, как в публикации,
- или потому что не оптимизируете nuisance‑параметры (наклон \(i\), distance, \(\Upsilon_\*\)), которые SPARC‑подгонки обычно учитывают.

Но при этом вы опираетесь на другие метрики (scatter 0.183 dex, доля good fits), что может быть адекватнее для MVP. Просто надо честно зафиксировать: **\(\chi^2_{\rm red}\) в абсолюте не сопоставим с “хорошо/плохо” без определения noise model**.

### 3.2. Scatter в RAR: 0.183 dex
Вы выводите:

- scatter \(= \mathrm{std}(\log_{10}(g_{\mathrm{obs}}/g_{\mathrm{model}})) = 0.183\ \mathrm{dex}\)
- сравнение с публикацией: ~0.13 dex

Интерпретация:
- 0.183 dex — это уже “в пределах” вашего TargetSpec (\(\le 0.20\)) и достаточно неплохо для простого протокола.
- Но вы явно хуже, чем “идеальная” обработка SPARC (0.13 dex). Это нормально для MVP, если вы **не подгоняете** distance/inclination/M/L.

Рекомендация для усиления результата без “скрытого фита”: явно зафиксировать, какие nuisance‑параметры вы *не трогаете* и почему; а также посчитать scatter отдельно:
- на high‑quality подвыборке SPARC,
- с/без экстремальных галактик,
- с робастной оценкой (MAD вместо std).

### 3.3. Доля хороших галактик \(f_{\mathrm{good}}\approx 54\%\)
Вы вводите критерий “\(\chi^2<5\)” для галактик и получаете ~54%.

Это хороший “продуктовый” показатель, но важно:
- какой именно \(\chi^2\) (по \(v(r)\) или по \(g\))?
- учитываются ли систематики (наклон, расстояние)?
- одинаково ли нормированы ошибки для всех?

Сейчас в pdf видно, что вы строите для каждой галактики графики:
- \(v_{\rm obs}\) с ошибками,
- \(v_{\rm bar}\),
- \(v_{\rm fit}\).

Значит, \(\chi^2\) вероятно считается по скорости, что более стандартно. Тогда нужно документировать:  
\[
\chi^2 = \sum_i \left(\frac{v_{\rm obs}(r_i)-v_{\rm fit}(r_i)}{\sigma_v(r_i)}\right)^2,
\quad \chi^2_{\rm red}=\chi^2/(n-k).
\]

---

## 4) TargetSpec (MVP): сильные стороны и что бы я уточнил

### 4.1. Сильные стороны
- Есть **три независимых критерия**: качество по галактикам (f_good), глобальная RAR‑сходимость (σ_RAR), и “shape” \(\mu\) (E_μ).
- HIT = AND по критериям — правильно для “не допустить победу одной метрикой”.

### 4.2. Что нужно уточнить, чтобы TargetSpec было “необманываемым”
1) **Определение E_μ (shape)**  
Сейчас есть число ~0.05, порог 0.10. Но в pdf не видно строгого определения:  
- это L2‑ошибка между извлечённой \(\mu(x)\) и целевой \(1-e^{-\sqrt{x}}\) на лог‑сетка по x?  
- или K‑L divergence по распределению?  
- или фит параметризации?

Без этого E_μ легко “улучшить” просто сменив сетку/веса.

2) **Стабильность по нормализации/калибровке**  
Раз вы ввели “нормализацию”, вам нужно показать, что:
- при небольшом изменении калибровки единиц (±5–10%) метрики не рушатся,
- и главное: что калибровка одна и применяется ко всем галактикам.

3) **Связь D_eff(r) → δ(r) → μ(x)**  
Сейчас в сводке написано D_eff≈1.5–1.8. Хорошо бы в отчёт добавлять:
- график \(A(r)\) или \(N(r)\) (рост числа вершин/границы шара),
- оценку локального наклона (локальная размерность),
- и корреляцию “где меняется D_eff — там меняется δ”.

Иначе D_eff выглядит как декоративная вставка.

---

## 5) Главные риски/вопросы к интерпретации (не “хейтерство”, а что проверять дальше)

1) **Почему лучшая μ(x) получается MOND‑овская уже на уровне феноменологического фита?**  
Это ожидаемо (RAR известен), но важно: ваш “мост” ценен только если вы покажете, что **та же** форма получается из графа без ручного выбора функционального вида.

2) **\(\chi^2_{\rm red}\) очень большой**  
Это не обязательно проблема, но требует пояснения noise model. Я бы в отчёт добавил:
- “мы не оптимизируем distance/inclination/M/L, поэтому χ²_red не интерпретируем буквально; используем scatter+f_good”.

3) **Систематический сдвиг остатков**  
Вы печатаете mean residual ~0.0618 (положительный). Это намекает на небольшое систематическое недо/пере‑предсказание. Хорошо бы посмотреть:
- остатки как функция \(g_{\rm bar}\) (есть ли “кривая”),
- остатки как функция радиуса,
- отдельные классы галактик (LSB/HSB).

---

## 6) Что я бы добавил в следующий прогон (в рамках того же ноутбука), чтобы результат стал “железным”
Минимальные усиления без усложнения физики:

1) **Явное определение E_μ и сохранение кривой \(\mu_{\rm extracted}(x)\) в JSON**  
Чтобы можно было регрессионно тестировать форму.

2) **Bootstrap по галактикам для σ_RAR**  
Оценка неопределённости: “σ_RAR = 0.183 ± 0.0X dex”.

3) **Стабильность калибровки**  
Отчёт: “при \(\kappa\to 1.05\kappa\) HIT сохраняется / разрушается”.

4) **Связь с графом**  
Сохранить в отчёт минимальные графовые профили: `A(r)`, `D_eff(r)`, `delta(r)` и окно, где выполняется \(1/r^2\).

---

## 7) Итог по файлу
Файл **[experiment_A_sparc(1).pdf](/files/CZS6TdQK1oL4xjVLrarQG)** — это качественный шаг вперёд относительно “просто фита MOND”:

- появился формализованный **TargetSpec (MVP)** с HIT ✅,
- заявлена и частично реализована связь **\(\delta \leftrightarrow \mu\)**,
- фиксируется “лучшая форма μ” (та, что действительно хорошо работает на RAR),
- и есть задел на автоматизацию отчёта (JSON).

Чтобы это стало “graph-origin justification” в строгом смысле Experiments_v1, критически важно в следующей итерации сделать прозрачными и воспроизводимыми:
- определение \(g_{\mathrm{eff}}\) на графе,
- определение \(g_N\) и калибровку единиц,
- формулу \(E_\mu\),
- и связку \(D_{\mathrm{eff}}(r)\Rightarrow \delta(r)\) (не просто рядом в сводке).

Если хотите, могу на основе структуры `EXPERIMENT_A_REPORT` предложить точную JSON-схему (Pydantic) и список артефактов, которые репорт должен сохранять (CSV/PNG/NPZ) для полного воспроизведения HIT в CI.

Ниже — (1) **точная JSON‑схема** для `EXPERIMENT_A_REPORT` в виде Pydantic‑моделей и (2) **минимальный набор артефактов** (CSV/PNG/NPZ + manifest) для *полного воспроизведения HIT в CI*.

Ссылаюсь на ваши файлы: постановка эксперимента и требование “TargetSpec/метрики/репо/тесты воспроизводимости” идут из [Experiments_v1.pdf](/files/2OSgJNaaGTQaJ5a6CTSXL); текущая структура отчёта и метрики (`f_good`, `σ_RAR`, `E_μ`, `κ`, `a0_collapse`) видны в [experiment_A_sparc(1).pdf](/files/CZS6TdQK1oL4xjVLrarQG).

---

# 1) JSON‑схема отчёта (Pydantic)

## 1.1. Дизайн‑принципы схемы (чтобы CI мог воспроизводить HIT)
1) **Версионирование**: `schema_version`, `code_version` (git commit), `data_version` (SPARC checksum), `world_hash`.
2) **Детерминизм**: фиксированные `seeds`/`rng`.
3) **Восстановимость**: все скаляры TargetSpec + ссылки на артефакты, из которых их можно пересчитать.
4) **Разделение слоёв**:
   - `phenomenology_results` (RAR/rotation fit),
   - `graph_origin` (δ, μ из δ, D_eff, kappa),
   - `target_spec` (hit/score + пороги + детали).

## 1.2. Pydantic модели (v2)
```python
from __future__ import annotations
from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, HttpUrl, conlist

# -------------------------
# Small utility models
# -------------------------

class VersionInfo(BaseModel):
    schema_version: str = Field(..., description="Report schema version, e.g. 'A-1.0.0'")
    report_version: str = Field(..., description="Notebook/report version, e.g. '2.0'")
    code_commit: str = Field(..., description="Git commit hash of code that produced the report")
    code_dirty: bool = Field(..., description="Whether repo had uncommitted changes")
    python: str
    numpy: str
    scipy: Optional[str] = None
    platform: Optional[str] = None


class DataProvenance(BaseModel):
    source_name: str = Field(..., description="e.g. 'SPARC Database (Lelli+2016)'")
    source_url: Optional[HttpUrl] = None
    n_galaxies: int
    n_rar_points: int
    sparc_checksum: str = Field(..., description="Hash of raw SPARC bundle or canonical processed parquet")
    preprocessing_checksum: Optional[str] = Field(None, description="Hash of preprocessing pipeline config/artifact")


class WorldParams(BaseModel):
    N: int
    alpha: float
    L: int
    n_edges: Optional[int] = None
    avg_degree: Optional[float] = None
    distance_metric: Literal["hops"] = "hops"
    laplacian_type: Optional[Literal["combinatorial", "normalized"]] = None
    solver: Optional[str] = None
    world_hash: str = Field(..., description="Hash of (ruleset, graph generator params, etc.)")


class Calibration(BaseModel):
    # hops -> kpc
    kappa_kpc_per_hop: float = Field(..., gt=0)
    # potential/gradient -> physical acceleration scale (optional but recommended)
    gamma_accel_scale: Optional[float] = Field(None, description="Global accel scaling, if used")
    method: Literal["single_galaxy_a0_crossing", "single_galaxy_newton_window"] = "single_galaxy_a0_crossing"
    calib_galaxy_id: str
    calib_details: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Phenomenology results
# -------------------------

class MuModelSpec(BaseModel):
    name: str = Field(..., description="e.g. 'MOND_like'")
    formula: str = Field(..., description="Human-readable formula string")
    a0_best_mps2: float = Field(..., gt=0)
    chi2_red_global: Optional[float] = None


class RotationSummary(BaseModel):
    chi2_red_median: Optional[float] = None
    f_good: float = Field(..., ge=0, le=1)
    threshold_chi2_red: float = Field(..., gt=0)
    n_good: int
    n_total: int


class RARSummary(BaseModel):
    scatter_dex: float = Field(..., ge=0)
    residual_mean: float
    residual_median: float
    residual_std: float
    residual_iqr: float


class A0Universality(BaseModel):
    # Only if per-galaxy a0 estimated
    median_a0_mps2: Optional[float] = None
    sigma_log10_a0: Optional[float] = None
    n_used: Optional[int] = None
    selection: Optional[str] = None


class PhenomenologyResults(BaseModel):
    best_mu: MuModelSpec
    rotation: RotationSummary
    rar: RARSummary
    a0_universality: Optional[A0Universality] = None


# -------------------------
# Graph-origin results
# -------------------------

class GraphOriginSummary(BaseModel):
    g_eff_method: Literal["flux_shell", "dirichlet_shell"] = "flux_shell"
    delta_definition: Literal["g_eff_over_gN_minus_1"] = "g_eff_over_gN_minus_1"
    mu_from_delta_definition: Literal["mu_equals_1_over_1_plus_delta"] = "mu_equals_1_over_1_plus_delta"

    # Effective dimension estimates (optional but good to persist)
    D_eff_min: Optional[float] = None
    D_eff_max: Optional[float] = None
    D_eff_notes: Optional[str] = None

    # Shape match metric
    mu_template_formula: str = Field(..., description="e.g. '1 - exp(-sqrt(x))'")
    E_mu: float = Field(..., ge=0)

    # Collapse-based a0 estimate (if computed)
    a0_collapse_mps2: Optional[float] = None
    collapse_objective: Optional[str] = None
    collapse_score: Optional[float] = None


# -------------------------
# TargetSpec results
# -------------------------

class TargetThresholds(BaseModel):
    f_good_min: float = Field(..., ge=0, le=1)
    chi2_red_max: float = Field(..., gt=0)
    scatter_dex_max: float = Field(..., gt=0)
    E_mu_max: float = Field(..., gt=0)
    sigma_log10_a0_max: Optional[float] = Field(None, gt=0)


class HitDetails(BaseModel):
    hit_rotation: bool
    hit_rar: bool
    hit_mu: bool
    hit_a0: Optional[bool] = None


class ScoreDetails(BaseModel):
    s_rot: float = Field(..., ge=0, le=1)
    s_rar: float = Field(..., ge=0, le=1)
    s_mu: float = Field(..., ge=0, le=1)
    s_a0: Optional[float] = Field(None, ge=0, le=1)


class TargetSpecReport(BaseModel):
    hit: bool
    score: float = Field(..., ge=0, le=1)
    thresholds: TargetThresholds
    hit_details: HitDetails
    score_details: ScoreDetails


# -------------------------
# Artifact manifest
# -------------------------

class ArtifactRef(BaseModel):
    path: str = Field(..., description="Relative path within run directory")
    sha256: str
    kind: Literal["json", "csv", "npz", "png", "md", "parquet"]


class ArtifactManifest(BaseModel):
    run_dir: str
    artifacts: List[ArtifactRef]


# -------------------------
# Top-level report
# -------------------------

class ExperimentAReport(BaseModel):
    experiment: Literal["A"] = "A"
    name: str = "SPARC fit + Graph-origin justification"
    created_at: str

    versions: VersionInfo
    data: DataProvenance
    world: WorldParams
    calibration: Calibration

    phenomenology: PhenomenologyResults
    graph_origin: GraphOriginSummary
    target_spec: TargetSpecReport

    manifest: ArtifactManifest
```

### 1.3. Что это “гарантирует” для CI
- CI может:
  1) скачать/найти `run_dir`,
  2) проверить sha256 артефактов,
  3) пересчитать ключевые метрики из NPZ/CSV,
  4) убедиться, что `target_spec.hit` совпадает и что пороги те же.

---

# 2) Список артефактов для полного воспроизведения HIT в CI

Цель: **воспроизвести HIT без ручных шагов** и без необходимости перегенерировать “мир” с нуля, но при этом иметь возможность *пересчитать* метрики.

Ниже — минимальный набор, который закрывает три критерия TargetSpec из [experiment_A_sparc(1).pdf](/files/CZS6TdQK1oL4xjVLrarQG): `f_good`, `σ_RAR`, `E_μ` (+ опционально `a0_collapse`).

## 2.1. Обязательные артефакты (must-have)

### (1) `report.json`
- Полный JSON по схеме выше.
- Содержит thresholds, hit_details, ссылки на файлы.
- CI использует его как “истину версии”.

**Path:** `runs/expA_<timestamp>/report.json`  
**Kind:** `json`

---

### (2) `sparc_points.parquet` или `sparc_points.csv`
Таблица на **уровне точек RAR** (3367 строк), минимум:

- `galaxy_id`
- `r_kpc`
- `v_obs_kms`, `v_err_kms`
- `g_obs_mps2` (можно пересчитать, но лучше сохранить)
- `g_bar_mps2`
- `quality_flags` (если применяете cuts)

Это позволяет CI повторить RAR scatter без графа.

**Path:** `data/sparc_points.parquet` (рекомендую parquet)  
**Kind:** `parquet` (или `csv`)

---

### (3) `pred_points.parquet` (или CSV) — предсказания на тех же точках
Таблица с тем же индексом точек (совпадение по `galaxy_id`+`r_kpc` или через `point_id`):

- `point_id` (лучше явный)
- `galaxy_id`, `r_kpc`
- `v_pred_kms`
- `g_pred_mps2`
- `residual_log10` = `log10(g_obs/g_pred)` (можно пересчитать, но удобно хранить)
- (опционально) `mu_pred`, `delta_pred`

**Зачем:** CI пересчитывает:
- scatter dex,
- распределение остатков,
- любые графики/таблицы.

**Kind:** `parquet`/`csv`.

---

### (4) `galaxy_fit_summary.csv`
На **уровне галактик** (171 строк), минимум:

- `galaxy_id`
- `n_points`
- `chi2`
- `chi2_red`
- `dof`
- `is_good` (chi2_red < threshold)
- (опционально) `a0_fit_mps2` если вы фитите a0 по галактикам

**Зачем:** CI пересчитывает `f_good` и подтверждает соответствие порога.

---

### (5) `mu_extraction.npz`
NPZ с данными, позволяющими пересчитать `E_mu` (shape match) *в точности*:

Обязательные массивы:
- `x_grid` (например лог‑сетка)
- `mu_template` на этой сетке
- `mu_extracted` на той же сетке (после всех ваших процедур сглаживания/бинирования)
- `weights` (если используете взвешенную норму)
- `E_mu` (как число, для удобства)

Также полезно:
- `raw_x_points`, `raw_mu_points` (до бининга) — чтобы CI мог проверить, что бининг не менялся.

**Почему NPZ:** это фиксирует ваш алгоритм вычисления E_mu как “данные + формула”, а не как картинка.

---

### (6) `graph_origin_profiles.npz`
Чтобы “graph-origin justification” был воспроизводим, даже если CI не генерит граф заново:

- `r_hops` (1..Rmax)
- `g_eff_hops_units` (как вы считаете на графе; например flux-shell)
- `gN_hops_units` (baseline 1/r^2 в hops или другой ваш baseline)
- `delta_r` (проверка консистентности)
- `A_r` (граница/“площадь” между слоями) — чтобы подтверждать D_eff
- (опционально) `D_eff_r` если считаете

Это позволяет CI:
- пересчитать delta,
- пересчитать mu=1/(1+delta),
- (минимально) проверить, что “потоковый g_eff без биннинга” действительно используется.

---

### (7) `manifest.json` (sha256 всех файлов)
CI:
- проверяет целостность,
- не допускает “тихой подмены” данных.

---

## 2.2. Рекомендуемые артефакты (should-have)

### (8) PNG‑фигуры (чтобы отчёт был читабелен человеком, но не обязателен для CI)
Минимум:
- `fig_rotation_showcase.png` (несколько ключевых галактик)
- `fig_rar_scatter.png` (RAR scatter plot + binned curve)
- `fig_mu_shape.png` (mu_extracted vs mu_template)
- `fig_graph_origin_profiles.png` (g_eff, delta, A(r), D_eff)

CI может лишь проверять, что файлы существуют и их sha256 соответствует `manifest`, но не обязан “понимать” картинку.

---

### (9) `config.yaml`
Конфиг прогона:
- параметры мира (`N, alpha, L, solver`),
- параметры selection/cuts,
- thresholds TargetSpec,
- калибровка (какая галактика якорь, как определён a0 crossing).

Это даёт воспроизводимость “с нуля” вне CI.

---

## 2.3. Минимальный CI‑скрипт проверки HIT (логика)
В CI вы делаете:

1) Загружаете `report.json`, проверяете `schema_version`.
2) Верифицируете `manifest.json` (sha256).
3) Пересчитываете:
   - `f_good` из `galaxy_fit_summary.csv`,
   - `scatter_dex` из `pred_points.parquet` + `sparc_points.parquet`,
   - `E_mu` из `mu_extraction.npz` по формуле (и сверяете с числом).
4) Сравниваете с `target_spec.thresholds` и утверждаете `hit`.

---

# 3) Конкретные имена путей (конвенция run directory)

Рекомендую фиксировать run‑директорию как:

`runs/expA_YYYYMMDD_HHMMSS_<code_commit_short>/`

Внутри:
```
report.json
manifest.json
config.yaml

tables/
  sparc_points.parquet
  pred_points.parquet
  galaxy_fit_summary.csv

npz/
  mu_extraction.npz
  graph_origin_profiles.npz

figures/
  fig_rotation_showcase.png
  fig_rar_scatter.png
  fig_mu_shape.png
  fig_graph_origin_profiles.png
```

---

# 4) Небольшие уточнения под ваш текущий ноутбук
Из [experiment_A_sparc(1).pdf](/files/CZS6TdQK1oL4xjVLrarQG) видно, что у вас уже есть:
- структура `EXPERIMENT_A_REPORT[...]`,
- печать порогов,
- сохранение JSON с конвертацией numpy типов,
- метрики: `f_good`, `σ_RAR`, `E_μ`, `κ`, `a0_collapse`.

Значит, вам осталось “довести до CI‑уровня” две вещи:
1) стандартизировать вычисление `E_μ` так, чтобы оно зависело от сохранённого `x_grid/weights`, а не от внутренних временных массивов ноутбука;
2) сохранять **таблицы предсказаний** на уровне точек/галактик, чтобы CI мог пересчитать `f_good` и scatter независимо от кода.

Если скажете, в каком формате сейчас у вас лежит SPARC (csv файловый набор / уже pandas DataFrame), я предложу конкретные функции `export_tables(...)` и `build_manifest(...)` с аккуратной привязкой по `point_id`.
