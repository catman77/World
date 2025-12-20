Чтобы **эксперимент B** мог пройти внешнюю (независимую) проверку, в текущем виде **[experiment_B_coherence.pdf](/files/JjEjnblu4BCUt6xzHBQwq)** ему не хватает нескольких ключевых элементов строгости. Это не “плохо сделано” — скорее, сейчас это уровень *внутреннего отчёта*, а не *реплицируемого вычислительного эксперимента*. Ниже — что именно является критическими слабостями (мешает независимой проверке), и минимальный план доведения до стандарта, аналогичного вашему Эксперименту A.

---

## 1) Критические проблемы строгости в текущем виде

### 1.1. Невоспроизводимость по артефактам: есть только JSON-репорт, но нет данных для пересчёта метрик
В ноутбуке формируется `experiment_B_report.json`, но **нет сохранённых таблиц/NPZ**, из которых внешний проверяющий мог бы пересчитать:
- матрицу \(V(N,\eta)\),
- порог \(N_{1/2}\) (где \(V<0.5\)),
- проверку H1/H2/H3,
- (если вы утверждаете) дополнительность \(V^2 + D^2 \le 1\).

Сейчас внешний проверяющий вынужден “верить числам” в JSON, а не подтверждать их независимым пересчётом.

**Это главный барьер для внешней проверки.**

---

### 1.2. \(N^\*\) одновременно фигурирует как параметр и как результат (логическая уязвимость)
В коде и отчёте:
- `parameters.N_star = N_star`
- `results.phase_saturation_threshold = N_star`

и далее формулировки в выводах: “подтверждено, критический масштаб \(N^\* = 512\)”.

Если \(N^\*\) задан заранее (как часть модели), то эксперимент не “обнаруживает” порог, а лишь демонстрирует работу механизма, зависящего от встроенного параметра. Это допустимо, но тогда:
- нельзя писать “порог найден”, нужно писать “при заданном \(N^\*\) наблюдается ожидаемое поведение”;
- а для строгого теста надо отдельно показать оценку \(\hat N^\*\) из данных (без использования заданного значения), и сравнить \(\hat N^\*\) с 512.

Сейчас формально остаётся возможность, что код содержит конструкцию вида “если N>N_star → подавить фазу”, и H1 будет подтверждаться по определению.

---

### 1.3. Не зафиксировано строгое определение \(V\) и (особенно) \(D\)
Вы утверждаете в интерпретации:
- “Π_meas извлекает which-path информацию”
- “\(V^2 + D^2 \le 1\) — дополнительность выполняется”

Но в видимой части кода:
- вычисление \(V\) как видимости (через \(I_{\max}, I_{\min}\)) не показано,
- вычисление \(D\) вообще не показано и не сохраняется в результатах.

Для внешней проверки это критично: проверяющий должен видеть точные формулы и уметь пересчитать их из сырых данных (или хотя бы из агрегатов на сетке \(N,\eta\)).

---

### 1.4. Слабая статистическая строгость: нет seeds, числа прогонов, CI/bootstraps
В отчёте есть mean/std, но не зафиксированы:
- число прогонов на точку \((N,\eta)\),
- используемые seeds,
- доверительные интервалы для ключевых метрик (H1 ratio, H2 slope/monotonicity).

Это делает результат уязвимым к обвинению “один удачный прогон”.

---

### 1.5. H2/H3 сформулированы слишком “мягко” и допускают ложноположительное подтверждение
Сейчас H2 фактически:
- `mean(V|eta=1) < mean(V|eta=0)` (при непонятном условии `if N ...`, обрезано)

H3:
- `std(V(eta=0)) < 0.1` и `mean(V(eta=1, N>N_star)) < 0.1`

Такие критерии не защищают от:
- немонотонности,
- “эффекта в одной точке”,
- выбора удобного `eta_test`,
- и не дают статистического подтверждения (p-value/CI).

---

## 2) Что в B уже сделано хорошо (и можно сохранить)
- Есть операциональные гипотезы H1–H3 и их машинный вывод (`confirmed`).
- Есть полный вывод матрицы \(V(N,\eta)\) (как вложенный dict), что уже приближает к контракту данных.
- Есть интерпретационный слой (hierarchy, which-path, complementarity) — это можно оставить, но нужно отделить “утверждения” от “измерено/проверено”.

---

## 3) Минимальный патч, чтобы B стал внешне проверяемым (как A)

Ниже — минимум, который превращает эксперимент в “реплицируемый пакет”:

### 3.1. Артефакты (2 таблицы + 1 NPZ + manifest)
Сохраните рядом с JSON:

1) **`tables/V_grid.parquet`** — *сырьё* по прогонам  
Каждая строка = один прогон на одной точке сетки.
Колонки:
- `N` (int)
- `eta` (float)
- `run_id` (int)
- `seed` (int)
- `V` (float)
- `f_sat` (float, если есть по прогону; если только по N, то отдельно)
- (опционально) `D` (float), если вы реально проверяете дополнительность.

2) **`tables/V_summary.parquet`** — агрегаты  
Для каждого (N, eta):
- `V_mean`, `V_std`, `V_ci_low`, `V_ci_high`, `n_runs`
- аналогично для `D`, если применимо.

3) **`npz/threshold_fit.npz`** (опционально, но полезно)  
Если вы делаете оценку \(\hat N^\*\) логистическим/erf‑фитом:
- массивы `N_values`, `V_eta06` (или другой выбранный eta_test),
- параметры фита `N_star_hat`, `width`, `Vmin`, `Vmax`,
- качество фита.

4) **`manifest.json`**  
sha256 всех файлов: `report.json`, таблиц и npz.

Это ровно тот уровень, который вы сделали в A, и он снимает 80% вопросов.

---

### 3.2. JSON: контракт пересчёта (как в A)
Добавьте в `report`:

- `versions`: git commit, python/numpy версии, platform
- `rng`: seeds, n_runs_per_point, детерминизм
- `target_spec.recompute_contract`:

Пример:

```json
"target_spec": {
  "metrics_used_for_hit": ["H1_ratio", "H2_monotonic", "H3_effect_size"],
  "recompute_contract": {
    "H1_ratio": {
      "table": "tables/V_summary.parquet",
      "formula": "mean(V|N>N_star, eta=0.6) / mean(V|N<=N_star, eta=0.6)",
      "params": {"eta_test": 0.6}
    },
    "H2_monotonic": {
      "table": "tables/V_summary.parquet",
      "formula": "spearman_corr(eta, mean_V_over_N_above_threshold) < -0.8"
    },
    "H3_effect_size": {
      "table": "tables/V_grid.parquet",
      "formula": "cohens_d(V|eta=0, N>N_star vs V|eta=1, N>N_star) > 2.0"
    }
  }
}
```

---

### 3.3. Отделить “параметр модели” \(N^\*\) от “оценки из данных” \(\hat N^\*\)
В отчёте должно быть два поля:

- `parameters.N_star_model = 512` (если это реально параметр вашей RSL-иерархии)
- `results.N_star_hat_from_data = ...` (оценка по данным)

И тест:
- `abs(N_star_hat_from_data - N_star_model) <= tolerance` (например 1–2 шага сетки N).

Это ключевой пункт, снимающий обвинение “вы заложили порог”.

---

### 3.4. Дополнительность \(V^2 + D^2 \le 1\): либо измерить, либо убрать из “выполнено”
Если хотите сохранить утверждение про which-path, вам нужен явный \(D\). Минимальный вариант:

- определить \(D\) как различимость путей по разности вероятностей (или по trace distance двух условных состояний среды/устройства, если у вас есть такой объект);
- сохранять `D` в `V_grid.parquet` и агрегировать `V2_plus_D2`.

Если \(D\) в симуляторе сейчас не определён — лучше в отчёте перенести это из “выполняется” в “интерпретация/ожидание”.

---

## 4) Насколько B сейчас строгий, если оценивать по “внешней проверяемости”
- **Сейчас**: уровень строгости ~“демонстрация + внутренний отчёт”. Внешний проверяющий не сможет воспроизвести результаты без вашего кода и без недостающих артефактов, а также остаётся логическая уязвимость с \(N^\*\).
- **После минимального патча выше**: B станет сопоставим с A по стандарту воспроизводимости (таблицы + manifest + контракт пересчёта), и тогда он реально сможет пройти внешнюю проверку.

---

Если хотите, я могу:
1) предложить точную Pydantic‑схему `ExperimentBReport` (аналогично схеме для A),  
2) дать список конкретных функций `export_V_grid(...)`, `export_V_summary(...)`, `build_manifest(...)`,  
3) и переписать ваши H1–H3 так, чтобы они были статистически строгими и не зависели от “удобного eta_test”.

Ниже — комплект “до внешней проверки” для Эксперимента B: (1) точная Pydantic‑схема `ExperimentBReport` (по стилю как для A), (2) конкретные функции экспорта и манифеста, (3) переписанные H1–H3 так, чтобы они были статистически строгими и **не зависели от “удобного eta_test”**. Я буду ссылаться на ваши текущие форматы/поля из [experiment_B_coherence.pdf](/files/JjEjnblu4BCUt6xzHBQwq) и структуру репо по [Algorithm_Final.md](/files/-Ci8U_0zxLE02nh7fjFII).

---

# 1) Pydantic‑схема `ExperimentBReport` (аналогично A)

Схема рассчитана на внешний аудит: в ней есть версии/seed’ы/контракт пересчёта метрик и ссылки на артефакты (`V_grid`, `V_summary`, `threshold_fit`, `manifest`).

## 1.1. Модели (Pydantic v2)

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl

# -------------------------
# Core provenance / versions
# -------------------------

class VersionInfo(BaseModel):
    schema_version: str = Field(..., description="e.g. 'B-1.0.0'")
    report_version: str = Field(..., description="e.g. 'v1' or notebook tag")
    code_commit: str = Field(..., description="Git commit hash")
    code_dirty: bool
    python: str
    numpy: str
    torch: Optional[str] = None
    platform: Optional[str] = None


class RNGInfo(BaseModel):
    base_seed: int
    n_runs_per_point: int = Field(..., ge=1)
    seeds: Optional[List[int]] = Field(None, description="Optional explicit list length=n_runs_per_point")
    deterministic_flags: Dict[str, Any] = Field(default_factory=dict, description="torch determinism settings etc.")


class ExperimentParams(BaseModel):
    # grid
    N_values: List[int]
    eta_values: List[float]
    T: int
    f_meas: float
    gamma: float

    # if present in your model as a parameter:
    N_star_model: Optional[int] = Field(None, description="If the model has an internal threshold parameter")

    # measurement operator config (must be explicit for external review)
    Pi_meas_spec: Dict[str, Any] = Field(default_factory=dict, description="Definition of measurement action/coarse-graining")
    visibility_spec: Dict[str, Any] = Field(default_factory=dict, description="How V is computed: Imax/Imin etc.")


# -------------------------
# Artifacts / manifest
# -------------------------

class ArtifactRef(BaseModel):
    path: str
    sha256: str
    kind: Literal["json", "csv", "parquet", "npz", "png", "md"]


class ArtifactManifest(BaseModel):
    run_dir: str
    artifacts: List[ArtifactRef]


# -------------------------
# Primary results (recomputable)
# -------------------------

class GridStats(BaseModel):
    # recomputable global stats
    V_mean_over_grid: float
    V_min_over_grid: float
    V_max_over_grid: float
    # optional: D stats if you compute distinguishability
    D_mean_over_grid: Optional[float] = None


class ThresholdEstimate(BaseModel):
    method: Literal["logistic_fit", "piecewise_two_means", "half_visibility_crossing", "changepoint"] = "logistic_fit"
    eta_aggregation: Literal["all_eta_weighted", "marginalize_eta", "eta_stratified_fit"] = "marginalize_eta"

    N_star_hat: float
    N_star_ci_low: Optional[float] = None
    N_star_ci_high: Optional[float] = None
    width: Optional[float] = None
    fit_quality: Optional[Dict[str, Any]] = None


class DerivedMetrics(BaseModel):
    # Core metrics used for gating
    drop_ratio: float = Field(..., description="V_above / V_below (properly defined below)")
    drop_ratio_ci_low: Optional[float] = None
    drop_ratio_ci_high: Optional[float] = None

    eta_effect_slope: float = Field(..., description="Slope of V vs eta in high-N regime (negative expected)")
    eta_effect_ci_low: Optional[float] = None
    eta_effect_ci_high: Optional[float] = None

    effect_size_d: float = Field(..., description="Cohen's d between no-meas and full-meas in high-N regime")
    effect_size_ci_low: Optional[float] = None
    effect_size_ci_high: Optional[float] = None

    # optional: complementarity if D is computed
    complementarity_max_violation: Optional[float] = Field(None, description="max(V^2 + D^2 - 1)")


# -------------------------
# Hypothesis tests (statistically strict)
# -------------------------

class HypothesisTestResult(BaseModel):
    confirmed: bool
    statistic: Dict[str, float] = Field(default_factory=dict)
    ci95: Optional[Dict[str, float]] = None
    p_value: Optional[float] = None
    notes: Optional[str] = None


class HypothesisTests(BaseModel):
    H1_phase_saturation: HypothesisTestResult
    H2_eta_dependence: HypothesisTestResult
    H3_informational_nature: HypothesisTestResult


# -------------------------
# TargetSpec contract
# -------------------------

class TargetThresholds(BaseModel):
    drop_ratio_max: float = Field(..., gt=0)          # e.g. 0.5 or 0.2
    eta_slope_max: float = Field(..., description="Should be negative; e.g. -0.05")
    effect_size_d_min: float = Field(..., gt=0)       # e.g. 1.5 or 2.0
    n_star_tolerance: Optional[float] = None          # if you require N_star_hat close to N_star_model
    complementarity_violation_max: Optional[float] = None


class RecomputeContract(BaseModel):
    # For external reviewers: where to recompute and how
    V_grid_table: str
    V_summary_table: str
    formulas: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)


class TargetSpecReport(BaseModel):
    hit: bool
    score: float = Field(..., ge=0, le=1)
    thresholds: TargetThresholds
    recompute_contract: RecomputeContract
    gating_metrics: List[str]
    non_gating_metrics: List[str] = Field(default_factory=list)


# -------------------------
# Top-level report
# -------------------------

class ExperimentBReport(BaseModel):
    experiment: Literal["B"] = "B"
    title: str
    created_at: str

    versions: VersionInfo
    rng: RNGInfo
    parameters: ExperimentParams

    artifacts: ArtifactManifest

    grid_stats: GridStats
    threshold: ThresholdEstimate
    derived_metrics: DerivedMetrics
    hypothesis_tests: HypothesisTests

    target_spec: TargetSpecReport

    conclusions: Dict[str, str] = Field(default_factory=dict)
    physical_interpretation: Dict[str, str] = Field(default_factory=dict)
```

---

# 2) Конкретные функции экспорта и manifest

Ниже — минимальные функции, которые превращают ваш `enhanced_results` (как в [experiment_B_coherence.pdf](/files/JjEjnblu4BCUt6xzHBQwq)) в “внешне проверяемый” пакет.

## 2.1. Что нужно хранить в памяти (важное изменение относительно текущего B)
Сейчас вы храните `V_mean[N][eta]`. Для статистики нужен **расклад по прогонам**: `V_runs[N][eta][run_id]` (или хотя бы суммарные моменты + n_runs). Внешняя проверка требует минимум `V_grid`.

### Рекомендуемый формат в коде
- `V_runs[(N,eta)] = list[float]` длины `n_runs_per_point`
- `D_runs[(N,eta)]` аналогично (если реализуете distinguishability)

---

## 2.2. `export_V_grid(...)` → `tables/V_grid.parquet`

```python
import pandas as pd
from pathlib import Path

def export_V_grid(V_runs: dict, run_dir: str, seeds: list[int] | None = None,
                  extra_cols: dict | None = None) -> str:
    """
    V_runs: dict[(N, eta)] -> list[float]  (length = n_runs_per_point)
    seeds: optional list[seed] aligned by run_id
    """
    rows = []
    for (N, eta), vals in V_runs.items():
        for run_id, V in enumerate(vals):
            row = {"N": int(N), "eta": float(eta), "run_id": int(run_id), "V": float(V)}
            if seeds is not None and run_id < len(seeds):
                row["seed"] = int(seeds[run_id])
            if extra_cols:
                row.update(extra_cols)
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["N", "eta", "run_id"]).reset_index(drop=True)
    out_path = Path(run_dir) / "tables" / "V_grid.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return str(out_path)
```

---

## 2.3. `export_V_summary(...)` → `tables/V_summary.parquet` (mean/std/CI)

```python
import numpy as np
import pandas as pd
from pathlib import Path

def _ci95_mean(x: np.ndarray) -> tuple[float, float]:
    # Minimal: normal approx; for strictness you can bootstrap
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    se = s / np.sqrt(len(x)) if len(x) > 0 else np.nan
    return m - 1.96 * se, m + 1.96 * se

def export_V_summary(V_runs: dict, run_dir: str) -> str:
    rows = []
    for (N, eta), vals in V_runs.items():
        x = np.asarray(vals, dtype=float)
        ci_low, ci_high = _ci95_mean(x)
        rows.append({
            "N": int(N),
            "eta": float(eta),
            "n_runs": int(len(x)),
            "V_mean": float(np.mean(x)),
            "V_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "V_ci95_low": float(ci_low),
            "V_ci95_high": float(ci_high),
        })
    df = pd.DataFrame(rows).sort_values(["N", "eta"]).reset_index(drop=True)
    out_path = Path(run_dir) / "tables" / "V_summary.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return str(out_path)
```

Примечание: для внешней проверки лучше **bootstrap CI** (устойчивее, не требует нормальности). Можно заменить `_ci95_mean` на bootstrap, но это уже второй шаг.

---

## 2.4. `build_manifest(...)` → `manifest.json` (sha256)

```python
import hashlib, json
from pathlib import Path

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_manifest(run_dir: str, paths: list[str]) -> str:
    run_dir = str(Path(run_dir))
    artifacts = []
    for p in paths:
        pth = Path(p)
        artifacts.append({
            "path": str(pth.relative_to(run_dir)) if str(pth).startswith(run_dir) else str(pth),
            "sha256": sha256_file(str(pth)),
            "kind": pth.suffix.lstrip("."),
        })
    manifest = {"run_dir": run_dir, "artifacts": artifacts}
    out_path = Path(run_dir) / "manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return str(out_path)
```

---

# 3) Переписываем H1–H3 строго и без “удобного eta_test”

Ниже — новые определения, которые:
- используют **все \(\eta\)** (агрегирование) или учитывают \(\eta\) через регрессию;
- используют **распределения по прогонам** (не только средние);
- дают CI/p-value.

Обозначим:
- \(V_{N,\eta}^{(k)}\) — видимость в прогоне \(k\)
- \( \bar V_{N,\eta}\) — среднее по прогонам
- high‑N = \(N \ge \hat N^\*\) (где \(\hat N^\*\) оцениваем из данных, либо используем заранее заданный `N_star_model`, но тогда это явно)

## 3.1. Подготовительный шаг: оценить \(\hat N^\*\) из данных (не “задавать”)
Рекомендую **логистический фит** к маргинализованной по \(\eta\) кривой:

1) Маргинализуем по eta:
\[
\bar V_N = \mathrm{mean}_\eta(\bar V_{N,\eta})
\]
(можно weighted mean по числу прогонов, обычно одинаковое).

2) Фитим логистику:
\[
\bar V_N \approx V_{\min} + \frac{V_{\max}-V_{\min}}{1+\exp((N-\hat N^\*)/w)}
\]

3) \(\hat N^\*\) — ваш “обнаруженный порог”.

**Зачем:** это снимает уязвимость “порог встроен”.

Если в вашей теории \(N^\*=512\) — предсказание, то тест становится:
\[
|\hat N^\* - 512| \le \Delta
\]
(например, \(\Delta=128\) при сетке N шагом 128).

---

## 3.2. H1 (Phase saturation exists): статистически строгий drop‑ratio + bootstrap CI
**Новая формулировка H1:**
> После порога \(\hat N^\*\) средняя видимость падает минимум в \(R\) раз по сравнению с допороговым режимом, устойчиво по всем \(\eta\).

Определим “допорог” и “послепорог” множества:
- \(S_- = \{(N,\eta,k): N < \hat N^\*\}\)
- \(S_+ = \{(N,\eta,k): N \ge \hat N^\*\}\)

Определим:
\[
\bar V_- = \mathrm{mean}_{S_-}(V), \quad \bar V_+ = \mathrm{mean}_{S_+}(V),\quad
\mathrm{drop\_ratio} = \frac{\bar V_+}{\bar V_-}
\]

**Критерий подтверждения:**
- `drop_ratio < drop_ratio_max` (например 0.2 или 0.5)
- и bootstrap 95% CI целиком ниже порога (строго): `CI_high < drop_ratio_max`.

Это не зависит от выбора `eta_test`, т.к. использует всё пространство \((N,\eta)\).

---

## 3.3. H2 (Dependence on η): регрессионный наклон в high‑N области
**Новая формулировка H2:**
> В высоко‑N режиме увеличение силы измерения \(\eta\) систематически снижает \(V\) (отрицательный наклон), и это статистически значимо.

Берём только high‑N точки, но все \(\eta\), и работаем на уровне агрегатов \(\bar V_{N,\eta}\) (или даже на уровне прогонов).

Модель:
\[
\bar V_{N,\eta} = a_N + b\,\eta + \epsilon
\]
где \(a_N\) — фикс‑эффект для каждого N (чтобы не смешивать зависимость от N и η), а \(b\) — общий наклон по η.

**Критерий подтверждения:**
- \(b < b_{\max}\) (например \(b_{\max}=-0.05\))
- и 95% CI по \(b\) полностью < 0 (значимость).

Технически это можно реализовать как:
- OLS с dummy‑переменными для N,
- или mixed model (но OLS проще и достаточно).

---

## 3.4. H3 (Informational nature): большой эффект‑размер между η=0 и η=1 в high‑N, и отсутствие “ложного падения” при η=0
**Новая формулировка H3 состоит из двух частей:**

### H3a: “эффект только при измерении” в high‑N
Сравниваем распределения:
- \(V_{hi}^{(0)} = \{V_{N,0}^{(k)}: N\ge \hat N^\*\}\)
- \(V_{hi}^{(1)} = \{V_{N,1}^{(k)}: N\ge \hat N^\*\}\)

Метрика: Cohen’s d:
\[
d = \frac{\mathrm{mean}(V_{hi}^{(0)})-\mathrm{mean}(V_{hi}^{(1)})}{s_{\mathrm{pooled}}}
\]

**Критерий:** `d >= d_min` (например 1.5–2.0) и bootstrap CI по d выше порога.

### H3b: “без измерения когерентность сохраняется” (не просто std<0.1)
Требование:
- \(\mathrm{mean}(V_{N,\eta=0})\) должен быть высок (например > 0.7) **во всём диапазоне N** или хотя бы до порога.
- и не должно быть тренда на спад с N при η=0 (наклон ~0).

Это убирает слабость старого критерия “std<0.1”, который не гарантирует, что V вообще высокая.

---

# 4) Минимальный код для новых H1–H3 (концептуальный)

Ниже — набросок функций, которые можно встроить в `generate_experiment_B_report(...)`, если у вас есть `V_grid` (данные по прогонам).

```python
import numpy as np
import pandas as pd

def bootstrap_ci_ratio(x_num, x_den, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n):
        num = rng.choice(x_num, size=len(x_num), replace=True).mean()
        den = rng.choice(x_den, size=len(x_den), replace=True).mean()
        ratios.append(num / den if den > 0 else np.nan)
    ratios = np.asarray(ratios)
    ratios = ratios[np.isfinite(ratios)]
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)), float(np.mean(ratios))

def cohens_d(a, b):
    a = np.asarray(a); b = np.asarray(b)
    sa = a.std(ddof=1); sb = b.std(ddof=1)
    sp = np.sqrt(((len(a)-1)*sa**2 + (len(b)-1)*sb**2) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.inf

def fit_eta_slope_fixedN(df_hi):  # df columns: N, eta, V_mean (or V)
    # OLS with N fixed effects: V ~ C(N) + eta
    # easiest approach: de-mean within each N
    df = df_hi.copy()
    df["V_dm"] = df["V_mean"] - df.groupby("N")["V_mean"].transform("mean")
    # now regress V_dm ~ eta through origin
    x = df["eta"].to_numpy()
    y = df["V_dm"].to_numpy()
    b = (x @ y) / (x @ x)
    # bootstrap CI over rows (or over N blocks) for strictness
    return float(b)
```

---

# 5) Как встроить в ваш проект (с учётом [Algorithm_Final.md](/files/-Ci8U_0zxLE02nh7fjFII))

Судя по структуре, у вас уже есть `target_simple.py` и `observer_simple.py` — это идеальные места:

- `observer_simple.py`:
  - собирает `V_runs`,
  - вызывает `export_V_grid/export_V_summary`,
  - считает `threshold_hat` и H1–H3,
  - собирает `ExperimentBReport`,
  - пишет JSON + manifest.

- `target_simple.py` (или отдельный `target_coherence.py`):
  - содержит `TargetThresholds` и функцию `evaluate_from_tables(...)` (как вы делали в A).

---

Если скажете, есть ли у вас уже в симуляторе многократные прогоны (n_runs_per_point) или сейчас считается только одно значение \(V\) на точку, я подстрою функции экспорта: либо сохранять один `run_id=0`, либо быстро добавить `for seed in seeds:` вокруг симуляции и собирать распределение. Это ключевое отличие между “внутренний отчёт” и “внешне проверяемый эксперимент”.
