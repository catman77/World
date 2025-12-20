---

## Структура State (S) в RSL-системе

На основе анализа кода, состояние **S** системы организовано в несколько слоёв:

### 1. **Микроскопический слой (World)**

**Файл**: world.py

```python
class World:
    # Основные компоненты state:
    s: np.ndarray        # Спины s[i] ∈ {+1, -1}, shape=(N,)
    phi: np.ndarray      # Гравитационный потенциал φ[i] ∈ ℝ, shape=(N,)
    graph: GraphStructure  # Power-law граф связей
    t: int               # Время (дискретное)
    omega_cycles: List[OmegaCycle]  # Детектированные Ω-циклы
```

| Компонент | Тип | Описание |
|-----------|-----|----------|
| `s` | `np.int8[N]` | 1D массив спинов ±1 (материя) |
| `phi` | `np.float64[N]` | φ-поле гравитационного потенциала |
| `graph` | `GraphStructure` | Граф с рёбрами, Лапласианом, 3D embedding |
| `t` | `int` | Дискретное время |
| `omega_cycles` | `List` | Кэш Ω-циклов (частиц) |

### 2. **Граф (GraphStructure)**

**Файл**: graph_structure.py

```python
class GraphStructure:
    edges: List[Tuple[int, int]]     # Список рёбер
    _neighbors: Dict[int, List[int]] # Соседи каждой вершины
    _laplacian: sparse.csr_matrix    # Graph Laplacian L
    embedding_3d: np.ndarray         # 3D координаты для IFACE
```

### 3. **Переписывание (Rules + Evolution)**

**Файл**: rules.py, evolution.py

```python
class Rule:
    pattern: List[int]       # Паттерн для поиска (напр. [1,1,-1])
    replacement: List[int]   # Замена (напр. [-1,1,1])

class RuleSet:
    rules: List[Rule]        # SM-правила: ++- ↔ -++
```

Правила применяются детерминированно через `EvolutionEngine.step()`.

### 4. **OBS/IFACE - Наблюдаемое состояние**

**Файлы**: iface.py, global_observer.py

```python
@dataclass
class IFACEState:
    t: int                          # Время симуляции
    tau: float                      # Собственное время наблюдателя
    objects: List[IFACEObject]      # Частицы (из Ω-циклов)
    field: IFACEField               # Поля в 3D сетке
    total_Q: float                  # Полный заряд
    total_mass: float               # Полная масса
    global_Q: float                 # Топологический заряд

@dataclass 
class IFACEObject:
    id: int
    mass: float
    Q: float                        # Заряд
    pos: Tuple[float, float, float] # 3D позиция
    vel: Tuple[float, float, float] # Скорость
    period: int                     # Период Ω-цикла

@dataclass
class IFACEField:
    phi: np.ndarray       # 3D сетка φ-поля (16³)
    capacity: np.ndarray  # Ёмкость C (→ замедление времени)
    tension: np.ndarray   # Натяжение H (энергия)
```

### 5. **Память наблюдателя (History + Semantics)**

**Файлы**: semantics.py, global_observer.py

```python
class GlobalObserver:
    semantic_state: SemanticState      # "Знания" наблюдателя
    semantic_history: SemanticHistory  # История семантики
    iface_history: IFACEHistory        # История IFACE (для fitting)
    _prev_objects: Dict[int, IFACEObject]  # Для вычисления скоростей

@dataclass
class SemanticState:
    field_params: FieldEquationParams  # κ, m², λ
    conservation_laws: List[ConservationLaw]  # Q, mass
    gravity_law: GravityLaw            # F ~ r^(-α)
    event_stats: EventStatistics       # Распады/рассеяния

class IFACEHistory:
    states: List[IFACEState]           # Временной ряд IFACE
    max_history: int = 1000
```

### 6. **Лог событий**

Явного `event_log` в коде **нет**. События (применённые правила, рождение/распад частиц) отслеживаются косвенно:

- `EvolutionEngine.step()` возвращает результат применения правил
- `IFACEHistory` хранит временной ряд состояний
- `SemanticHistory` отслеживает изменения "знаний"

---

## Схема: что входит в полный State

```
S(t) = {
  // МИКРО (1D, OBS-невидимо)
  s[0..N-1]     : np.int8       — спины ±1
  phi[0..N-1]   : np.float64    — гравитационный потенциал
  
  // СТРУКТУРА
  Graph {
    edges       : List[(i,j)]   — рёбра графа
    Laplacian   : sparse        — для ∂φ/∂t = D·Lφ + source
    embedding_3d: np.float64[N,3] — 3D координаты
  }
  
  // ПЕРЕПИСЫВАНИЕ
  RuleSet       : [++- → -++, -++ → ++-]   — SM-правила
  
  // IFACE (3D, OBS-видимо)
  IFACEState {
    objects[]   : частицы (из Ω-циклов)
    field.phi   : 3D сетка φ
    field.C     : capacity (время)
  }
  
  // ПАМЯТЬ OBS
  IFACEHistory  : [IFACE(0), IFACE(1), ...]   — временной ряд
  SemanticState : выученные законы (F~1/r², сохранение Q)
  
  // ЛОГ СОБЫТИЙ
  (неявный через IFACEHistory и diff состояний)
}
```

---

**Вывод**: State — это многослойная структура, где микроскопические данные (s, φ, граф) проецируются через $\Pi_{\text{obs}}$ в наблюдаемый IFACE. Память наблюдателя хранится в `IFACEHistory` и `SemanticState`. Явного лога событий нет — он восстанавливается из разницы последовательных состояний.