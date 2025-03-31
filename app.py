from flask import Flask, render_template, request
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')  # Важно: используем бэкенд, не создающий GUI
import matplotlib.pyplot as plt
import io
import base64
import itertools
from typing import List, Dict, Tuple, Union
import gc  # Для сборки мусора

app = Flask(__name__)


class Item:
    def __init__(self, weight: int, value: int):
        self.weight = weight
        self.value = value
        self.ratio = value / weight if weight != 0 else 0


def generate_items(n: int, max_weight: int = 100) -> List[Item]:
    np.random.seed(int(time.time()))  # Уникальный seed каждый раз
    items = []
    used_combinations = set()  # Для отслеживания уникальных комбинаций weight/value

    # Генерируем предметы с разными характеристиками
    weight_range = (5, 70)
    value_range = (5, 200)

    # Создаем пул возможных весов и значений
    possible_weights = list(range(weight_range[0], weight_range[1] + 1))
    possible_values = list(range(value_range[0], value_range[1] + 1))

    # Перемешиваем для случайности
    np.random.shuffle(possible_weights)
    np.random.shuffle(possible_values)

    # Генерируем уникальные предметы
    while len(items) < n and (possible_weights and possible_values):
        w = possible_weights.pop()
        v = possible_values.pop()

        # Проверяем уникальность комбинации
        if (w, v) not in used_combinations:
            items.append(Item(w, v))
            used_combinations.add((w, v))

    # Если не хватило уникальных комбинаций, дополняем случайными
    while len(items) < n:
        w = np.random.randint(weight_range[0], weight_range[1] + 1)
        v = np.random.randint(value_range[0], value_range[1] + 1)

        # Проверяем уникальность
        if (w, v) not in used_combinations:
            items.append(Item(w, v))
            used_combinations.add((w, v))

    # Добавляем вариативности в соотношение ценности к весу
    for item in items:
        # С небольшой вероятностью создаем "особые" предметы
        if np.random.random() < 0.2:
            if np.random.random() < 0.5:
                # Создаем очень ценный предмет
                item.value = int(item.value * np.random.uniform(1.5, 3.0))
            else:
                # Создаем "ловушку" - тяжелый, но малоценный
                item.weight = int(item.weight * np.random.uniform(1.5, 3.0))

        item.ratio = item.value / item.weight if item.weight != 0 else 0

    return items


# ========== 0/1 Knapsack Algorithms ==========
def brute_force_01(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    max_value = 0
    best_subset = []
    for r in range(1, len(items) + 1):
        for subset in itertools.combinations(items, r):
            total_weight = sum(item.weight for item in subset)
            if total_weight > capacity:
                continue
            total_value = sum(item.value for item in subset)
            if total_value > max_value:
                max_value = total_value
                best_subset = subset
    return list(best_subset), max_value


def greedy_01(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
    total_weight = 0
    total_value = 0
    selected = []
    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
            total_value += item.value
    return selected, total_value


def dp_01(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if items[i - 1].weight <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - items[i - 1].weight] + items[i - 1].value)
            else:
                dp[i][w] = dp[i - 1][w]

    w = capacity
    selected = []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(items[i - 1])
            w -= items[i - 1].weight

    return selected, dp[n][capacity]


def branch_and_bound_01(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    items = sorted(items, key=lambda x: x.ratio, reverse=True)
    best_value = 0
    best_subset = []

    def backtrack(i, current_weight, current_value, selected):
        nonlocal best_value, best_subset
        if current_weight > capacity:
            return
        if current_value > best_value:
            best_value = current_value
            best_subset = selected.copy()
        if i == len(items):
            return

        selected.append(items[i])
        backtrack(i + 1, current_weight + items[i].weight,
                  current_value + items[i].value, selected)
        selected.pop()

        backtrack(i + 1, current_weight, current_value, selected)

    backtrack(0, 0, 0, [])
    return best_subset, best_value


# ========== Fractional Knapsack Algorithms ==========
def brute_force_fractional(items: List[Item], capacity: int) -> Tuple[List[Tuple[Item, float]], float]:
    max_value = 0.0
    best_selection = []

    for r in range(1, len(items) + 1):
        for subset in itertools.combinations(items, r):
            remaining_cap = capacity
            current_value = 0.0
            current_selection = []

            for item in sorted(subset, key=lambda x: x.ratio, reverse=True):
                if remaining_cap <= 0:
                    break
                take = min(item.weight, remaining_cap)
                fraction = take / item.weight
                current_selection.append((item, fraction))
                current_value += item.value * fraction
                remaining_cap -= take

            if current_value > max_value:
                max_value = current_value
                best_selection = current_selection

    return best_selection, max_value


def greedy_fractional(items: List[Item], capacity: int) -> Tuple[List[Tuple[Item, float]], float]:
    sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
    remaining = capacity
    total_value = 0.0
    selected = []

    for item in sorted_items:
        if remaining <= 0:
            break
        take = min(item.weight, remaining)
        fraction = take / item.weight
        selected.append((item, fraction))
        total_value += item.value * fraction
        remaining -= take

    return selected, total_value


def dp_fractional(items: List[Item], capacity: int) -> Tuple[List[Tuple[Item, float]], float]:
    return greedy_fractional(items, capacity)


def branch_and_bound_fractional(items: List[Item], capacity: int) -> Tuple[List[Tuple[Item, float]], float]:
    return greedy_fractional(items, capacity)


# ========== Unbounded Knapsack Algorithms ==========
def brute_force_unbounded(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    max_value = 0
    best_combination = []

    def backtrack(remaining_cap, current_value, current_items, start_idx):
        nonlocal max_value, best_combination
        if remaining_cap < 0:
            return
        if current_value > max_value:
            max_value = current_value
            best_combination = current_items.copy()

        for i in range(start_idx, len(items)):
            item = items[i]
            if item.weight <= remaining_cap:
                current_items.append(item)
                backtrack(remaining_cap - item.weight,
                          current_value + item.value,
                          current_items, i)
                current_items.pop()

    backtrack(capacity, 0, [], 0)
    return best_combination, max_value


def greedy_unbounded(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    sorted_items = sorted(items, key=lambda x: x.ratio, reverse=True)
    remaining = capacity
    total_value = 0
    selected = []

    for item in sorted_items:
        while item.weight <= remaining:
            selected.append(item)
            total_value += item.value
            remaining -= item.weight

    return selected, total_value


def dp_unbounded(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    dp = [0] * (capacity + 1)
    item_combinations = [[] for _ in range(capacity + 1)]

    for w in range(1, capacity + 1):
        for item in items:
            if item.weight <= w and dp[w - item.weight] + item.value > dp[w]:
                dp[w] = dp[w - item.weight] + item.value
                item_combinations[w] = item_combinations[w - item.weight] + [item]

    return item_combinations[capacity], dp[capacity]


def branch_and_bound_unbounded(items: List[Item], capacity: int) -> Tuple[List[Item], int]:
    items = sorted(items, key=lambda x: x.ratio, reverse=True)
    best_value = 0
    best_combination = []

    def backtrack(remaining_cap, current_value, current_items, start_idx, upper_bound):
        nonlocal best_value, best_combination
        if remaining_cap < 0:
            return
        if current_value > best_value:
            best_value = current_value
            best_combination = current_items.copy()
        if start_idx >= len(items):
            return

        item = items[start_idx]
        max_possible = remaining_cap // item.weight
        new_upper_bound = current_value + max_possible * item.value

        if new_upper_bound <= best_value:
            return

        for count in range(max_possible, -1, -1):
            new_cap = remaining_cap - count * item.weight
            new_value = current_value + count * item.value
            new_items = current_items + [item] * count
            backtrack(new_cap, new_value, new_items, start_idx + 1, new_upper_bound)

    initial_upper_bound = sum((capacity // item.weight) * item.value for item in items)
    backtrack(capacity, 0, [], 0, initial_upper_bound)
    return best_combination, best_value


def solve_all_knapsacks(items: List[Item], capacity: int) -> Dict[str, Dict[str, Dict[str, Union[float, int]]]]:
    results = {}

    # 0/1 Knapsack
    methods_01 = {
        "Полный перебор": brute_force_01,
        "Жадный алгоритм": greedy_01,
        "Динамич. програм.": dp_01,
        "Мет. ветв. и границ": branch_and_bound_01
    }

    for name, method in methods_01.items():
        start = time.perf_counter()
        selected, value = method(items, capacity)
        elapsed = (time.perf_counter() - start) * 1_000_000
        results.setdefault("0/1", {})[name] = {
            "value": value,
            "time": elapsed,
            "weight": sum(item.weight for item in selected)
        }

    # Fractional Knapsack
    methods_frac = {
        "Полный перебор": brute_force_fractional,
        "Жадный алгоритм": greedy_fractional,
        "Динамич. програм.": dp_fractional,
        "Метод ветвей и границ": branch_and_bound_fractional
    }

    for name, method in methods_frac.items():
        start = time.perf_counter()
        selected, value = method(items, capacity)
        elapsed = (time.perf_counter() - start) * 1_000_000
        weight = sum(item.weight * frac for item, frac in selected) if isinstance(selected[0], tuple) else sum(
            item.weight for item in selected)
        results.setdefault("Fractional", {})[name] = {
            "value": value,
            "time": elapsed,
            "weight": weight
        }

    # Unbounded Knapsack
    methods_unb = {
        "Полный перебор": brute_force_unbounded,
        "Жадный алгоритм": greedy_unbounded,
        "Динамич. програм.": dp_unbounded,
        "Метод ветвей и границ": branch_and_bound_unbounded
    }

    for name, method in methods_unb.items():
        start = time.perf_counter()
        selected, value = method(items, capacity)
        elapsed = (time.perf_counter() - start) * 1_000_000
        results.setdefault("Unbounded", {})[name] = {
            "value": value,
            "time": elapsed,
            "weight": sum(item.weight for item in selected)
        }

    return results


def create_plots(results: Dict) -> str:
    plt.close('all')

    # Создаем фигуру с увеличенной высотой для отступов
    fig = plt.figure(figsize=(18, 20))  # Увеличили высоту с 18 до 20

    # Настройка сетки с увеличенным верхним отступом
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3,
                      top=0.88)  # top=0.88 - оставляет 12% пространства сверху

    # Словарь перевода
    task_types_translation = {
        "0/1": "Целочисленный рюкзак",
        "Fractional": "Дробный рюкзак",
        "Unbounded": "Неограниченный рюкзак"
    }

    colors = ['#ff9ff3', '#feca57', '#ff6b6b', '#48dbfb']

    try:
        # Добавляем основной заголовок с увеличенным отступом
        fig.suptitle("Сравнение алгоритмов для задачи о рюкзаке",
                     fontsize=18, y=0.95,  # y=0.95 - позиционируем заголовок в верхней части
                     fontweight='bold')  # Делаем шрифт полужирным

        for row, k_type in enumerate(["0/1", "Fractional", "Unbounded"]):
            if k_type not in results:
                continue

            translated_type = task_types_translation[k_type]
            methods = list(results[k_type].keys())
            times = [results[k_type][m]["time"] for m in methods]
            values = [results[k_type][m]["value"] for m in methods]

            # График времени
            ax1 = fig.add_subplot(gs[row, 0])
            bars_time = ax1.bar(methods, times, color=colors[:len(methods)])
            ax1.set_title(f'{translated_type} - Время (мкс)', fontsize=14, pad=12)

            # График ценности
            ax2 = fig.add_subplot(gs[row, 1])
            bars_value = ax2.bar(methods, values, color=colors[:len(methods)])
            ax2.set_title(f'{translated_type} - Макс. ценность', fontsize=14, pad=12)

            # Добавляем значения на столбцы
            for ax in [ax1, ax2]:
                for bar in ax.patches:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.1f}',
                            ha='center', va='bottom', fontsize=10)

        # Настройка общего расположения с дополнительным верхним отступом
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.90])  # Верхняя граница 0.90 вместо 0.95

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return plot_data

    finally:
        plt.close(fig)
        buf.close()
        gc.collect()


@app.route('/', methods=['GET', 'POST'])
def index():
    items = []
    plot_data = None
    error = ""
    n_items = 5
    capacity = int(request.form.get('capacity', 40)) if request.method == 'POST' else 40

    if request.method == 'POST':
        if 'generate' in request.form:
            try:
                n_items = min(int(request.form.get('n_items', 5)), 15)  # Ограничиваем максимум 15
                items = generate_items(n_items)
            except Exception as e:
                error = f"Ошибка генерации: {str(e)}"

        elif 'solve' in request.form:
            try:
                capacity = max(10, int(request.form.get('capacity', 40)))  # Минимум 10
                items_data = request.form.getlist('items')
                items = []
                for item_str in items_data:
                    weight, value = map(int, item_str.split(','))
                    items.append(Item(weight, value))

                if not items:
                    items = generate_items(n_items)

                results = solve_all_knapsacks(items, capacity)
                plot_data = create_plots(results)

            except Exception as e:
                error = f"Ошибка решения: {str(e)}"

    return render_template('index.html',
                           items=items,
                           plot_data=plot_data,
                           error=error,
                           n_items=n_items,
                           capacity=capacity)


if __name__ == '__main__':
    print("\n➜ Сервер запущен! Откройте в браузере: \033[94mhttp://localhost:5000\033[0m")
    print('Hello, World!')
    app.run(debug=True)