import random
import math
import sys
import time
import logging
from colorama import init, Fore
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def clip(value, low, high):
    return max(low, min(value, high))


class LocalSearchOptimizer:
    """
    Клас із функцією Сфери та трьома алгоритмами:
      - Hill Climbing
      - Random Local Search
      - Simulated Annealing
    Для 2D-випадку: bounds=[(-5,5),(-5,5)].
    """

    def __init__(self, bounds, iterations=1000, epsilon=1e-6):
        """
        :param bounds: список кортежів [(low, high), (low, high), ...].
        :param iterations: макс. кількість ітерацій на один запуск
        :param epsilon: чутливість (мінімальне покращення)
        """
        self.bounds = bounds
        self.iterations = iterations
        self.epsilon = epsilon

    @staticmethod
    def sphere_function(x):
        """Функція Сфери: f(x) = sum(x_i^2)."""
        return sum(xi**2 for xi in x)

    def hill_climbing(self):
        """
        Мінімізація Сфери алгоритмом Hill Climbing:
         - початкова випадкова точка
         - кожна ітерація робимо малий крок, якщо покращує > epsilon => приймаємо
        Повертає (best_solution, best_value, history)
        """
        current = [random.uniform(low, high) for (low, high) in self.bounds]
        current_eval = self.sphere_function(current)
        step_size = 0.01

        history = []

        for _ in range(self.iterations):
            history.append(current_eval)
            idx = random.randint(0, len(self.bounds) - 1)
            direction = random.choice([-1, 1])
            neighbor = list(current)
            neighbor[idx] += direction * step_size

            low, high = self.bounds[idx]
            neighbor[idx] = clip(neighbor[idx], low, high)

            neighbor_eval = self.sphere_function(neighbor)
            improvement = current_eval - neighbor_eval

            if improvement > self.epsilon:
                current, current_eval = neighbor, neighbor_eval

        # Останній запис
        history.append(current_eval)
        return current, current_eval, history

    def random_local_search(self):
        """
        Випадковий локальний пошук:
         - випадкова початкова точка
         - генеруємо candidate, якщо краща > epsilon => приймаємо
         - якщо покращення дуже мале => break
        Повертає (best_solution, best_value, history).
        """
        best = [random.uniform(low, high) for (low, high) in self.bounds]
        best_eval = self.sphere_function(best)

        history = []

        for _ in range(self.iterations):
            history.append(best_eval)
            candidate = [random.uniform(low, high) for (low, high) in self.bounds]
            candidate_eval = self.sphere_function(candidate)
            improvement = best_eval - candidate_eval
            if improvement > self.epsilon:
                best, best_eval = candidate, candidate_eval
            elif 0 < improvement < self.epsilon:
                best, best_eval = candidate, candidate_eval
                break

        history.append(best_eval)
        return best, best_eval, history

    def simulated_annealing(self, temp=1000, cooling_rate=0.95):
        """
        Імітація відпалу:
         - початкова випадкова точка
         - генеруємо сусіда, якщо кращий => беремо, якщо гірший => беремо з імовірністю exp(-delta/temp)
         - temp *= cooling_rate
         - якщо temp < epsilon => break
        Повертає (best_solution, best_value, history).
        """
        current = [random.uniform(low, high) for (low, high) in self.bounds]
        current_eval = self.sphere_function(current)
        best, best_eval = list(current), current_eval
        temperature = temp

        history = []

        for _ in range(self.iterations):
            history.append(current_eval)
            if temperature < self.epsilon:
                break

            neighbor = list(current)
            idx = random.randint(0, len(self.bounds) - 1)
            (low, high) = self.bounds[idx]
            step_size = (high - low) * 0.1
            neighbor[idx] += random.uniform(-step_size, step_size)
            neighbor[idx] = clip(neighbor[idx], low, high)

            neighbor_eval = self.sphere_function(neighbor)
            delta = neighbor_eval - current_eval

            if delta < 0:
                current, current_eval = neighbor, neighbor_eval
                if neighbor_eval < best_eval:
                    best, best_eval = neighbor, neighbor_eval
            else:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current, current_eval = neighbor, neighbor_eval

            temperature *= cooling_rate

        history.append(current_eval)
        return best, best_eval, history


class LocalSearchDemo:
    """
    Порівняння трьох алгоритмів:
      - Hill Climbing
      - Random Local Search
      - Simulated Annealing
    з кількома повтореннями (runs) та побудовою:
      - таблиці (Solution(2D), mean±std, avg_time, Closeness(%))
      - графіка "ітерація -> f(x)" (для одного "типового" запуску)
      - контурної діаграми
    """

    def __init__(self):
        # Межі: 2D
        self.bounds = [(-5, 5), (-5, 5)]
        self.iterations = 2000
        self.epsilon = 1e-6
        self.cooling_rate = 0.95
        self.temperature = 1000

        self.optimizer = LocalSearchOptimizer(
            bounds=self.bounds,
            iterations=self.iterations,
            epsilon=self.epsilon,
        )

        self.runs = 1000  # кількість повторних запусків

    def run_method_once(self, method_name):
        """Запускає method_name (один раз), повертає (sol, val, history, elapsed)."""
        start = time.time()
        if method_name == "hill_climbing":
            sol, val, hist = self.optimizer.hill_climbing()
        elif method_name == "random_local_search":
            sol, val, hist = self.optimizer.random_local_search()
        else:
            sol, val, hist = self.optimizer.simulated_annealing(
                temp=self.temperature, cooling_rate=self.cooling_rate
            )
        end = time.time()
        elapsed = end - start
        return sol, val, hist, elapsed

    def run_method_multiple_times(self, method_name):
        """
        Запускає method_name self.runs разів.
        Повертає список (sol, val, elapsed).
        """
        results = []
        for _ in range(self.runs):
            sol, val, hist, elapsed = self.run_method_once(method_name)
            results.append((sol, val, elapsed))
        return results

    @staticmethod
    def closeness_to_zero(fval):
        """
        Обчислюємо штучну метрику:
          closeness = 100 * (1 / (1 + fval))
        Якщо fval=0 => 100%. Якщо fval великий => ~0%.
        """
        return 100.0 * (1.0 / (1.0 + fval))

    def run_comparison(self):
        logger.info(Fore.CYAN + "=== Запуск трьох алгоритмів (кожен кілька разів) ===")

        # 1) Зберемо результати runs разів
        hc_results = self.run_method_multiple_times("hill_climbing")
        rls_results = self.run_method_multiple_times("random_local_search")
        sa_results = self.run_method_multiple_times("simulated_annealing")

        # 2) Для "типового" запуску (один раз) дістанемо history, щоб побудувати графік
        hc_sol, hc_val, hc_hist, hc_time = self.run_method_once("hill_climbing")
        rls_sol, rls_val, rls_hist, rls_time = self.run_method_once(
            "random_local_search"
        )
        sa_sol, sa_val, sa_hist, sa_time = self.run_method_once("simulated_annealing")

        def analyze_results(res_list):
            """
            res_list[i] = (sol, val, elapsed)
            """
            vals = [r[1] for r in res_list]
            times = [r[2] for r in res_list]
            avg_val = sum(vals) / len(vals)
            std_val = (sum((v - avg_val) ** 2 for v in vals) / len(vals)) ** 0.5
            avg_time = sum(times) / len(times)
            return avg_val, std_val, avg_time

        # Аналіз усіх повторів
        hc_avg_val, hc_std_val, hc_avg_time = analyze_results(hc_results)
        rls_avg_val, rls_std_val, rls_avg_time = analyze_results(rls_results)
        sa_avg_val, sa_std_val, sa_avg_time = analyze_results(sa_results)

        # Формуємо таблицю
        data = []

        def make_row(name, sol, val, avg_val, std_val, avg_time):
            # sol (2D) => формат строкою "[x.xx, y.yy]"
            sol_str = f"[{sol[0]:.3f}, {sol[1]:.3f}]"
            mean_std_str = f"{avg_val:.6e} ± {std_val:.2e}"
            time_str = f"{avg_time:.4f}s"
            # closeness
            closeness = self.closeness_to_zero(avg_val)
            closeness_str = f"{closeness:.2f}%"
            return [name, sol_str, mean_std_str, time_str, closeness_str]

        data.append(
            make_row(
                "Hill Climbing", hc_sol, hc_val, hc_avg_val, hc_std_val, hc_avg_time
            )
        )
        data.append(
            make_row(
                "Random Local", rls_sol, rls_val, rls_avg_val, rls_std_val, rls_avg_time
            )
        )
        data.append(
            make_row("Sim. Anneal", sa_sol, sa_val, sa_avg_val, sa_std_val, sa_avg_time)
        )

        table = tabulate(
            data,
            headers=[
                "Algorithm",
                "Solution (2D)",
                "f(x): mean ± std",
                f"Avg Time({self.runs} runs)",
                "Closeness(%)",
            ],
            tablefmt="github",
        )
        print(Fore.CYAN + "\n=== ПІДСУМКОВІ РЕЗУЛЬТАТИ ===")
        print(Fore.GREEN + table)

        # ---- Графік "ітерація -> f(x)"
        plt.figure(figsize=(7, 5))
        plt.title("Прогрес f(x) за ітераціями (приклад одного запуску)")
        plt.plot(hc_hist, label="Hill Climbing", color="red")
        plt.plot(rls_hist, label="Random Local", color="blue")
        plt.plot(sa_hist, label="Sim. Annealing", color="green")
        plt.xlabel("Iteration")
        plt.ylabel("f(x)")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # ---- Контурна діаграма
        logger.info(
            Fore.CYAN + "Будуємо контурну діаграму f(x)=x^2+y^2 + фінальні точки."
        )
        N = 200
        x_vals = np.linspace(self.bounds[0][0], self.bounds[0][1], N)
        y_vals = np.linspace(self.bounds[1][0], self.bounds[1][1], N)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = X**2 + Y**2

        plt.figure(figsize=(6, 5))
        plt.title("Контурна діаграма: f(x)=x^2+y^2 та фінальні точки")

        contour = plt.contourf(X, Y, Z, levels=30, cmap="viridis")
        plt.colorbar(contour)
        plt.contour(X, Y, Z, levels=30, colors="black", alpha=0.3)

        # Збираємо всі фінальні точки
        def get_final_points(method):
            """r[i] = (sol, val, time)"""
            r = self.run_method_multiple_times(method)
            return [ri[0] for ri in r]

        hc_points = get_final_points("hill_climbing")
        rls_points = get_final_points("random_local_search")
        sa_points = get_final_points("simulated_annealing")

        hc_x = [p[0] for p in hc_points]
        hc_y = [p[1] for p in hc_points]
        plt.scatter(hc_x, hc_y, color="red", label="HC runs")

        rls_x = [p[0] for p in rls_points]
        rls_y = [p[1] for p in rls_points]
        plt.scatter(rls_x, rls_y, color="blue", label="RLS runs")

        sa_x = [p[0] for p in sa_points]
        sa_y = [p[1] for p in sa_points]
        plt.scatter(sa_x, sa_y, color="green", label="SA runs")

        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    sys.setrecursionlimit(10**7)

    demo = LocalSearchDemo()
    demo.run_comparison()


if __name__ == "__main__":
    main()
