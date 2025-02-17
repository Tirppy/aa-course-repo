import time
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
import math
import sys

sys.setrecursionlimit(1000000)

def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_memoization(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    return memo[n]

def fibonacci_bottom_up(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def matrix_multiply(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]

def matrix_exponentiation(n):
    def power(M, p):
        result = [[1, 0], [0, 1]]
        base = M
        while p > 0:
            if p % 2 == 1:
                result = matrix_multiply(result, base)
            base = matrix_multiply(base, base)
            p //= 2
        return result

    if n == 0:
        return 0
    F = [[1, 1], [1, 0]]
    result = power(F, n - 1)
    return result[0][0]

def binet_formula(n):
    getcontext().prec = 60  
    phi = Decimal(1 + math.sqrt(5)) / 2
    psi = Decimal(1 - math.sqrt(5)) / 2
    return int((phi ** n - psi ** n) / Decimal(math.sqrt(5)))

def fibonacci_fast_doubling(n):
    def fib(n):
        if n == 0:
            return (0, 1)
        a, b = fib(n // 2)
        c = a * (2 * b - a)
        d = a * a + b * b
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)
    return fib(n)[0]

def measure_time_and_plot(fib_function, max_n, label, ax):
    times = []
    fibonacci_values = []
    nodes = 10
    step = max_n // nodes
    
    for n in range(1, max_n + 1, step):
        start_time = time.time()  
        fib_value = fib_function(n)
        end_time = time.time()  
        times.append(end_time - start_time)
        fibonacci_values.append(fib_value)
        
    ax.plot(range(1, max_n + 1, step), times, marker='o', label=label)
    ax.set_xlabel('Fibonacci Term (n)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time vs Fibonacci Term')
    ax.grid(True)
    ax.legend()

recursive_n = 35
memoization_n = 460000
bottom_up_n = 190000
matrix_exponentiation_n = 1000000
binet_formula_n = 1000000
fast_doubling_n = 1000000

fig, ax = plt.subplots(figsize=(12, 8))

measure_time_and_plot(fibonacci_recursive, recursive_n, "Recursive Method", ax)
measure_time_and_plot(fibonacci_memoization, memoization_n, "Memoization Method", ax)
measure_time_and_plot(fibonacci_bottom_up, bottom_up_n, "Bottom-Up Method", ax)
measure_time_and_plot(matrix_exponentiation, matrix_exponentiation_n, "Matrix Exponentiation Method", ax)
measure_time_and_plot(binet_formula, binet_formula_n, "Binet's Formula Method", ax)
measure_time_and_plot(fibonacci_fast_doubling, fast_doubling_n, "Fast Doubling Method", ax)

plt.title('Comparison of Fibonacci Methods')
plt.grid(True)
plt.show()
