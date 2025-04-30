import time
import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal, getcontext, InvalidOperation

getcontext().prec = 1000  

input_series_1 = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37]
input_series_2 = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849, 
                  20000, 25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000]
input_series_3 = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849, 
                  20000, 25000, 30000, 35000, 40000, 45000]
input_series_4 = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849, 
                  20000, 25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000]

def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_memoization(n):
    memo = {0: 0, 1: 1}
    for i in range(2, n + 1):
        memo[i] = memo[i - 1] + memo[i - 2]
    return memo[n]

def fibonacci_bottom_up(n):
    if n <= 1:
        return n
    a, b = 0, 1
    
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
    
def matrix_mult(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]

def matrix_power(M, p):
    res = [[1, 0], [0, 1]]  
    while p:
        if p % 2:
            res = matrix_mult(res, M)
        M = matrix_mult(M, M)
        p //= 2
    return res

def fibonacci_matrix_exponentiation(n):
    if n <= 1:
        return n
    M = [[0, 1], [1, 1]]
    result = matrix_power(M, n - 1)
    return result[1][1]

def fibonacci_binet(n):
    try:
        sqrt_5 = Decimal(5).sqrt()
        phi = (Decimal(1) + sqrt_5) / Decimal(2)
        return int((phi**n / sqrt_5).quantize(Decimal(1)))  
    except InvalidOperation:
        return None  

def fibonacci_fast_doubling(n):
    if n == 0:
        return 0
    a, b = 0, 1
    for bit in bin(n)[2:]:
        c = a * ((b << 1) - a)
        d = a * a + b * b
        a, b = (c, d) if bit == '0' else (d, c + d)
    return a

def measure_execution_time(func, input_series):
    results = []
    times = []
    for n in input_series:
        time_measurements = []
        for _ in range(3):  
            start_time = time.time()
            fib_n = func(n)
            elapsed_time = time.time() - start_time
            time_measurements.append(elapsed_time)
        
        avg_time = round(sum(time_measurements) / len(time_measurements), 4)  
        results.append(time_measurements)  
        times.append(avg_time)
    
    return results, times

methods = {
    "Recursive": fibonacci_recursive,
    # "Memoization": fibonacci_memoization,
    # "Bottom-Up": fibonacci_bottom_up,
    # "Matrix Exponentiation": fibonacci_matrix_exponentiation,
    # "Binet Formula": fibonacci_binet,
    # "Fast Doubling": fibonacci_fast_doubling,
}

all_results = {}

input_series_dict = {
    "Recursive": input_series_1,
    "Memoization": input_series_2,
    "Bottom-Up": input_series_2,
    "Matrix Exponentiation": input_series_2,
    "Binet Formula": input_series_2,
    "Fast Doubling": input_series_2
}

for name, method in methods.items():
    input_series = input_series_dict[name]
    print(f"Running {name} method...")
    results, times = measure_execution_time(method, input_series)
    
    
    all_results[name] = (results, times)

    df = pd.DataFrame(results, columns=["Test 0", "Test 1", "Test 2"], index=input_series)
    print(f"\nResults for {name}:")
    print(df.T)  

    plt.figure()
    plt.plot(input_series[:len(times)], times, marker='o', label=name)
    plt.xlabel("n-th Fibonacci Term")
    plt.ylabel("Execution Time (seconds)")
    plt.title(f"Performance of {name} Method")
    plt.legend()
    plt.grid()
    plt.savefig(f"{name}_performance.png")

plt.figure(figsize=(10, 6))
for name, (results, times) in all_results.items():
    input_series = input_series_dict[name]
    plt.plot(input_series[:len(times)], times, marker='o', label=name)

plt.xlabel("n-th Fibonacci Term")
plt.ylabel("Execution Time (seconds)")
plt.title("Comparison of Fibonacci Algorithms")
plt.legend()
plt.grid()
plt.savefig("Comparison_performance.png")

plt.show()