import tkinter as tk
import random
import time
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("Sorting Visualizer & Performance Plotter")

# Global variables
numbers = []          # The array to be sorted.
sort_generator = None # Generator for animation.
plot_canvas = None    # Matplotlib canvas widget in the bottom plot frame.
data_points = []      # Accumulates tuples of (array_length, sort_time, color).
current_algo = None   # Tracks which sorting algorithm is currently used.

# Define unique colors for each sorting algorithm.
algo_colors = {
    "merge": "red",
    "heap": "blue",
    "quick": "orange",
    "insertion": "purple"
}

# Tkinter BooleanVars for checkboxes.
float_var = tk.BooleanVar(value=False)       # If True, generate floats (2 decimals).
visualizer_var = tk.BooleanVar(value=True)     # If True, animate sort; if False, plot performance.

# ---------------- Utility Functions ----------------

def get_array_from_entry():
    """
    Reads the array from the editable text box.
    Tries to safely evaluate the text as a Python list,
    or splits by commas if needed.
    """
    text = entry_array.get().strip()
    try:
        arr = ast.literal_eval(text)
        if not isinstance(arr, list):
            raise ValueError("Not a list")
        return arr
    except Exception:
        try:
            arr = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
            return arr
        except Exception:
            return None

def generate_array():
    """
    Generates an array using the length from the length entry.
    Also reads the min and max values from the corresponding boxes.
    If "Generate Floats" is on, the array contains floats (2 decimals);
    otherwise, it contains integers.
    The generated array is placed in the editable array box.
    """
    try:
        length = int(entry_length.get())
    except ValueError:
        entry_array.delete(0, tk.END)
        entry_array.insert(0, "Invalid length!")
        return

    try:
        min_val = int(entry_min.get())
    except ValueError:
        min_val = 1

    try:
        max_val = int(entry_max.get())
    except ValueError:
        max_val = 100

    if min_val > max_val:
        entry_array.delete(0, tk.END)
        entry_array.insert(0, "Invalid range!")
        return

    if float_var.get():
        arr = [round(random.uniform(min_val, max_val), 2) for _ in range(length)]
    else:
        arr = [random.randint(min_val, max_val) for _ in range(length)]
    entry_array.delete(0, tk.END)
    entry_array.insert(0, str(arr))

def draw_array(array, color_positions={}):
    """
    Draws the array on the visualizer canvas as vertical bars.
    The bars are scaled using the 'Min' and 'Max' entry values so that negative
    values are handled correctly.
    """
    canvas.delete("all")
    if not array:
        return
    canvas_width = 300
    canvas_height = 200
    try:
        lower_bound = float(entry_min.get())
    except:
        lower_bound = 0
    try:
        upper_bound = float(entry_max.get())
    except:
        upper_bound = 100
    if upper_bound == lower_bound:
        upper_bound = lower_bound + 1
    bar_width = canvas_width / len(array)
    for i, value in enumerate(array):
        x0 = i * bar_width
        # Map value from [lower_bound, upper_bound] to [0, canvas_height]
        normalized = (value - lower_bound) / (upper_bound - lower_bound)
        y0 = canvas_height - (normalized * canvas_height)
        x1 = (i + 1) * bar_width
        y1 = canvas_height
        color = color_positions.get(i, "grey")
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
    canvas.update()

def plot_graph():
    """
    Creates and updates a matplotlib plot in the bottom frame.
    X axis: Array Length, Y axis: Sort Time (s). Data points are grouped by algorithm
    and connected with lines. The legend displays the sort algorithm name.
    """
    global plot_canvas
    if plot_canvas is not None:
        plot_canvas.get_tk_widget().destroy()
        plot_canvas = None

    # Group data points by color.
    groups = {}
    for x, y, col in data_points:
        groups.setdefault(col, []).append((x, y))
    
    # Invert the algo_colors dictionary to get algorithm names from colors.
    color_to_algo = {v: k.capitalize() for k, v in algo_colors.items()}
    
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, points in groups.items():
        # Sort points by array length.
        points.sort(key=lambda pt: pt[0])
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        algo_name = color_to_algo.get(col, "Unknown")
        ax.plot(xs, ys, marker='o', color=col, label=algo_name)
    ax.set_xlabel("Array Length")
    ax.set_ylabel("Sort Time (s)")
    ax.set_title("Sorting Performance")
    ax.legend()
    plot_canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    plot_canvas.draw()
    widget = plot_canvas.get_tk_widget()
    widget.pack(fill=tk.BOTH, expand=True)

def reset_plot():
    """Clears the accumulated performance data and resets the plot."""
    global data_points
    data_points = []
    plot_graph()

def animate():
    """
    Animates the sorting process using the current sort_generator.
    Highlights swapped indices using the current algorithm’s color.
    """
    global sort_generator
    try:
        array_state, idx1, idx2 = next(sort_generator)
        if idx1 is not None and idx2 is not None:
            highlight_color = algo_colors.get(current_algo, "red")
            draw_array(array_state, {idx1: highlight_color, idx2: highlight_color})
        else:
            draw_array(array_state)
        root.after(100, animate)
    except StopIteration:
        entry_array.delete(0, tk.END)
        entry_array.insert(0, str(numbers))
        draw_array(numbers)

# ---------------- Sorting Generators ----------------

def merge_sort_generator():
    """Iterative merge sort generator that yields the array state after each merge."""
    global numbers
    n = len(numbers)
    curr_size = 1
    while curr_size < n:
        for left in range(0, n, 2 * curr_size):
            mid = min(left + curr_size, n)
            right = min(left + 2 * curr_size, n)
            merged = []
            i, j = left, mid
            while i < mid and j < right:
                if numbers[i] <= numbers[j]:
                    merged.append(numbers[i])
                    i += 1
                else:
                    merged.append(numbers[j])
                    j += 1
            while i < mid:
                merged.append(numbers[i])
                i += 1
            while j < right:
                merged.append(numbers[j])
                j += 1
            for k, val in enumerate(merged):
                numbers[left + k] = val
            yield numbers, left, right - 1
        curr_size *= 2
    yield numbers, None, None

def heap_sort_generator():
    """Heap sort generator that yields the array state at each swap."""
    global numbers
    n = len(numbers)
    def heapify(i, heap_size):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < heap_size and numbers[left] > numbers[largest]:
            largest = left
        if right < heap_size and numbers[right] > numbers[largest]:
            largest = right
        if largest != i:
            numbers[i], numbers[largest] = numbers[largest], numbers[i]
            yield numbers, i, largest
            yield from heapify(largest, heap_size)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(i, n)
    for i in range(n - 1, 0, -1):
        numbers[0], numbers[i] = numbers[i], numbers[0]
        yield numbers, 0, i
        yield from heapify(0, i)
    yield numbers, None, None

def quick_sort_generator():
    """Iterative quick sort generator that yields the array state at each swap."""
    global numbers
    stack = [(0, len(numbers) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot = numbers[high]
            i = low
            for j in range(low, high):
                if numbers[j] < pivot:
                    numbers[i], numbers[j] = numbers[j], numbers[i]
                    yield numbers, i, j
                    i += 1
            numbers[i], numbers[high] = numbers[high], numbers[i]
            yield numbers, i, high
            stack.append((low, i - 1))
            stack.append((i + 1, high))
    yield numbers, None, None

def insertion_sort_generator():
    """Insertion sort generator that yields the array state after each insertion."""
    global numbers
    n = len(numbers)
    for i in range(1, n):
        key = numbers[i]
        j = i - 1
        while j >= 0 and numbers[j] > key:
            numbers[j + 1] = numbers[j]
            yield numbers, j, j+1
            j -= 1
        numbers[j + 1] = key
        yield numbers, j+1, i
    yield numbers, None, None

# ---------------- Performance Points Generation ----------------

def generate_multiple_performance_points():
    """
    Reads the array from the editable array box (which should contain a list of numbers).
    For each number n in that list, it generates a random array of length n,
    times the sort using the currently selected algorithm,
    and appends a point (x = n, y = sort time) to the performance data.
    Then updates the plot.
    """
    global data_points, current_algo, numbers
    arr = get_array_from_entry()
    if arr is None:
        entry_array.delete(0, tk.END)
        entry_array.insert(0, "Invalid array input!")
        return
    if current_algo is None:
        current_algo = "insertion"
    algo_func = {
        "merge": merge_sort_generator,
        "heap": heap_sort_generator,
        "quick": quick_sort_generator,
        "insertion": insertion_sort_generator
    }.get(current_algo, insertion_sort_generator)
    
    for n in arr:
        try:
            n_int = int(n)
        except:
            continue
        if float_var.get():
            test_arr = [round(random.uniform(1, 100), 2) for _ in range(n_int)]
        else:
            test_arr = [random.randint(1, 100) for _ in range(n_int)]
        start_time = time.perf_counter()
        numbers = test_arr.copy()
        for _ in algo_func():
            pass
        end_time = time.perf_counter()
        sort_time = end_time - start_time
        data_points.append((n_int, sort_time, algo_colors.get(current_algo, "black")))
    plot_graph()

# ---------------- Unified Sorting Function ----------------

def sort_algorithm(sort_gen_func):
    """
    Reads the array from the editable box, then either animates the sort (if visualizer is on)
    or—in non-visualizer mode—calls the multiple-points routine.
    """
    global numbers, sort_generator
    arr = get_array_from_entry()
    if arr is None:
        entry_array.delete(0, tk.END)
        entry_array.insert(0, "Invalid array input!")
        return
    numbers = arr

    if visualizer_var.get():
        draw_array(numbers)
        sort_generator = sort_gen_func()
        animate()
    else:
        generate_multiple_performance_points()

# ---------------- Analyze Array Function ----------------

def analyze_array():
    """
    Analyzes the array (from the editable box) and selects the most efficient
    sorting algorithm based on several criteria:
    
    1. Size of the Dataset:
       - Small datasets (fewer than 10–20 elements): Insertion sort is efficient.
    
    2. Existing Order (Presortedness):
       - Nearly sorted data (≥90% adjacent pairs in order): Insertion sort works exceptionally well.
       - Moderately sorted data (70%-90%): Quick sort is appropriate.
       - Random or reverse-ordered data (40%-70%): Merge sort is chosen for its consistency.
       - Largely unsorted data (<40%): Heap sort is selected for predictable performance.
    
    The chosen algorithm and condition are displayed, and that algorithm is used to sort the array.
    """
    global current_algo
    arr = get_array_from_entry()
    if arr is None or len(arr) < 2:
        entry_array.delete(0, tk.END)
        entry_array.insert(0, "Invalid or too short array!")
        return

    n = len(arr)
    count = sum(1 for i in range(n - 1) if arr[i] <= arr[i + 1])
    ratio = count / (n - 1)  # Ratio of sorted adjacent pairs

    if n < 20:
        chosen = "insertion"
        condition = "Small dataset (<20 elements)"
    elif ratio >= 0.9:
        chosen = "insertion"
        condition = "Nearly sorted (≥90% sorted)"
    elif ratio >= 0.7:
        chosen = "quick"
        condition = "Moderately sorted (70%-90% sorted)"
    elif ratio >= 0.4:
        chosen = "merge"
        condition = "Random/reverse-ordered (40%-70% sorted)"
    else:
        chosen = "heap"
        condition = "Largely unsorted (<40% sorted)"
    
    current_algo = chosen
    label_analysis_result.config(text=f"Chosen Algorithm: {chosen.capitalize()} ({condition})")
    
    algo_func = {
        "insertion": insertion_sort_generator,
        "merge": merge_sort_generator,
        "quick": quick_sort_generator,
        "heap": heap_sort_generator
    }.get(chosen, quick_sort_generator)
    
    sort_algorithm(algo_func)

# ---------------- Button Command Wrappers ----------------

def sort_merge():
    global current_algo
    current_algo = "merge"
    sort_algorithm(merge_sort_generator)

def sort_heap():
    global current_algo
    current_algo = "heap"
    sort_algorithm(heap_sort_generator)

def sort_quick():
    global current_algo
    current_algo = "quick"
    sort_algorithm(quick_sort_generator)

def sort_insertion():
    global current_algo
    current_algo = "insertion"
    sort_algorithm(insertion_sort_generator)

# ---------------- Layout Setup ----------------

# Top frame holds two subframes: left for controls, right for visualizer canvas.
frame_top = tk.Frame(root)
frame_top.pack(side=tk.TOP, padx=10, pady=10)

frame_left = tk.Frame(frame_top)
frame_left.pack(side=tk.LEFT, padx=10, pady=10)

frame_right = tk.Frame(frame_top)
frame_right.pack(side=tk.RIGHT, padx=10, pady=10)

# Editable array text box.
entry_array = tk.Entry(frame_left, width=50)
entry_array.pack(padx=5, pady=5)

# Frame to hold the length, min, and max entries.
frame_entries = tk.Frame(frame_left)
frame_entries.pack(padx=5, pady=5)

# Entry for desired array length.
label_length = tk.Label(frame_entries, text="Length:")
label_length.grid(row=0, column=0, padx=5)
entry_length = tk.Entry(frame_entries, width=10)
entry_length.grid(row=0, column=1, padx=5)

# Entry for min number.
label_min = tk.Label(frame_entries, text="Min:")
label_min.grid(row=0, column=2, padx=5)
entry_min = tk.Entry(frame_entries, width=10)
entry_min.grid(row=0, column=3, padx=5)

# Entry for max number.
label_max = tk.Label(frame_entries, text="Max:")
label_max.grid(row=0, column=4, padx=5)
entry_max = tk.Entry(frame_entries, width=10)
entry_max.grid(row=0, column=5, padx=5)

# Frame for buttons under the entries.
frame_buttons = tk.Frame(frame_left)
frame_buttons.pack(padx=5, pady=5)

# Button to generate the array.
generate_button = tk.Button(frame_buttons, text="Generate Array", command=generate_array)
generate_button.pack(side=tk.LEFT, padx=5, pady=5)

# New "Analyze Array" button to the right of "Generate Array".
analyze_button = tk.Button(frame_buttons, text="Analyze Array", command=analyze_array)
analyze_button.pack(side=tk.LEFT, padx=5, pady=5)

# Label to show which algorithm was chosen by analyze_array.
label_analysis_result = tk.Label(frame_left, text="Chosen Algorithm: None")
label_analysis_result.pack(padx=5, pady=5)

# 2x2 grid for sort buttons.
sort_button_frame = tk.Frame(frame_left)
sort_button_frame.pack(padx=5, pady=5)

merge_button = tk.Button(sort_button_frame, text="Merge Sort", command=sort_merge)
merge_button.grid(row=0, column=0, padx=5, pady=5)

heap_button = tk.Button(sort_button_frame, text="Heap Sort", command=sort_heap)
heap_button.grid(row=0, column=1, padx=5, pady=5)

quick_button = tk.Button(sort_button_frame, text="Quick Sort", command=sort_quick)
quick_button.grid(row=1, column=0, padx=5, pady=5)

insertion_button = tk.Button(sort_button_frame, text="Insertion Sort", command=sort_insertion)
insertion_button.grid(row=1, column=1, padx=5, pady=5)

# Checkboxes.
float_checkbox = tk.Checkbutton(frame_left, text="Generate Floats (2 decimals)", variable=float_var)
float_checkbox.pack(padx=5, pady=5)

visualizer_checkbox = tk.Checkbutton(frame_left, text="Enable Visualizer", variable=visualizer_var)
visualizer_checkbox.pack(padx=5, pady=5)

# Button to reset the performance plot.
reset_plot_button = tk.Button(frame_left, text="Reset Plot", command=reset_plot)
reset_plot_button.pack(padx=5, pady=5)

# Visualizer canvas in the right frame.
canvas = tk.Canvas(frame_right, width=300, height=200, bg="white")
canvas.pack()

# Bottom frame for the performance plot.
frame_plot = tk.Frame(root)
frame_plot.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
