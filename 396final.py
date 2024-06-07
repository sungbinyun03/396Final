import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from zipfile import ZipFile
from io import BytesIO
import requests
import random

def extract_from_zip(url, file_name):
    print("Extracting data from zip file...")
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as z:
        with z.open(file_name) as f:
            data = pd.read_csv(f, delim_whitespace=True, header=None)
    print("Extraction complete.")
    return data

def visualize_data(data, title='Data'):
    print(f"Visualizing {title}...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    axs[0].hist(data, bins=50, color='tab:blue', alpha=0.6)
    axs[0].set_title(f'{title} Distribution')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].boxplot(data, vert=False)
    axs[1].set_title(f'{title} Boxplot')
    axs[1].set_xlabel('Value')

    summary = pd.Series(data).describe()
    axs[2].bar(summary.index, summary.values, color='tab:green', alpha=0.6)
    axs[2].set_title(f'{title} Statistical Summary')
    axs[2].set_ylabel('Value')

    fig.tight_layout()
    plt.show()
    print(f"Visualization of {title} complete.")

def non_private_mean(data):
    return np.mean(data)

def histogram_based_approach(data, epsilon, R):
    print("Running Histogram-based Approach...")
    n = len(data)
    interval_width = np.sqrt(np.log(n))
    intervals = np.arange(-R - interval_width, R + interval_width, interval_width)
    counts = np.zeros(len(intervals) - 1)

    for x in data:
        for i in range(len(intervals) - 1):
            if intervals[i] <= x < intervals[i + 1]:
                counts[i] += 1

    noisy_counts = counts + np.random.laplace(0, 1/epsilon, size=counts.shape)
    max_index = np.argmax(noisy_counts)
    selected_interval = (intervals[max_index], intervals[max_index + 1])

    clipped_data = np.clip(data, selected_interval[0], selected_interval[1])
    private_mean = np.mean(clipped_data) + np.random.laplace(0, (selected_interval[1] - selected_interval[0]) / (n * epsilon))
    
    print("Histogram-based Approach complete.")
    return private_mean

def shrinking_confidence_intervals(data, epsilon, R, iterations=10):
    print("Running Shrinking Confidence Intervals Approach...")
    n = len(data)
    interval = (-R - np.sqrt(np.log(n)), R + np.sqrt(np.log(n)))
    all_intervals = []
    all_means = []
    
    for i in range(iterations):
        clipped_data = np.clip(data, interval[0], interval[1])
        empirical_mean = np.mean(clipped_data)
        noise = np.random.laplace(0, (interval[1] - interval[0]) / (n * epsilon))
        private_mean = empirical_mean + noise
        interval_width = np.sqrt(1/n + (interval[1] - interval[0]) / (n * epsilon))
        interval = (private_mean - interval_width, private_mean + interval_width)
        
        all_intervals.append(interval)
        all_means.append(private_mean)
        
        print(f"Iteration {i+1}:")
        print(f"  Clipped Data: {clipped_data[:10]}...")  # Print first 10 values for brevity
        print(f"  Empirical Mean: {empirical_mean}")
        print(f"  Noise: {noise}")
        print(f"  Private Mean: {private_mean}")
        print(f"  Interval: {interval}\n")
    
    print("Shrinking Confidence Intervals Approach complete.")
    return private_mean, all_intervals, all_means

def simple_private_gaussian_mean(data, epsilon, R):
    print("Running Simple Private Gaussian Mean...")
    n = len(data)
    clipped_data = np.clip(data, -R - np.sqrt(np.log(n)), R + np.sqrt(np.log(n)))
    empirical_mean = np.mean(clipped_data)
    sensitivity = 2 * R / n
    noise = np.random.laplace(0, sensitivity / epsilon)
    private_mean = empirical_mean + noise
    print("Simple Private Gaussian Mean complete.")
    return private_mean

# Performance Evaluation
def evaluate_algorithms(data, epsilon, R, delta, iterations=10):
    print("Evaluating algorithms...")
    results = {}
    true_mean = np.mean(data)
    
    non_private_mean_value = non_private_mean(data)
    results['Non-Private'] = {'mean': non_private_mean_value, 'mse': 0, 'time': 0, 'bias': 0}
    
    start_time = time.time()
    hist_mean = histogram_based_approach(data, epsilon, R)
    hist_time = time.time() - start_time
    hist_mse = mean_squared_error([true_mean], [hist_mean])
    hist_bias = abs(hist_mean - true_mean)
    results['Histogram-based'] = {'mean': hist_mean, 'mse': hist_mse, 'time': hist_time, 'bias': hist_bias}
    
    start_time = time.time()
    sci_mean, all_intervals, all_means = shrinking_confidence_intervals(data, epsilon, R, iterations)
    sci_time = time.time() - start_time
    sci_mse = mean_squared_error([true_mean], [sci_mean])
    sci_bias = abs(sci_mean - true_mean)
    results['Shrinking Confidence Intervals'] = {'mean': sci_mean, 'mse': sci_mse, 'time': sci_time, 'bias': sci_bias, 'intervals': all_intervals, 'means': all_means}
    
    start_time = time.time()
    spg_mean = simple_private_gaussian_mean(data, epsilon, R)
    spg_time = time.time() - start_time
    spg_mse = mean_squared_error([true_mean], [spg_mean])
    spg_bias = abs(spg_mean - true_mean)
    results['Simple Private Gaussian'] = {'mean': spg_mean, 'mse': spg_mse, 'time': spg_time, 'bias': spg_bias}
    
    print("Algorithm evaluation complete.")
    return results

# Grid Search for Parameter Optimization
def grid_search_parameters(data, max_time=300):
    print("Starting grid search for parameter optimization...")
    epsilons = [0.1, 0.5, 1.0]
    Rs = [0.5, 1.0, 2.0]
    deltas = [1e-5, 1e-6]
    iterations_list = [10, 20]

    best_params = None
    best_mse = float('inf')
    start_time = time.time()

    for epsilon in epsilons:
        for R in Rs:
            for delta in deltas:
                for iterations in iterations_list:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_time:
                        print("Grid search exceeded max time limit.")
                        return best_params

                    print(f"Testing with epsilon={epsilon}, R={R}, delta={delta}, iterations={iterations}")
                    results = evaluate_algorithms(data, epsilon, R, delta, iterations)
                    avg_mse = np.mean([res['mse'] for res in results.values()])
                    if avg_mse < best_mse:
                        best_mse = avg_mse
                        best_params = (epsilon, R, delta, iterations)
    
    print("Grid search complete.")
    return best_params

def plot_results(results, title='Algorithm Performance Comparison'):
    print("Plotting results...")
    algorithms = list(results.keys())
    means = [results[alg]['mean'] for alg in algorithms]
    mse = [results[alg]['mse'] for alg in algorithms]
    times = [results[alg]['time'] for alg in algorithms]
    biases = [results[alg]['bias'] for alg in algorithms]
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 24))

    axs[0].bar(algorithms, means, color='tab:blue', alpha=0.6)
    axs[0].set_title('Mean Values')
    axs[0].set_ylabel('Mean')

    axs[1].bar(algorithms, mse, color='tab:blue', alpha=0.6)
    axs[1].set_title('Mean Squared Error (MSE)')
    axs[1].set_ylabel('MSE')

    axs[2].plot(algorithms, times, color='tab:red', marker='o')
    axs[2].set_title('Execution Time')
    axs[2].set_ylabel('Time (s)')

    axs[3].bar(algorithms, biases, color='tab:green', alpha=0.6)
    axs[3].set_title('Bias')
    axs[3].set_ylabel('Bias')
    axs[3].set_xlabel('Algorithm')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("Plotting complete.")

def demonstrate_non_private_sensitivity(data, sample_size=1000):
    print("Demonstrating non-private mean sensitivity...")
    true_mean = np.mean(data)
    sensitivities = []
    sample_indices = random.sample(range(len(data)), min(sample_size, len(data)))

    for i in sample_indices:
        data_copy = np.delete(data, i)
        new_mean = np.mean(data_copy)
        sensitivity = abs(true_mean - new_mean)
        sensitivities.append(sensitivity)

    print(f"Average sensitivity of non-private mean: {np.mean(sensitivities)}")
    return sensitivities

def demonstrate_privacy(data, epsilon, R):
    print("Demonstrating privacy...")
    n = len(data)
    
    non_private_mean_value = non_private_mean(data)
    
    noise = np.random.laplace(0, 1/(n * epsilon))
    private_mean_with_naive_noise = non_private_mean_value + noise

    hist_mean = histogram_based_approach(data, epsilon, R)
    sci_mean, _, _ = shrinking_confidence_intervals(data, epsilon, R)
    spg_mean = simple_private_gaussian_mean(data, epsilon, R)
    
    print(f"Non-Private Mean: {non_private_mean_value}")
    print(f"Private Mean with Naive Noise: {private_mean_with_naive_noise}")
    print(f"Histogram-based Private Mean: {hist_mean}")
    print(f"Shrinking Confidence Intervals Private Mean: {sci_mean}")
    print(f"Simple Private Gaussian Private Mean: {spg_mean}")

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
file_name = 'UCI HAR Dataset/train/X_train.txt'
uci_data = extract_from_zip(url, file_name)
uci_data = uci_data.values.flatten()  # Flatten to 1D array for mean estimation

visualize_data(uci_data, title='Original UCI HAR Data')

sqrt_uci_data = np.sqrt(uci_data - np.min(uci_data))

visualize_data(sqrt_uci_data, title='Square Root Transformed UCI HAR Data')

uci_sensitivity = demonstrate_non_private_sensitivity(sqrt_uci_data)
print(f"UCI Data Mean Sensitivity: {np.mean(uci_sensitivity)}")

best_params_uci = grid_search_parameters(sqrt_uci_data)
epsilon, R, delta, iterations = best_params_uci
print(f"Optimized Parameters for UCI Data: epsilon={epsilon}, R={R}, delta={delta}, iterations={iterations}")


results_uci = evaluate_algorithms(sqrt_uci_data, epsilon, R, delta, iterations)
plot_results(results_uci, title='UCI HAR Data Algorithm Performance')

demonstrate_privacy(sqrt_uci_data, epsilon, R)

iris = load_iris()
iris_data = iris.data[:, 0]  # Use the first feature (sepal length)

visualize_data(iris_data, title='Iris Data (Sepal Length)')

iris_sensitivity = demonstrate_non_private_sensitivity(iris_data)
print(f"Iris Data Mean Sensitivity: {np.mean(iris_sensitivity)}")
best_params_iris = grid_search_parameters(iris_data)
epsilon, R, delta, iterations = best_params_iris
print(f"Optimized Parameters for Iris Data: epsilon={epsilon}, R={R}, delta={delta}, iterations={iterations}")

results_iris = evaluate_algorithms(iris_data, epsilon, R, delta, iterations)
plot_results(results_iris, title='Iris Data Algorithm Performance')

demonstrate_privacy(iris_data, epsilon, R)


