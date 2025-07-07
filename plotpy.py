#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:06:40 2025

@author: sajedehnorouzi
"""

import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# ✅ Path to your model folder
base_dir = '/Users/sajedehnorouzi/Desktop/TCOMletter/TcomFigs/model'

# ✅ All four algorithms
algorithms = {
    'DDPGagdrp': 'DDPGagdrp',
    'DDPGbaseline': 'DDPGbaseline',
    'SACagdrp': 'SACagdrp',
    'SACbaseline': 'SACbaseline'
}
display_name_map = {
    'DDPGagdrp': 'DDPG AGDRP',
    'DDPGbaseline': 'DDPG Baseline',
    'SACagdrp': 'SAC AGDRP',
    'SACbaseline': 'SAC Baseline'
}



# ✅ Metric files to load
metrics = ['Age_vehicle.mat', 'Flage_vehicle.mat', 'TT_link.mat', 'TT_vehicle.mat', 'reward.mat']

# ✅ Store data
data = {metric.split('.')[0]: {} for metric in metrics}

# ✅ Smoothing function (moving average)
def smooth(data, window_size=1):
    if len(data) < window_size:
        return data  # don't smooth short sequences
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def format_algo_name(name):
    return display_name_map.get(name, name)

# ✅ Load .mat files
for algo_name, folder in algorithms.items():
    folder_path = os.path.join(base_dir, folder)
    for metric_file in metrics:
        metric_name = metric_file.split('.')[0]
        file_path = os.path.join(folder_path, metric_file)
        if os.path.exists(file_path):
            mat_data = scipy.io.loadmat(file_path)
            print(f'✅ Loaded {metric_file} from {folder}: {list(mat_data.keys())}')
            if metric_name in mat_data:
                data[metric_name][algo_name] = mat_data[metric_name].squeeze()
            else:
                for key in mat_data:
                    if not key.startswith('__'):
                        data[metric_name][algo_name] = mat_data[key].squeeze()
                        break


#%%

# ✅ Plot with smoothing
for metric_name, algo_data in data.items():
    plt.figure(figsize=(10, 6))
    for algo_name, values in algo_data.items():
        if values is not None and len(values) > 0:
            smoothed = smooth(values, window_size=1)
            plt.plot(smoothed, label=algo_name)
        else:
            print(f"⚠️ No data to plot for {algo_name} in {metric_name}")
    plt.title(f'{metric_name} over Episodes (Smoothed)')
    plt.xlabel('Episode')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ Bar charts for TT_vehicle and TT_link with confidence intervals, custom Y-axis and colors
bar_metrics = ['TT_vehicle', 'TT_link']
y_axis_limits = {'TT_vehicle': 350, 'TT_link': 10000}

# ✅ Custom colors per algorithm
color_map = {
    'DDPGagdrp': 'blue',
    'DDPGbaseline': 'orange',
    'SACagdrp': 'green',
    'SACbaseline': 'red'
}

for metric_name in bar_metrics:
    algo_data = data.get(metric_name, {})
    if not algo_data:
        print(f"⚠️ No data found for {metric_name}")
        continue

    algo_names = []
    avg_values = []
    ci_values = []
    colors = []

    for algo_name, values in algo_data.items():
        if values is not None and len(values) > 1:
            values = np.array(values)
            mean = np.mean(values)
            std_dev = np.std(values, ddof=1)
            ci = 1.96 * (std_dev / np.sqrt(len(values)))

            algo_names.append(algo_name)
            avg_values.append(mean)
            ci_values.append(ci)
            colors.append(color_map.get(algo_name, 'gray'))  # Default to gray if unknown
        else:
            print(f"⚠️ Not enough data for confidence interval for {algo_name} in {metric_name}")

    # Plot bar chart with error bars and custom colors
    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, avg_values, yerr=ci_values, capsize=5,
            color=colors, edgecolor='black')
    plt.title(f'Average {metric_name} with 95% Confidence Interval')
    plt.ylabel(metric_name)
    plt.grid(axis='y')

    if metric_name in y_axis_limits:
        plt.ylim(bottom=y_axis_limits[metric_name])

    plt.tight_layout()
    plt.show()



#%%plot rewards 
# ✅ Plot only the 'reward' metric with smoothing
metric_name = 'reward'
algo_data = data.get(metric_name, {})

plt.figure(figsize=(10, 6))
for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        smoothed = smooth(values, window_size=1)
        plt.plot(smoothed, label=algo_name)
    else:
        print(f"⚠️ No data to plot for {algo_name} in {metric_name}")

plt.title(f'{metric_name} over Episodes (Smoothed)')
plt.xlabel('Episode')
plt.ylabel(metric_name)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ✅ Bar charts for TT_vehicle and TT_link with confidence intervals, custom Y-axis and light colors
bar_metrics = ['TT_vehicle', 'TT_link']
y_axis_lower_limits = {'TT_vehicle': 390, 'TT_link': 1}

# ✅ Light pastel colors per algorithm
color_map = {
    'DDPGagdrp': '#ADD8E6',      # Light Blue
    'DDPGbaseline': '#FFD580',   # Light Orange
    'SACagdrp': '#90EE90',       # Light Green
    'SACbaseline': '#FF9999'     # Light Red
}

for metric_name in bar_metrics:
    algo_data = data.get(metric_name, {})
    if not algo_data:
        print(f"⚠️ No data found for {metric_name}")
        continue

    algo_names = []
    avg_values = []
    ci_values = []
    colors = []

    for algo_name, values in algo_data.items():
        if values is not None and len(values) > 1:
            values = np.array(values)
            
            # ✅ Rescale TT_link values
            if metric_name == 'TT_link':
                values = values / 1e4

            mean = np.mean(values)
            std_dev = np.std(values, ddof=1)
            ci = 1.96 * (std_dev / np.sqrt(len(values)))

            algo_names.append(algo_name)
            avg_values.append(mean)
            ci_values.append(ci)
            colors.append(color_map.get(algo_name, 'lightgray'))
        else:
            print(f"⚠️ Not enough data for confidence interval for {algo_name} in {metric_name}")

    # ✅ Plot
    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, avg_values, yerr=ci_values, capsize=5,
            color=colors, edgecolor='black')
    plt.title(f'Average {metric_name} with 95% Confidence Interval')

    # ✅ Y-axis label with unit adjustment
    ylabel = metric_name
    if metric_name == 'TT_link':
        ylabel += r' ($\times 10^4$)'
    plt.ylabel(ylabel)

    # ✅ Set dynamic upper limit and fixed lower limit
    lower = y_axis_lower_limits.get(metric_name, 0)
    upper = max(avg_values) + max(ci_values)  # A little space above the tallest bar
    plt.ylim(bottom=lower, top=upper)

    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()






#%%

# ✅ Bar chart for Age_vehicle metric with custom pastel colors and Y-axis divided by 1000
# and formatted algorithm names

metric_name = 'Age_vehicle'
algo_data = data.get(metric_name, {})

# Light pastel colors per algorithm
color_map = {
    'DDPGagdrp': '#ADD8E6',      # Light Blue
    'DDPGbaseline': '#FFD580',   # Light Orange
    'SACagdrp': '#90EE90',       # Light Green
    'SACbaseline': '#FF9999'     # Light Red
}



algo_names = []
avg_values = []
std_values = []
colors = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        values = np.array(values)
        mean = np.mean(values) / 40  # divide by 1000
        std = np.std(values, ddof=1) / 1000  # divide by 1000
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        avg_values.append(mean)
        std_values.append(std)
        colors.append(color_map.get(algo_name, 'lightgray'))
    else:
        print(f"⚠️ No data for {algo_name} in {metric_name}")

plt.figure(figsize=(8, 5))
plt.bar(algo_names, avg_values, yerr=std_values, capsize=5,
        color=colors, edgecolor='black')
plt.title(f'Average {metric_name} with Standard Deviation')
plt.ylabel(f'{metric_name}')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#%%

#%%

# ✅ Bar chart for Age_vehicle metric with last values and custom pastel colors,
# with Y-axis divided by 40 and formatted algorithm names

metric_name = 'Age_vehicle'
algo_data = data.get(metric_name, {})

# Light pastel colors per algorithm
color_map = {
    'DDPGagdrp': '#ADD8E6',      # Light Blue
    'DDPGbaseline': '#FFD580',   # Light Orange
    'SACagdrp': '#90EE90',       # Light Green
    'SACbaseline': '#FF9999'     # Light Red
}


algo_names = []
last_values = []
std_values = []
colors = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        values = np.array(values)
        last_val = values[-1] / 40           # last value divided by 40
        std = np.std(values, ddof=1) / 1000 # std divided by 1000
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        last_values.append(last_val)
        std_values.append(std)
        colors.append(color_map.get(algo_name, 'lightgray'))
    else:
        print(f"⚠️ No data for {algo_name} in {metric_name}")

plt.figure(figsize=(8, 5))
plt.bar(algo_names, last_values, yerr=std_values, capsize=5,
        color=colors, edgecolor='black')
plt.title(f'Last {metric_name} Value with Standard Deviation')
plt.ylabel(f'{metric_name}')
plt.grid(axis='y')
plt.tight_layout()
plt.show()



#%%

# ✅ Bar chart for Age_vehicle metric with last values and custom pastel colors,
# with Y-axis divided by 40 and formatted algorithm names

metric_name = 'Age_vehicle'
algo_data = data.get(metric_name, {})

# Light pastel colors per algorithm
color_map = {
    'DDPGagdrp': '#ADD8E6',      # Light Blue
    'DDPGbaseline': '#FFD580',   # Light Orange
    'SACagdrp': '#90EE90',       # Light Green
    'SACbaseline': '#FF9999'     # Light Red
}

algo_names = []
last_values = []
std_values = []
colors = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        values = np.array(values)
        last_val = values[-1] /100           # last value divided by 40
        std = np.std(values, ddof=1) / 1000 # std divided by 1000
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        last_values.append(last_val)
        std_values.append(std)
        colors.append(color_map.get(algo_name, 'lightgray'))
    else:
        print(f"⚠️ No data for {algo_name} in {metric_name}")

plt.figure(figsize=(8, 5))
plt.bar(algo_names, last_values, yerr=std_values, capsize=5,
        color=colors, edgecolor='black')
plt.title(f'Last {metric_name} Value with Standard Deviation')
plt.ylabel(f'{metric_name}')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%

# ✅ Bar chart for Age_vehicle metric with last values and custom pastel colors,
# with Y-axis divided by 100 and formatted algorithm names, no error bars

metric_name = 'Age_vehicle'
algo_data = data.get(metric_name, {})

# Light pastel colors per algorithm
color_map = {
    'SACagdrp': '#ADD8E6',      # Light Blue SACagdrp
    'DDPGagdrp': '#FFD580',   # Light Orange DDPGagdrp
    'DDPGbaseline': '#90EE90',       # Light Green DDPGbaseline
    'SACbaseline': '#FF9999'     # Light Red SACbaseline
}

algo_names = []
last_values = []
colors = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        values = np.array(values)
        last_val = values[-1] / 40        # last value divided by 100
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        last_values.append(last_val)
        colors.append(color_map.get(algo_name, 'lightgray'))
    else:
        print(f"⚠️ No data for {algo_name} in {metric_name}")

plt.figure(figsize=(8, 5))
plt.bar(algo_names, last_values, 
        color=colors, edgecolor='black')
plt.title(f'Last {metric_name} Value')
plt.ylabel(f'{metric_name}')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


metric_name = 'Age_vehicle'
algo_data = data.get(metric_name, {})

color_map = {
    'SACagdrp': '#ADD8E6',      # Light Blue SACagdrp
    'DDPGagdrp': '#FFD580',     # Light Orange DDPGagdrp
    'DDPGbaseline': '#90EE90',  # Light Green DDPGbaseline
    'SACbaseline': '#FF9999'    # Light Red SACbaseline
}

algo_names = []
last_values = []
colors = []
error_bars = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 1:  # Need >1 for CI
        values = np.array(values) / 40  # Dividing all values by 40, consistent with last_val scaling
        last_val = values[-1]
        
        # Calculate 95% confidence interval for the last few samples (e.g., last 5 or all)
        # You can customize how many points to consider for CI:
        sample_size = min(5, len(values))  # last 5 values or less if not available
        sample = values[-sample_size:]
        
        mean = np.mean(sample)
        sem = stats.sem(sample)  # Standard error of mean
        ci = sem * stats.t.ppf((1 + 0.95) / 2, sample_size - 1)  # 95% CI
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        last_values.append(last_val)
        colors.append(color_map.get(algo_name, 'lightgray'))
        error_bars.append(ci)
    else:
        print(f"⚠️ No sufficient data for {algo_name} in {metric_name}")
           
plt.figure(figsize=(8, 5))
plt.bar(algo_names, last_values, 
        color=colors, edgecolor='black', yerr=error_bars, capsize=5)
plt.title(f'Last {metric_name} Value with 95% Confidence Interval')
plt.ylabel(f'{metric_name} (scaled)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

metric_name = 'TT_vehicle'
algo_data = data.get(metric_name, {})

color_map = {
    'SACagdrp': '#ADD8E6',      # Light Blue SACagdrp
    'DDPGagdrp': '#FFD580',     # Light Orange DDPGagdrp
    'DDPGbaseline': '#90EE90',  # Light Green DDPGbaseline
    'SACbaseline': '#FF9999'    # Light Red SACbaseline
}

algo_names = []
last_values = []
colors = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        values = np.array(values)    # scaling as before
        last_val = values[-1]
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        last_values.append(last_val)
        colors.append(color_map.get(algo_name, 'lightgray'))
    else:
        print(f"⚠️ No data for {algo_name} in {metric_name}")

plt.figure(figsize=(8, 5))
plt.bar(algo_names, last_values, color=colors, edgecolor='black')
plt.ylabel('Average Travel Times of CVs (s)')

plt.grid(axis='y')  # horizontal grid lines
plt.grid(axis='x', linestyle='--', alpha=0.5)  # vertical grid lines with dashed style and lighter opacity

plt.tight_layout()
#plt.savefig('AverageAoI.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

indices = [0, 250, 500, 999]
algos = list(algo_data.keys())

plt.figure(figsize=(10, 6))

for algo_name in algos:
    values = algo_data.get(algo_name, [])
    if values is not None and len(values) > max(indices):
        values = np.array(values)
        selected_values = values[indices]
        display_name = format_algo_name(algo_name)
        plt.plot(indices, selected_values, marker='o', label=display_name, color=color_map.get(algo_name, 'gray'))
    else:
        print(f"⚠️ Insufficient data for {algo_name} in {metric_name}")

plt.xlabel('Data Index')
plt.ylabel('Average Travel Times of CVs (s)')
plt.xticks(indices)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

indices = [0, 250, 500, 999]
algos = list(algo_data.keys())

# Collect values for each algorithm at the selected indices
values_matrix = []
for algo_name in algos:
    values = algo_data.get(algo_name, [])
    if values is not None and len(values) > max(indices):
        values = np.array(values)
        selected_values = values[indices]
    else:
        selected_values = [np.nan] * len(indices)  # or zeros if preferred
        print(f"⚠️ Insufficient data for {algo_name} in {metric_name}")
    values_matrix.append(selected_values)

values_matrix = np.array(values_matrix)  # shape: (num_algos, num_indices)

x = np.arange(len(indices))  # one position per index on x-axis
width = 0.15  # width of each bar
num_algos = len(algos)

plt.figure(figsize=(10, 6))

for i, algo_name in enumerate(algos):
    display_name = format_algo_name(algo_name)
    plt.bar(x + i*width - width*(num_algos-1)/2, values_matrix[i], width,
            label=display_name, color=color_map.get(algo_name, 'lightgray'), edgecolor='black')

plt.xticks(x, indices)
plt.xlabel('Data Index')
plt.ylabel('Average Travel Times of CVs (s)')
plt.grid(axis='y')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.legend(title='Algorithm')
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# metric_name = 'Age_vehicle'
# metric_name = 'TT_link'
metric_name = 'TT_vehicle'
algo_data = data.get(metric_name, {})

color_map = {
    'SACagdrp': '#ADD8E6',      # Light Blue SACagdrp
    'DDPGagdrp': '#FFD580',     # Light Orange DDPGagdrp
    'DDPGbaseline': '#90EE90',  # Light Green DDPGbaseline
    'SACbaseline': '#FF9999'    # Light Red SACbaseline
}

algo_names = []
last_values = []
colors = []

for algo_name, values in algo_data.items():
    if values is not None and len(values) > 0:
        values = np.array(values)    # scaling as before
        #last_val = values[-1]
        #mean_val = np.mean(values) /60 # age vehicle 
        # mean_val = values[-1] /60 # age vehicle 
        #mean_val = np.mean(values) /24/60 # tt link 
        # mean_val = values[-1] /24/60 # tt link 
        # mean_val = np.mean(values) / 6 # tt vehicle 
        mean_val = values[-1]  / 6 # tt vehicle 
        
        display_name = format_algo_name(algo_name)
        
        algo_names.append(display_name)
        last_values.append(mean_val)
        colors.append(color_map.get(algo_name, 'lightgray'))
    else:
        print(f"⚠️ No data for {algo_name} in {metric_name}")

plt.figure(figsize=(8, 5))
plt.bar(algo_names, last_values, color=colors, edgecolor='black')
# plt.ylabel('Average AoI of CVs (ms)', fontsize = 18)
# plt.ylabel('Average Travel Times of Roads (m)', fontsize = 18)
plt.ylabel('Average Travel Times of CVs (m)', fontsize = 18)
plt.ylim(bottom=30) # just for tt vehicle 
plt.grid(axis='y')  # horizontal grid lines
plt.grid(axis='x', linestyle='--', alpha=0.5)  # vertical grid lines with dashed style and lighter opacity
#plt.yscale('log')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig('agevehicle.pdf', format='pdf', bbox_inches='tight', dpi=300)
# plt.savefig('ttlink.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('ttvehicle.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()








