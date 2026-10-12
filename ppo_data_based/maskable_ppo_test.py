from sb3_contrib import MaskablePPO
import improved_gym_capoptimizer as cp
# import runway_optimization_ga as ga
# import runway_optimization_nlp as nlp
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
import scipy.stats as stats

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Constraints:         1054
# Variables:           2400 (2400 integer)
# Auxiliaries:         2168 (2150 integer)

###############################################################################
# Function for day counter
###############################################################################

def day_of_the_year(day_string):
    
    day_str_split = day_string.split('-')
    year = int(day_str_split[0])
    month = int(day_str_split[1])
    day = int(day_str_split[2])
    
    if year % 4 == 0:
        if year % 100 == 0 and year % 400 != 0:
            month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
    counter = 0
    for i in range(month - 1):
        counter += month_days[i]
    counter += day - 1
    
    return counter

def previous_day(day_string):
    
    day_str_split = day_string.split('-')
    year = int(day_str_split[0])
    month = int(day_str_split[1])
    day = int(day_str_split[2])
    
    if year % 4 == 0:
        if year % 100 == 0 and year % 400 != 0:
            month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if day == 1:
        if month == 1:
            prev_year = year - 1
            prev_month = 12
            prev_day = 31
        else:
            prev_year = year
            prev_month = month - 1
            prev_day = month_days[prev_month - 1]
    else:
        prev_year = year
        prev_month = month
        prev_day = day - 1
    
    prev_day_string = str(prev_year) + '-' + str(prev_month) + '-' + str(prev_day)
    return prev_day_string

###############################################################################
# Runway Configuration Capacity Envelope (RCCE)
###############################################################################

def f(x,y,u): # Departures upper bound for the VMC curve
    v_ub = [] # Upper bound of departures v
    for i in range(len(u)): # For each arrival u_i
        for j in range(len(x)-1):  # For each intersecting point x_j
            if x[j] <= u[i] < x[j+1]: # If x_j <= u_i <= x_(j + 1)
                dy = y[j+1] - y[j]
                dx = x[j+1] - x[j]
                if dx == 0:
                    v_ub.append(y[j])
                else:
                    v_ub.append(dy/dx*(u[i] - x[j]) + y[j])
        if u[i] == x[-1]: # If u_i is the final point of the curve
            dy = y[-1] - y[-2]
            dx = x[-1] - x[-2]
            if dx == 0: # Vertical line
                v_ub.append(y[-2])
            else:
                v_ub.append(0)
        if u[i] > x[-1]: # If u_i is after the final point of the curve
            v_ub.append(0)
        if u[i] < 0: # If u_i is before the initial point of the curve
            v_ub.append(0)
    return np.array(v_ub, dtype=np.int64)

###############################################################################
# Data preprocessing
###############################################################################

# Repository root (parent folder of this script)
wd = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

# Importing data preprocessed in R
print('\nLoading METAR and traffic data...')
data = pd.read_csv(join(wd, 'data', 'data.csv'))
print('Data loaded.\n')

# Importing TAF data preprocessed in R
print('\nLoading TAF data...')
taf_data = pd.read_csv(join(wd, 'data', 'new_taf_list.csv'))
print('Data loaded.\n')

# Importing standard TAF data preprocessed in R
print('\nLoading standard TAF data...')
std_taf_data = pd.read_csv(join(wd, 'data', 'std_taf.csv'))
print('Data loaded.\n')

print('Removing partial periods...')
while(data.loc[0,'time'] != data.loc[0,'time'].replace(data.loc[0,'time'][-8:],'06:00:00')):
    data.drop(0, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
while(data.loc[len(data)-1,'time'] != data.loc[len(data)-1,'time'].replace(data.loc[len(data)-1,'time'][-8:],'05:45:00')):
    data.drop(len(data)-1, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
print('Done.\n')

print('Determining periods...')
data['day'] = data['time']
for row in range(len(data)):
    time_split = data.loc[row,'time'].split(' ')
    day_string = time_split[0].split('-')
    year = int(day_string[0])
    month = int(day_string[1])
    day = int(day_string[2])
    day_string = str(year) + '-' + str(month) + '-' + str(day)
    if len(time_split) > 1:
        hour = int(time_split[1][0:2])
    else:
        hour = 0
    if hour < 6:
      data.loc[row,'day'] = previous_day(day_string)
    else:
      data.loc[row,'day'] = day_string
print('Done.\n')

# Calculating demand error
data['arrival_error'] = data['arrival_demand_40'] - data['scheduled_arrivals_40']
data['arrival_error'] = data['arrival_error'].astype(int)
data['departure_error'] = data['departure_demand'] - data['scheduled_departures']
data['departure_error'] = data['departure_error'].astype(int)

# Error in function of scheduled arrivals
y1_mean = []
y1_std = []
data1 = []
for i in np.sort(data['scheduled_arrivals_40'].unique()):
    y1 = data.loc[data['scheduled_arrivals_40'] == i, 'arrival_error']
    data1.append(y1)
    y1_mean.append(np.mean(y1))
    y1_std.append(np.std(y1))
arr_mean = y1_mean
arr_std = y1_std
plt.figure(dpi=1200)
plt.plot(np.sort(data['scheduled_arrivals_40'].unique()), y1_mean)    
plt.plot(np.sort(data['scheduled_arrivals_40'].unique()), y1_std)    
plt.xlabel('Scheduled arrivals')
plt.ylabel('Error')
plt.legend(['Mean', 'Standard deviation'])
x1 = np.sort(data['scheduled_arrivals_40'].unique())
y1 = data['arrival_error']
plt.boxplot(data1, positions=x1)
plt.yticks(ticks=np.array(range(np.sort(y1.unique()).min(), np.sort(y1.unique()).max(), 3)))
plt.savefig('arr_variation.eps', format='eps')
plt.show()

# Error in function of scheduled arrivals
y2_mean = []
y2_std = []
data2 = []
for i in np.sort(data['scheduled_departures'].unique()):
    y2 = data.loc[data['scheduled_departures'] == i, 'departure_error']
    data2.append(y2)
    y2_mean.append(np.mean(y2))
    y2_std.append(np.std(y2))
dep_mean = y2_mean
dep_std = y2_std
plt.figure(dpi=1200)
plt.plot(np.sort(data['scheduled_departures'].unique()), y2_mean)    
plt.plot(np.sort(data['scheduled_departures'].unique()), y2_std)    
plt.xlabel('Scheduled departures')
plt.ylabel('Error')
plt.legend(['Mean', 'Standard deviation'])
x2 = np.sort(data['scheduled_departures'].unique())
y2 = data['departure_error']
plt.boxplot(data2, positions=x2)
plt.yticks(ticks=np.array(range(np.sort(y2.unique()).min(), np.sort(y2.unique()).max(), 3)))
plt.savefig('dep_variation.eps', format='eps')
plt.show()

# Plotting histograms
y1 = data['arrival_error']
plt.figure(dpi=1200)
plt.hist(y1, density=True, bins=np.sort(data['arrival_error'].unique()))
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Arrival Error')

# Plotting normal distribution
mu = np.average(data['arrival_error'])
sigma = np.std(data['arrival_error'])
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.legend(['Normal Distribution','Histogram'])
plt.savefig('arr_dist.eps', format='eps')
plt.show()

# Plotting histograms
y2 = data['departure_error']
plt.figure(dpi=1200)
plt.hist(y2, density=True, bins=np.sort(data['arrival_error'].unique()))
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Departure Error')

# Plotting normal distribution
mu = np.average(data['departure_error'])
sigma = np.std(data['departure_error'])
x = np.linspace(mu - 5*sigma, mu + 5*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))

plt.legend(['Normal Distribution','Histogram'])
plt.savefig('dep_dist.eps', format='eps')
plt.show()

# Arrivals and departures columns
while taf_data.loc[0, 'time'] != data.loc[0, 'time']:
    taf_data.drop(0, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
    std_taf_data.drop(0, axis='rows', inplace=True)
    std_taf_data.reset_index(drop=True, inplace=True)
while taf_data.loc[taf_data.shape[0]-1, 'time'] != data.loc[data.shape[0]-1, 'time']:
    taf_data.drop(taf_data.shape[0]-1, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
    std_taf_data.drop(std_taf_data.shape[0]-1, axis='rows', inplace=True)
    std_taf_data.reset_index(drop=True, inplace=True)
taf_data['arrival_demand_40'] = std_taf_data['arrival_demand_40'] = data['scheduled_arrivals_40']
taf_data['departure_demand'] = std_taf_data['departure_demand'] = data['scheduled_departures']

###############################################################################
# Encoding categorical features
###############################################################################

# Preprocessing data
print('Encoding runway configurations and weather condition...')
data['weather_condition'] = data['ifr'] + data['lifr']
data['10R_10L'] = np.zeros(len(data))
data['28L_28R'] = np.zeros(len(data))
for i in range(len(data)):
    if data.loc[i,'configuration'] == '10R|10L':
        data.loc[i,'configuration'] = 1 # '10R_10L'
        data.loc[i,'10R_10L'] = 1
    elif data.loc[i,'configuration'] == '28L|28R':
        data.loc[i,'configuration'] = 2 # '28L_28R'
        data.loc[i,'28L_28R'] = 1
    else:
        data.loc[i,'configuration'] = 0 # None
        
    if data.loc[i,'base_configuration'] == '10R|10L':
        data.loc[i,'base_configuration'] = 1 # '10R_10L'
    elif data.loc[i,'base_configuration'] == '28L|28R':
        data.loc[i,'base_configuration'] = 2 # '28L_28R'
    else:
        data.loc[i,'base_configuration'] = 0 # None
print('Done.\n')

# Defining wind states
print('Defining and encoding wind states...')
cw_max = 20 # Maxium crosswind (knots)
tw_max = 5 # Maximum tailwind (knots)
print('Crosswind limit of ' + str(cw_max) + ' knots.')
print('Tailwind limit of ' + str(tw_max) + ' knots.\n')
data['wind_state'] = np.zeros(len(data)) # Column of wind states
data['wind_state'] = pd.to_numeric(data['wind_state'], downcast='integer')
data['10R_10L_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
data['28L_28R_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
# 0 - Both available (0 is always for all runways available)
# 1 - 10R|10L available
# 2 - 28L|28R available
# 3 - Both unavailable
print('Case 0 - Both available.')
print('Case 1 - 10R|10L available.')
print('Case 2 - 28L|28R available.')
print('Case 3 - Both unavailable.\n')
for i in range(len(data)):
    if abs(data.loc[i,'crosswind_10']) > cw_max:
        data.loc[i,'wind_state'] = 3
        data.loc[i,'10R_10L_OK'] = 0
        data.loc[i,'28L_28R_OK'] = 0
    elif data.loc[i,'headwind_10'] < -tw_max:
        data.loc[i,'wind_state'] = 2
        data.loc[i,'10R_10L_OK'] = 0
        data.loc[i,'28L_28R_OK'] = 1
    elif data.loc[i,'headwind_10'] > tw_max:
        data.loc[i,'wind_state'] = 1
        data.loc[i,'10R_10L_OK'] = 1
        data.loc[i,'28L_28R_OK'] = 0
    else:
        data.loc[i,'wind_state'] = 0
        data.loc[i,'10R_10L_OK'] = 1
        data.loc[i,'28L_28R_OK'] = 1

for k in data['wind_state'].unique():
    print('Case ' + str(k) + ' has ' 
          + str(len(data.loc[data['wind_state'] == k,]))
          + ' occurences.')
print('')

###############################################################################
# Runway Configuration Capacity Envelopes (RCCEs)
###############################################################################

# Importing capacity envelopes
print('Generating capacity envelopes...')
points = {0: {}, 1: {}, 2: {}} # 0 - VMC, 1 - IMC, 2 - LIMC
legend = {0: {}, 1: {}, 2: {}}
folder = join(wd, 'curves', '')
files = [f for f in listdir(folder) if isfile(join(folder, f))]
for file in files:
    if file[:3] == 'VMC':
        wc = 0
        if file[4:11] == '28L_28R':
            p = 0.99
        else:
            p = 0.99
    else:
        if file[:3] == 'IMC':
            wc = 1
            if file[4:11] == '28L_28R':
                p = 0.95
            else:
                p = 0.95
        else:
            wc = 2
            if file[4:11] == '28L_28R':
                p = 0.9
            else:
                p = 0.9
    if file[-8:] == '{:.4f}'.format(p).replace('0.','') + '.csv':
        name = file.replace('_RCCE_' + '{:.4f}'.format(p).replace('0.','') + '.csv','')
        curve = pd.read_csv(folder + file)
        print(name)
        if name[-7:] == '10R_10L':
            rc = 1
        elif name[-7:] == '28L_28R':
            rc = 2
        points[wc][rc] = curve
        legend[wc][rc] = name
print('Done.\n')

# Plotting capacity envelopes
legends = []
print('Plotting VMC and IMC capacity envelopes...')
plt.figure(dpi=1200)
for wc in points:
    for rc in points[wc]:
        plt.plot(points[wc][rc]['x'],points[wc][rc]['y'])
        plt.title('RCCEs')
        legends.append(legend[wc][rc])
plt.legend(legends)
plt.xlabel('Arrivals')
plt.ylabel('Departures')
plt.savefig('rcces_gru.eps', format='eps')
plt.show()
print('Done.\n')

###############################################################################
# Maximum features' values
###############################################################################

# Total number of time slots
N = len(data)

# Calculating accumulated arrivals and departures
print('Calculating accumulated arrivals and departures per day...')
accumulated_arrivals = [data.loc[0,'arrival_demand_40']]
accumulated_departures = [data.loc[0,'departure_demand']]
day = data.loc[0,'day']
days = [day]
for i in range(1,N):
    if data.loc[i,'day'] == day:
        accumulated_arrivals[-1] += data.loc[i,'arrival_demand_40']
        accumulated_departures[-1] += data.loc[i,'departure_demand']
    else:
        accumulated_arrivals.append(data.loc[i,'arrival_demand_40'])
        accumulated_departures.append(data.loc[i,'departure_demand'])
        day = data.loc[i,'day']
        days.append(day)
accumulated_info = pd.DataFrame(np.array([days, accumulated_arrivals, accumulated_departures], dtype=object).T, columns=['day', 'accumulated_arrivals', 'accumulated_departures'])
a_max = 50#max(accumulated_info['accumulated_arrivals'])
d_max = 50#max(accumulated_info['accumulated_departures'])
print('Maximum accumulated arrivals: ' + str(a_max))
print('Maximum accumulated departures: ' + str(d_max) + '\n')

# Calculating maximum visibility, ceiling and wind speed
# vsby_max = np.max(data['vsby'])
# ceiling_max = np.max(data['ceiling'])
# sknt_max = np.max(data['sknt'])

###############################################################################
# Set of actions
###############################################################################

print('Creating curve points for each weather condition and runway configuration...')
A = 1 # '10R_10L'
B = 2 # '28L_28R'
C = 0 # None
configs = [C, A, B] 
weather_conds = [0, 1, 2] # ['VMC', 'IMC', 'LIMC'] # VMC - 0, IMC - 1, LIMC - 2
wind_states = {0: [A,B], 1: [A], 2: [B], 3: [C]}
u = {}
v = {}
u_max = 0
v_max = 0
for wc in weather_conds:
    for rc in configs:
        if rc == C:
            u[wc, rc] = np.array([0])
            v[wc, rc] = np.array([0])
        else:
            u[wc, rc] = np.array([i for i in range(np.max(points[wc][rc]['x']) + 1)])
            v[wc, rc] = np.array(f(np.array(points[wc][rc]['x']),
                                   np.array(points[wc][rc]['y']),
                                   u[wc, rc]), dtype=np.int64)
        if np.max(u[wc, rc]) > u_max:
            u_max = np.max(u[wc, rc])
        if np.max(v[wc, rc]) > v_max:
            v_max = np.max(v[wc, rc])
print('Done.\n')

# All possible actions

print('\nCreating the set of possible actions...')

possible_actions = [] # Possible actions list
for rc in configs:
    if rc == C:
        possible_actions.append([rc,0,0])
    else:
        for arr in u[0, rc]:
            for dep in range(v[0, rc][arr] + 1):
                possible_actions.append([rc, arr, dep])
                
# possible_actions = np.array(possible_actions)

print('Done.\n')

# Counting actions

print('Counting actions...')

max_state_action_counter = 0

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            state_action_counter += 1
                    state_action_counter += 1
                    if state_action_counter > max_state_action_counter:
                        max_state_action_counter = state_action_counter
                        
print('Done.\n')

# Determining the set of actions

print('Creating set of valid actions...')

keys = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                max_state_action_counter),
                -1,
                dtype='int')

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            action = [rc, arr, dep]
                            keys[ws, wc, a_i, d_i, state_action_counter] = possible_actions.index(action)
                            state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    action = [rc, arr_lim, dep]
                    keys[ws, wc, a_i, d_i, state_action_counter] = possible_actions.index(action)
                    state_action_counter += 1

possible_actions = np.array(possible_actions)

valid_bool = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                len(possible_actions)),
                False,
                dtype='bool')

for ws in range(keys.shape[0]):
    for wc in range(keys.shape[1]):
        for a_i in range(keys.shape[2]):
            for d_i in range(keys.shape[3]):
                for state_action_counter in range(keys.shape[4]):
                    idx = keys[ws, wc, a_i, d_i, state_action_counter]
                    if idx != -1:
                        valid_bool[ws, wc, a_i, d_i][idx] = True
        
print('All done.\n')

###############################################################################
# Filtering congested data
###############################################################################

print('Filtering congested data...')

# Observation low and high
# observation_low = [0,0,0,0,0,0.0,0.0,0,0.0,0.0]
# observation_high = [2,a_max,d_max,365,2,vsby_max,ceiling_max,3,sknt_max,360.0]
observation_low = [0,0,0,0,0]
observation_high = [2,a_max,d_max,2,3]

# Environment for data filtering
data_filter = cp.RunwayEnv(alpha = 2, delta = 0.3, data=data, 
                    training_days=days, 
                    u=u, u_max=u_max, v=v, v_max=v_max,
                    possible_actions=possible_actions,
                    valid_bool=valid_bool,
                    wind_states=wind_states, 
                    observation_low=observation_low, 
                    observation_high=observation_high)

congested_days = [] # Initializing list of congested days

for day in days: # For each day of the entire dataset

    print(day)
    
    # Compute single action following a greedy policy of a naive agent
    state, info = data_filter.reset(day=day) # Test for the same day
    test_reward = 0 # Initialize test reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Greedy policy of the naive agent
        action = data_filter.base_action()
        
        # Take the action and observe the state and reward
        state, reward, done, truncated, info = data_filter.base_step(action)
        
        # Updating return
        test_reward += reward

    # Store congested days
    if test_reward <= -10:
        congested_days.append(day)
    
# Filtering
data = data[data.day.isin(congested_days)].reset_index(drop=True)

print('Total of ' + str(len(congested_days)) + ' days.\n')

###############################################################################
# Creating datasets
###############################################################################

# Cutting time window and splitting data
# print('Splitting data into training and test datasets...')
# n_days = len(congested_days) # Total number of days to be considered
# split_ratio = 0.7 # Percentage of the dataset for training
# n_days = min(n_days, len(congested_days))
# sampled_days = random.sample(congested_days,k=n_days)
# training_days = random.sample(sampled_days, k=int(split_ratio*n_days))
# test_days = list(np.array(sampled_days)[~np.isin(sampled_days,training_days)])
# test_days_2 = random.sample(test_days,k=70)
# training_days_2 = list(np.array(congested_days)[~np.isin(congested_days,test_days_2)])
# test_days_3 = random.sample(test_days_2,k=7)
# training_days_3 = list(np.array(congested_days)[~np.isin(congested_days,test_days_3)])
# test_days_4 = random.sample(test_days_3,k=1)
# training_days_4 = list(np.array(congested_days)[~np.isin(congested_days,test_days_4)])
# print('Dataset split into ' 
#       + str(len(training_days)) + ' days for training and '
#       + str(len(test_days)) + ' days for testing.')

test_days = ['2023-9-10',
 '2023-12-6',
 '2022-9-19',
 '2022-5-29',
 '2023-6-1',
 '2023-6-30',
 '2023-1-29',
 '2023-11-15',
 '2022-8-2',
 '2023-11-1',
 '2023-1-30',
 '2022-8-17',
 '2023-12-7',
 '2023-1-5',
 '2022-3-18',
 '2023-11-25',
 '2023-3-9',
 '2022-10-14',
 '2023-3-7',
 '2022-9-2',
 '2023-10-7',
 '2023-2-6',
 '2022-5-23',
 '2023-3-23',
 '2023-5-19',
 '2022-2-17',
 '2023-5-3',
 '2023-11-4',
 '2022-1-16',
 '2022-7-29',
 '2023-2-1',
 '2023-3-21',
 '2023-8-8',
 '2023-5-24',
 '2022-8-20',
 '2023-8-19',
 '2023-4-28',
 '2023-8-10',
 '2022-8-1',
 '2023-10-28',
 '2022-10-4',
 '2023-2-15',
 '2022-11-24',
 '2022-11-4',
 '2022-10-29',
 '2022-10-30',
 '2022-11-20',
 '2023-8-11',
 '2022-12-7',
 '2023-9-23',
 '2023-6-29',
 '2023-2-7',
 '2022-2-28',
 '2023-6-21',
 '2022-9-13',
 '2022-10-19',
 '2023-9-18',
 '2023-1-1',
 '2022-6-12',
 '2022-1-8',
 '2023-7-9',
 '2023-1-3',
 '2022-12-2',
 '2023-10-18',
 '2022-12-22',
 '2022-12-9',
 '2023-2-10',
 '2023-4-30',
 '2022-11-19',
 '2022-8-6',
 '2023-8-3',
 '2022-12-30',
 '2022-5-10',
 '2023-12-2',
 '2023-7-17',
 '2023-10-27',
 '2022-11-12',
 '2023-8-24',
 '2023-7-21',
 '2022-11-26',
 '2023-11-23',
 '2023-12-5',
 '2023-4-12',
 '2022-7-14',
 '2022-5-27',
 '2023-6-11',
 '2023-5-12',
 '2022-11-10',
 '2022-7-9',
 '2022-5-4',
 '2022-3-22',
 '2023-5-9',
 '2022-10-8',
 '2023-9-1',
 '2023-10-8',
 '2023-11-18',
 '2023-12-13',
 '2023-3-25',
 '2022-10-17',
 '2023-12-11',
 '2022-8-24',
 '2022-6-17',
 '2022-9-26',
 '2022-12-6',
 '2022-11-1',
 '2022-11-3',
 '2023-10-15',
 '2023-9-4',
 '2022-11-16',
 '2023-10-30',
 '2022-3-30',
 '2022-1-6',
 '2023-8-16',
 '2023-3-12',
 '2022-2-16',
 '2023-6-14',
 '2022-8-3',
 '2022-4-17',
 '2022-11-7',
 '2022-10-1',
 '2023-3-31',
 '2022-11-25',
 '2023-9-16',
 '2023-9-15',
 '2023-1-18',
 '2023-12-21',
 '2023-1-12',
 '2023-10-22',
 '2023-6-17',
 '2022-1-4',
 '2023-10-20',
 '2022-10-28',
 '2022-6-4',
 '2022-3-25',
 '2023-4-8',
 '2022-11-22',
 '2022-4-30',
 '2022-3-24',
 '2022-2-26',
 '2022-12-28',
 '2022-4-25',
 '2022-7-15',
 '2023-11-30',
 '2022-3-12',
 '2022-1-28',
 '2023-2-13',
 '2023-5-23',
 '2023-2-9',
 '2022-6-9',
 '2022-5-26',
 '2022-4-23',
 '2023-8-14',
 '2022-11-27',
 '2023-8-6',
 '2022-5-16',
 '2023-12-22',
 '2022-2-11',
 '2022-4-29',
 '2022-5-19',
 '2022-9-17',
 '2022-7-24',
 '2023-7-6',
 '2022-10-23',
 '2022-4-21',
 '2023-9-25',
 '2022-3-11',
 '2022-10-21',
 '2023-11-21',
 '2022-7-2',
 '2023-12-17',
 '2023-1-24',
 '2023-8-5',
 '2022-5-3',
 '2022-12-29',
 '2023-4-26',
 '2022-3-15',
 '2022-5-28',
 '2023-2-28',
 '2022-6-15',
 '2022-6-10',
 '2022-4-14',
 '2023-1-14',
 '2022-1-24',
 '2022-9-30',
 '2022-12-10',
 '2022-1-13',
 '2023-12-12',
 '2022-9-24',
 '2023-3-14',
 '2022-6-13',
 '2023-6-8',
 '2022-4-4',
 '2023-3-13',
 '2022-7-7',
 '2022-1-2',
 '2022-9-6',
 '2022-8-11',
 '2023-2-5',
 '2023-11-8',
 '2023-8-20',
 '2023-5-25',
 '2022-4-1',
 '2023-5-30',
 '2023-5-28',
 '2023-9-22',
 '2023-5-8',
 '2022-9-27',
 '2022-6-29',
 '2023-11-5']

training_days = list(np.array(congested_days)[~np.isin(congested_days,test_days)])

test_days_2 = ['2023-8-5',
 '2023-1-12',
 '2022-8-17',
 '2023-9-15',
 '2022-12-10',
 '2022-12-30',
 '2023-5-12']

training_days_2 = list(np.array(congested_days)[~np.isin(congested_days,test_days_2)])

days_2 = [171, 126, 11, 123, 184, 71, 86]

###############################################################################
# Greedy policy
###############################################################################

# Introucing greedy policy information
greedy_env = cp.RunwayEnv(alpha = 2, delta = 0.3, data=data, 
                   training_days=congested_days, 
                   u=u, u_max=u_max, v=v, v_max=v_max, 
                   possible_actions=possible_actions, 
                   valid_bool=valid_bool,
                   wind_states=wind_states, 
                   observation_low=observation_low, 
                   observation_high=observation_high)#,
                   #render_mode='human')

data['greedy_configuration'] = data['base_configuration']
data['greedy_arrivals_40'] = data['base_arrivals_40']
data['greedy_departures'] = data['base_departures']
greedy_list = []

for day in test_days:
        
    # Greedy policy
    obs, info = greedy_env.reset(day=day)
    greedy_rew = 0
    counter = 0
    terminated = False
    while not terminated:
        action = greedy_env.naive_action()
        action_array = greedy_env.possible_actions[action]
        idx = data[data['day'] == day].iloc[counter].name
        data.loc[idx, 'greedy_configuration'] = action_array[0]
        data.loc[idx, 'greedy_arrivals_40'] = action_array[1]
        data.loc[idx, 'greedy_departures'] = action_array[2]
        obs, reward, terminated, truncated, info = greedy_env.step(action)
        greedy_rew += reward
        counter += 1
    print('Day: ' + day + ', greedy reward: ' + str(round(greedy_rew,1)))
    greedy_list.append(greedy_rew)

###############################################################################
# Running Algorithm
###############################################################################

print('Starting training process...')
start = time.time()
env = cp.RunwayEnv(alpha = 2, delta = 0.3, data=data, 
                   training_days=training_days, 
                   u=u, u_max=u_max, v=v, v_max=v_max, 
                   possible_actions=possible_actions, 
                   valid_bool=valid_bool,
                   wind_states=wind_states, 
                   observation_low=observation_low, 
                   observation_high=observation_high)#,
                   #render_mode='human')

# del model
model = MaskablePPO("MlpPolicy", env, gamma=1, verbose=1)
# prev_model = MaskablePPO.load(join(wd, 'trained_models', 'ppo', 'db_approach_1', 'ppo_model_db_1'), 
#                           env=env) # Load model
# model.set_parameters(prev_model.get_parameters())
model.learn(1_000_000)
end = time.time()
time_span_1 = end - start

print('Training finished in ' + str(round(time_span_1, 2)) + ' seconds.')

model.save("Results - DB Approach 1/ppo_model_db_1") # Save model

# model = MaskablePPO.load(join(wd, 'trained_models', 'ppo', 'db_approach_1', 'ppo_model_db_1'), env=env) # Load model

# Testing
opt_list_1 = []
model_list_1 = []
base_list_1 = []
# total_difference = 0
# threshold = 2

for day in test_days:

    # Compute single action using the model
    obs, info = env.reset(day=day)
    model_rew = 0
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action_idx, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        # action = env.possible_actions[action_idx]
        # RC_i = int(obs[0]) # Previous RC (during t-1)
        # a_i = int(obs[1]) # Arrival queue during t (including previous queue and demand)
        # d_i = int(obs[2]) # Departure queue during t (including previous queue and demand)
        # RC_f = action[0] # Runway configuration during t
        # a_f = a_i - action[1] # Arrival queue in the end of t
        # d_f = d_i - action[2] # Departure queue in the end of t
        # est_model_rew = env.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f)
        # greedy_action_idx = env.naive_action()
        # greedy_action = env.possible_actions[greedy_action_idx]
        # RC_i = int(obs[0]) # Previous RC (during t-1)
        # a_i = int(obs[1]) # Arrival queue during t (including previous queue and demand)
        # d_i = int(obs[2]) # Departure queue during t (including previous queue and demand)
        # RC_f = action[0] # Runway configuration during t
        # a_f = a_i - action[1] # Arrival queue in the end of t
        # d_f = d_i - action[2] # Departure queue in the end of t
        # est_greedy_rew = env.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f)
        # print('Greedy reward: ' + str(est_greedy_rew))
        # print(greedy_action_idx)
        # print('Model reward: ' + str(est_model_rew))
        # print(action_idx)
        # if est_model_rew < (1 + threshold) * est_greedy_rew:
        #     print('Final reward: ' + str(est_greedy_rew))
        #     action_idx = greedy_action_idx
        #     print(action_idx)
        # else:
        #     print('Final reward: ' + str(est_model_rew))
        #     print(action_idx)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        model_rew += reward
        
    # Greedy policy
    # obs, info = env.reset(day=day)
    # model.arrivals_served.setub(max_possible_arrivals)
    # greedy_rew = 0
    # terminated = False
    # while not terminated:
    #     action = env.naive_action()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     model.arrivals_served.setub(max_possible_arrivals)
    #     greedy_rew += reward
    
    # Solve the problem using MINLP
    # opt_res = opt_env.optimize_day(day=day, verbose=True, use_warm_start=True)
    # opt_rew = -opt_res['objective']

    # Greedy policy reward (the GA benchmark is computed afterwards in calculations.py)
    opt_rew = greedy_list[test_days.index(day)]
    
    # Compute single action following the base policy
    obs, info = env.reset(day=day)
    base_rew = 0
    terminated = False
    while not terminated:
        action = env.base_action()
        obs, reward, terminated, truncated, info = env.base_step(action)
        base_rew += reward

    # total_difference += (model_rew - greedy_rew)
    
    base_list_1.append(base_rew)
    opt_list_1.append(opt_rew)
    model_list_1.append(model_rew)
    
    print('Day: ' + str(day) 
          + ', model reward: ' + str(round(model_rew,2)) 
           + ', greedy policy reward: ' + str(round(opt_rew,2))
          + ', baseline policy reward: ' + str(round(base_rew,2)))
    
# print('Total difference (model minus greedy): ' + str(round(total_difference,2)))
print('Training finished in ' + str(round(time_span_1, 2)) + ' seconds.')

# Plotting results
plt.figure(dpi=1200)
plt.plot(opt_list_1,'g')
plt.plot(model_list_1,'r')
plt.plot(base_list_1,'b')
# plt.xticks(rotation=45)
plt.legend(['Optimal','PPO','Baseline'])
# plt.legend(['RL Model','Baseline'])
plt.xlabel('Test Days')
plt.ylabel('Total return')
plt.title('Data-Based Approach 1')
# plt.savefig('Results - DB Approach 1/ppo_db_1_results.eps', format='eps')
plt.show()

# Plotting results
plt.figure(dpi=1200)
plt.plot(opt_list_1,'g')
plt.plot(model_list_1,'r')
# plt.xticks(rotation=45)
plt.legend(['Optimal','PPO'])
# plt.legend(['RL Model','Baseline'])
plt.xlabel('Test Days')
plt.ylabel('Total return')
plt.title('Data-Based Approach 1')
# plt.savefig('Results - DB Approach 1/ppo_db_1_results_2.eps', format='eps')
plt.show()

# Function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i-0.2,y[i]-40,y[i])

# Plotting mean results
plt.figure(dpi=1200)
x = ['Optimal Policy', 'PPO Policy', 'Baseline Policy']
y = np.round([np.mean(opt_list_1)] + [np.mean(model_list_1)] + [np.mean(base_list_1)],2)
plt.bar(x, y, color=['g','r','b'])
addlabels(x,y)
# plt.xticks(rotation=45)
plt.ylabel('Mean return')
plt.title('Data-Based Approach 1')
# plt.savefig('Results - DB Approach 1/ppo_db_1_mean.eps', format='eps')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
base_improv_list = np.array(model_list_1) - np.array(base_list_1)
plt.hist(base_improv_list, bins=50)
# plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.xlabel('Return difference PPO vs Baseline')
plt.title('Data-Based Approach 1 - Baseline')
# plt.savefig('Results - DB Approach 1/ppo_db_1_base_improv.eps', format='eps')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
opt_improv_list = np.array(model_list_1) - np.array(opt_list_1)
plt.hist(opt_improv_list, bins=50)
# plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.xlabel('Return difference PPO vs Optimal')
plt.title('Data-Based Approach 1 - Optimal Policy')
# plt.savefig('Results - DB Approach 1/ppo_db_1_greedy_improv.eps', format='eps')
plt.show()

# Plotting results for specific days
plt.figure(dpi=1200)
plt.plot(np.array(opt_list_1)[days_2],'g')
plt.plot(np.array(model_list_1)[days_2],'r')
plt.plot(np.array(base_list_1)[days_2],'b')
# plt.xticks(rotation=45)
plt.legend(['Optimal','PPO','Baseline'])
# plt.legend(['RL Model','Baseline'])
plt.xlabel('Test Days')
plt.ylabel('Total return')
plt.title('Data-Based Approach 1')
# plt.savefig('Results - DB Approach 1/ppo_db_1_results_specific.eps', format='eps')
plt.show()

# Function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i-0.2,y[i]-35,y[i])

# Plotting mean results
plt.figure(dpi=1200)
x = ['Optimal Policy', 'PPO Policy', 'Base Policy']
y = ([round(np.mean(np.array(opt_list_1)[days_2]),2)] + 
    [round(np.mean(np.array(model_list_1)[days_2]),2)] + 
    [round(np.mean(np.array(base_list_1)[days_2]),2)])
plt.bar(x, y, color=['g','r','b'])
addlabels(x,y)
# plt.xticks(rotation=45)
plt.ylabel('Mean return')
plt.title('Data-Based Approach 1')
# plt.savefig('Results - DB Approach 1/ppo_db_1_mean_specific.eps', format='eps')
plt.show()

###############################################################################
# Running Algorithm
###############################################################################

print('Starting training process...')
start = time.time()
env = cp.RunwayEnv(alpha = 2, delta = 0.3, data=data, 
                   training_days=training_days_2, 
                   u=u, u_max=u_max, v=v, v_max=v_max, 
                   possible_actions=possible_actions,
                   valid_bool=valid_bool,
                   wind_states=wind_states, 
                   observation_low=observation_low, 
                   observation_high=observation_high)
# del model
# model = MaskablePPO("MlpPolicy", env, gamma=1, verbose=1)
# prev_model = MaskablePPO.load(join(wd, 'trained_models', 'ppo', 'db_approach_2', 'ppo_model_db_2'), 
#                           env=env) # Load model
# model.set_parameters(prev_model.get_parameters())
model.set_env(env) # Incremental training: DB Approach 1 model with the DB Approach 2 training days
model.learn(1_000_000)
end = time.time()
time_span_2 = end - start

print('Training finished in ' + str(round(time_span_2, 2)) + ' seconds.')

model.save("Results - DB Approach 2/ppo_model_db_2") # Save model

# Testing
greedy_list_2 = []
model_list_2 = []
base_list_2 = []
# total_difference = 0
# threshold = 0.1

for day in test_days_2:

    # Compute single action using the model
    obs, info = env.reset(day=day)
    model_rew = 0
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action_idx, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        # action = env.possible_actions[action_idx]
        # RC_i = int(obs[0]) # Previous RC (during t-1)
        # a_i = int(obs[1]) # Arrival queue during t (including previous queue and demand)
        # d_i = int(obs[2]) # Departure queue during t (including previous queue and demand)
        # RC_f = action[0] # Runway configuration during t
        # a_f = a_i - action[1] # Arrival queue in the end of t
        # d_f = d_i - action[2] # Departure queue in the end of t
        # est_model_rew = env.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f)
        # greedy_action_idx = env.naive_action()
        # greedy_action = env.possible_actions[greedy_action_idx]
        # RC_i = int(obs[0]) # Previous RC (during t-1)
        # a_i = int(obs[1]) # Arrival queue during t (including previous queue and demand)
        # d_i = int(obs[2]) # Departure queue during t (including previous queue and demand)
        # RC_f = action[0] # Runway configuration during t
        # a_f = a_i - action[1] # Arrival queue in the end of t
        # d_f = d_i - action[2] # Departure queue in the end of t
        # est_greedy_rew = env.compute_reward(RC_i, RC_f, a_i, d_i, a_f, d_f)
        # print('Greedy reward: ' + str(est_greedy_rew))
        # print(greedy_action_idx)
        # print('Model reward: ' + str(est_model_rew))
        # print(action_idx)
        # if est_model_rew < (1 - threshold) * est_greedy_rew:
        #     print('Final reward: ' + str(est_greedy_rew))
        #     action_idx = greedy_action_idx
        #     print(action_idx)
        # else:
        #     print('Final reward: ' + str(est_model_rew))
        #     print(action_idx)
        obs, reward, terminated, truncated, info = env.step(action_idx)
        model_rew += reward
        
    # Compute single action following a greedy policy of a naive agent
    obs, info = env.reset(day=day)
    greedy_rew = 0
    terminated = False
    while not terminated:
        action = env.naive_action()
        obs, reward, terminated, truncated, info = env.step(action)
        greedy_rew += reward
    
    # Compute single action following the base policy
    obs, info = env.reset(day=day)
    base_rew = 0
    terminated = False
    while not terminated:
        action = env.base_action()
        obs, reward, terminated, truncated, info = env.base_step(action)
        base_rew += reward

    # total_difference += (model_rew - greedy_rew)
    
    base_list_2.append(base_rew)
    greedy_list_2.append(greedy_rew)
    model_list_2.append(model_rew)
    
    print('Day: ' + day
           + ', model reward: ' + str(round(model_rew,2)) 
           + ', greedy policy reward: ' + str(round(greedy_rew,2))
           + ', baseline policy reward: ' + str(round(base_rew,2)))
    
# print('Total difference (model minus greedy): ' + str(round(total_difference,2)))
print('Training finished in ' + str(round(time_span_2, 2)) + ' seconds.')

# Plotting results
plt.figure(dpi=1200)
plt.plot(greedy_list_2,'g')
plt.plot(model_list_2,'r')
plt.plot(base_list_2,'b')
# plt.xticks(rotation=45)
plt.legend(['Greedy','PPO','Baseline'])
# plt.legend(['RL Model','Baseline'])
plt.xlabel('Test Days')
plt.ylabel('Total return')
plt.title('Data-Based Approach 2')
plt.savefig('Results - DB Approach 2/ppo_db_2_results.eps', format='eps')
plt.show()

# Function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i-0.2,y[i]-35,y[i])

# Plotting mean results
plt.figure(dpi=1200)
x = ['Greedy Policy', 'PPO Policy', 'Base Policy']
y = ([round(np.mean(np.array(greedy_list_2)),2)] + 
    [round(np.mean(np.array(model_list_2)),2)] + 
    [round(np.mean(np.array(base_list_2)),2)])
plt.bar(x, y, color=['g','r','b'])
addlabels(x,y)
# plt.xticks(rotation=45)
plt.ylabel('Mean return')
plt.title('Data-Based Approach 2')
plt.savefig('Results - DB Approach 2/ppo_db_2_mean.eps', format='eps')
plt.show()

