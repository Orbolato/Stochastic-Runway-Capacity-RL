###############################################################################
# Libraries and packages
###############################################################################

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import improved_dqn_agent as dqn
import improved_runway_env as env
from os import listdir
from os.path import isfile, join
import os
import scipy.stats as stats
import math

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
plt.savefig('rcces_gru.png')
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
data_filter = env.RunwayEnv(alpha = 2, delta = 0.3, data=data, 
                    training_days=days, u=u, u_max=u_max, v=v, v_max=v_max,
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
        state, reward, done, truncated, info = data_filter.step(action)
        
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

# days_2 = [171, 126, 11, 123, 184, 71, 86]

###############################################################################
# Reward function parameters and set of actions definition
###############################################################################

# Defining the arrival priority
alpha = 2.0#float(input('Enter the arrival priority (float number, default = 2.0): ')
              #or '2.0')

# Defining the cost of switching configurations
# print('\nIt may be important to define a delay cost factor - delta - of switching runways at time t.')
delta = 0.3#float(input('Enter the cost of switching runways (float number, default = 0.3): ')
              #or '0.3')

###############################################################################
# Hyperparameters, environment and RL agent definition
###############################################################################

# List of mean returns
G_list_1 = []

# List of epsilons
epsilon_list_1 = []

# List of test returns
test_list_1 = []

# Weight for the weighted average of returns and costs (0 < w <= 1)
w = 1 # (1e-12)**(1 / episodes)

# Number of episodes
episodes = 1_000_000

# Number of episodes for epsilon decay
decay_ep = 1_000_000

# Initial epsilon value for decaying dynamic epsilon-greedy policy (real)
initial_epsilon = 0.1

# Initial epsilon value for decaying dynamic epsilon-greedy policy (real)
final_epsilon = 0.1

# Step-size parameter for nonstationary and incremental implementation
# (between 0 and 1)
step_size = 0.1

# Discount rate for TD algorithms (between 0 and 1)
discount_rate = 1

# DQN architecture
n_features = 8 # 10 state features (RC_(t-1),a_t,d_t,day,wc_t,vsby_t,ceiling_t,ws_t,sknt_t,drct_t) and 3 action features (RC_t,u_t,v_t)
n_outputs = 1 # Q value
neurons = (n_features, 256, 256, n_outputs)

print("Neurons: " + str(neurons) + "\n")

# Neural Network Learning Rate (0 < float << 1)
lr = 1e-4

# Memory buffer size (steps)
mem_buffer_size = 100_000 # It includes each interation inside an episode

# Mini-batch size (steps)
mini_batch_size = 32

# Target network delay (episodes)
sync_delay = 1_000

# Testing delay (episodes)
test_delay = 10_000

# Observation low and high
# observation_low = [0,0,0,0,0,0.0,0.0,0,0.0,0.0]
# observation_high = [2,a_max,d_max,365,2,vsby_max,ceiling_max,3,sknt_max,360.0]
observation_low = [0,0,0,0,0]
observation_high = [2,a_max,d_max,2,3]
# in_scaler_min = observation_low + [0,0,0]
# in_scaler_max = observation_high + [2,a_max,d_max]

# DQN model
dqn_model_1 = None

# Creating environment
runway_1 = env.RunwayEnv(observation_low = observation_low, 
                        observation_high = observation_high,
                        data = data, training_days = training_days,
                        u=u, u_max=u_max, v=v, v_max=v_max,
                        possible_actions=possible_actions,
                        valid_bool=valid_bool, 
                        wind_states = wind_states,
                        alpha = 2.0, delta = 0.3)

# Creating Deep Agent
agent_1 = dqn.DQNAgent(layer_sizes = neurons,
                      # in_scaler_min = in_scaler_min, in_scaler_max = in_scaler_max,
                      step_size = step_size, discount_rate = discount_rate,
                      epsilon = initial_epsilon, final_epsilon = final_epsilon,
                      epsilon_decay = decay_ep, learning_rate = lr,
                      # dqn_model = dqn_model,
                      max_mem_buffer = mem_buffer_size,
                      batch_size = mini_batch_size 
                      )

###############################################################################
# Running algorithm
###############################################################################

ep = 0

# State-action value iteration
while ep < episodes:
    
    # Initializing mean G
    mean_G = 0
    
    for i in range(100):
    
        # Initializing return
        G = 0
        
        # Observing and initializing environment
        state, info = runway_1.reset()
        
        # Not done
        done = truncated = False
        
        # For each time period
        while not (done or truncated):
            
            # Current state
            curr_state = state
            
            # Determining the set of actions
            set_of_actions = runway_1.get_valid_actions()
    
            # Take an epsilon-greedy action over the environment
            action = agent_1.take_epsilon_greedy_action(curr_state, set_of_actions)
            
            # Take the action
            state, R, done, truncated, _ = runway_1.step(action.ravel())
            
            # Updating return
            G += R
            
            # Storing experience
            agent_1.store_exp(np.concatenate((curr_state, action)), R, state, set_of_actions, done or truncated)
        
        # Calculating mean G
        mean_G += G
        
        # Updating episodes
        ep += 1
    
    # Calculating mean G
    mean_G /= 100
        
    # Training DQN
    if agent_1.get_mem_len() >= mem_buffer_size:
        agent_1.train_model()
        
    # Synchronizing target network
    if ep % sync_delay == 0:
        agent_1.target_sync()
    
    # Testing
    if ep % test_delay == 0:
        
        total_difference = 0

        for day in test_days:
            
            # Compute single action using the model
            state, info = runway_1.reset(day=day) # Start environment
            model_reward = 0 # Initialize total reward
            done = False # Not done
            truncated = False # Not truncated
            
            # While day has not ended
            while not (done or truncated):
                
                # Determining the set of actions
                set_of_actions = runway_1.get_valid_actions()
    
                # Take an epsilon-greedy action over the environment
                action = agent_1.compute_single_action(state, set_of_actions)
                
                # Take the action and observe the state and reward
                state, R, done, truncated, _ = runway_1.step(action.ravel())
                
                # Updating return
                model_reward += R
                
            # Compute single action following a greedy policy of a naive agent
            state, info = runway_1.reset(day=day) # Test for the same day
            base_reward = 0 # Initialize test reward
            done = False # Not done
            truncated = False # Not truncated
            
            # While day has not ended
            while not (done or truncated):
                
                # Greedy policy of the naive agent
                action = runway_1.base_action()
                
                # Take the action and observe the state and reward
                state, R, done, truncated, info = runway_1.step(action)
                
                # Updating return
                base_reward += R
            
            total_difference += (model_reward - base_reward)
        
        print('Total difference (train minus test): ' + str(total_difference))
        
        # Storing total difference
        test_list_1.append(total_difference)
    
    # Storing information
    G_list_1.append(mean_G)
    epsilon_list_1.append(agent_1.get_epsilon())
    
    print('Episode ' + str(ep) 
          + ', G = ' + str(round(mean_G, 2)) 
          + ', epsilon = ' + str(agent_1.get_epsilon()))
    
    # End of episode
    agent_1.update_epsilon(ep)

model_list_1 = []
greedy_list_1 = []
base_list_1 = []

for day in test_days:
    
    # Compute single action using the model

    state, _ = runway_1.reset(day=day) # Start environment
    model_reward = 0 # Initialize total reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Determining the set of actions
        set_of_actions = runway_1.get_valid_actions()
    
        # Take an epsilon-greedy action over the environment
        action = agent_1.compute_single_action(state, set_of_actions)

        # Take the action and observe the state and reward
        state, R, done, truncated, _ = runway_1.step(action.ravel())
        
        # Updating return
        model_reward += R
    
    model_list_1.append(model_reward)
    
    # Compute single action using the model

    state, _ = runway_1.reset(day=day) # Start environment
    greedy_reward = 0 # Initialize total reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Determining the set of actions
        set_of_actions = runway_1.get_valid_actions()
    
        # Take an epsilon-greedy action over the environment
        action = runway_1.naive_action()

        # Take the action and observe the state and reward
        state, R, done, truncated, _ = runway_1.step(action.ravel())
        
        # Updating return
        greedy_reward += R
    
    greedy_list_1.append(greedy_reward)
        
    # Compute single action following a greedy policy of a naive agent

    state, _ = runway_1.reset(day=day) # Test for the same day
    base_reward = 0 # Initialize test reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Greedy policy of the naive agent
        action = runway_1.base_action()

        # Take the action and observe the state and reward
        state, reward, done, truncated, _ = runway_1.step(action)
        
        # Updating return
        base_reward += reward
    
    base_list_1.append(base_reward)
    
    print('Day: ' + day + 
          ', model reward: ' + str(model_reward) + 
          ', greedy reward: ' + str(greedy_reward) +
          ', baseline: ' + str(base_reward))

# Plotting results

print('Plotting results...')

plt.figure(dpi=1200)
plt.plot(G_list_1)
plt.xlabel('Steps')
plt.ylabel('Return')
plt.title('DB Approach 1')
plt.savefig('Results - DB Approach 1/dqn_db_1_training.eps', format='eps')
plt.show()

plt.figure(dpi=1200)
plt.plot(epsilon_list_1)
plt.xlabel('Steps')
plt.ylabel('Epsilon')
plt.title('DB Approach 1')
plt.savefig('Results - DB Approach 1/dqn_db_1_epsilon.eps', format='eps')
plt.show()

plt.figure(dpi=1200)
plt.plot(greedy_list_1,'g')
plt.plot(model_list_1,'r')
plt.plot(base_list_1,'b')
plt.xlabel('Days')
plt.ylabel('Return')
plt.title('DB Approach 1')
plt.savefig('Results - DB Approach 1/dqn_db_1_results.eps', format='eps')
plt.show()

plt.figure(dpi=1200)
plt.plot(test_list_1,'r')
plt.xlabel('Steps')
plt.ylabel('Return')
plt.title('Data-Based Approach 1')
plt.savefig('Results - DB Approach 1/dqn_db_1_learning.eps', format='eps')
plt.show()

plt.figure(dpi=1200)
greedy_improv_list = np.array(model_list_1) - np.array(greedy_list_1)
plt.hist(greedy_improv_list, bins=50)
# plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.xlabel('Return improvement with DQN')
plt.title('Data-Based Approach 1 - Greedy Policy')
plt.savefig('Results - DB Approach 1/dqn_db_1_greedy_improv.eps', format='eps')
plt.show()

plt.figure(dpi=1200)
base_improv_list = np.array(model_list_1) - np.array(base_list_1)
plt.hist(base_improv_list, bins=50)
# plt.xticks(rotation=45)
plt.ylabel('Frequency')
plt.xlabel('Return improvement with DQN')
plt.title('Data-Based Approach 1 - Baseline')
plt.savefig('Results - DB Approach 1/dqn_db_1_base_improv.eps', format='eps')
plt.show()

# Function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i-0.2,y[i]-40,y[i])

# Plotting mean results
plt.figure(dpi=1200)
x = ['Greedy Policy', 'DQN Policy', 'Base Policy']
y = [round(np.mean(greedy_list_1),2)] + [round(np.mean(model_list_1),2)] + [round(np.mean(base_list_1),2)]
plt.bar(x, y, color=['g','r','b'])
addlabels(x,y)
# plt.xticks(rotation=45)
plt.ylabel('Mean return')
plt.title('Data-Based Approach 1')
plt.savefig('Results - DB Approach 1/ppo_db_1_mean.eps', format='eps')
plt.show()

print('Done.\n')

# Saving DQN
dqn_model_1 = agent_1.get_dqn_model()

###############################################################################
# Hyperparameters, environment and RL agent definition
###############################################################################

# List of mean returns
G_list_2 = []

# List of epsilons
epsilon_list_2 = []

# List of test returns
test_list_2 = []

# Weight for the weighted average of returns and costs (0 < w <= 1)
w = 1 # (1e-12)**(1 / episodes)

# Number of episodes
episodes = 1_000_000

# Number of episodes for epsilon decay
decay_ep = 1_000_000

# Initial epsilon value for decaying dynamic epsilon-greedy policy (real)
initial_epsilon = 0.1

# Initial epsilon value for decaying dynamic epsilon-greedy policy (real)
final_epsilon = 0.1

# Step-size parameter for nonstationary and incremental implementation
# (between 0 and 1)
step_size = 0.1

# Discount rate for TD algorithms (between 0 and 1)
discount_rate = 1

# DQN architecture
n_features = 8 # 10 state features (RC_(t-1),a_t,d_t,day,wc_t,vsby_t,ceiling_t,ws_t,sknt_t,drct_t) and 3 action features (RC_t,u_t,v_t)
n_outputs = 1 # Q value
neurons = (n_features, 256, 256, n_outputs)

print("Neurons: " + str(neurons) + "\n")

# Neural Network Learning Rate (0 < float << 1)
lr = 1e-4

# Memory buffer size (steps)
mem_buffer_size = 100_000 # It includes each interation inside an episode

# Mini-batch size (steps)
mini_batch_size = 32

# Target network delay (episodes)
sync_delay = 1_000

# Testing delay (episodes)
test_delay = 10_000

# Observation low and high
# observation_low = [0,0,0,0,0,0.0,0.0,0,0.0,0.0]
# observation_high = [2,a_max,d_max,365,2,vsby_max,ceiling_max,3,sknt_max,360.0]
observation_low = [0,0,0,0,0]
observation_high = [2,a_max,d_max,2,3]
# in_scaler_min = observation_low + [0,0,0]
# in_scaler_max = observation_high + [2,a_max,d_max]

# DQN model
dqn_model_2 = None

# Creating environment
runway_2 = env.RunwayEnv(observation_low = observation_low,
                        observation_high = observation_high,
                        data = data, training_days = training_days_2,
                        u=u, u_max=u_max, v=v, v_max=v_max,
                        possible_actions=possible_actions,
                        valid_bool=valid_bool, 
                        wind_states = wind_states,
                        alpha = 2.0, delta = 0.3)

# Creating Deep Agent
agent_2 = dqn.DQNAgent(layer_sizes = neurons,
                      # in_scaler_min = in_scaler_min, in_scaler_max = in_scaler_max,
                      step_size = step_size, discount_rate = discount_rate,
                      epsilon = initial_epsilon, final_epsilon = final_epsilon,
                      epsilon_decay = decay_ep, learning_rate = lr,
                      # dqn_model = dqn_model,
                      max_mem_buffer = mem_buffer_size,
                      batch_size = mini_batch_size
                      )

# Incremental training: starting from the DB Approach 1 network
agent_2.model.load_state_dict(dqn_model_1.state_dict())
agent_2.target_sync()

###############################################################################
# Running algorithm
###############################################################################

ep = 0

# State-action value iteration
while ep < episodes:
    
    # Initializing mean G
    mean_G = 0
    
    for i in range(100):
    
        # Initializing return
        G = 0
        
        # Observing and initializing environment
        state, info = runway_2.reset()
        
        # Not done
        done = truncated = False
        
        # For each time period
        while not (done or truncated):
            
            # Current state
            curr_state = state
            
            # Determining the set of actions
            set_of_actions = runway_2.get_valid_actions()
    
            # Take an epsilon-greedy action over the environment
            action = agent_2.take_epsilon_greedy_action(curr_state, set_of_actions)
            
            # Take the action
            state, R, done, truncated, _ = runway_2.step(action.ravel())
            
            # Updating return
            G += R
            
            # Storing experience
            agent_2.store_exp(np.concatenate((curr_state, action)), R, state, set_of_actions, done or truncated)
        
        # Calculating mean G
        mean_G += G
        
        # Updating episodes
        ep += 1
    
    # Calculating mean G
    mean_G /= 100
        
    # Training DQN
    if agent_2.get_mem_len() >= mem_buffer_size:
        agent_2.train_model()
        
    # Synchronizing target network
    if ep % sync_delay == 0:
        agent_2.target_sync()
    
    # Testing
    if ep % test_delay == 0:
        
        total_difference = 0

        for day in test_days:
            
            # Compute single action using the model
            state, info = runway_2.reset(day=day) # Start environment
            model_reward = 0 # Initialize total reward
            done = False # Not done
            truncated = False # Not truncated
            
            # While day has not ended
            while not (done or truncated):
                
                # Determining the set of actions
                set_of_actions = runway_2.get_valid_actions()
    
                # Take an epsilon-greedy action over the environment
                action = agent_2.compute_single_action(state, set_of_actions)
                
                # Take the action and observe the state and reward
                state, R, done, truncated, _ = runway_2.step(action.ravel())
                
                # Updating return
                model_reward += R
                
            # Compute single action following a greedy policy of a naive agent
            state, info = runway_2.reset(day=day) # Test for the same day
            base_reward = 0 # Initialize test reward
            done = False # Not done
            truncated = False # Not truncated
            
            # While day has not ended
            while not (done or truncated):
                
                # Greedy policy of the naive agent
                action = runway_2.base_action()
                
                # Take the action and observe the state and reward
                state, R, done, truncated, info = runway_2.step(action)
                
                # Updating return
                base_reward += R
            
            total_difference += (model_reward - base_reward)
        
        print('Total difference (train minus test): ' + str(total_difference))
        
        # Storing total difference
        test_list_2.append(total_difference)
    
    # Storing information
    G_list_2.append(mean_G)
    epsilon_list_2.append(agent_2.get_epsilon())
    
    print('Episode ' + str(ep) 
          + ', G = ' + str(round(mean_G, 2)) 
          + ', epsilon = ' + str(agent_2.get_epsilon()))
    
    # End of episode
    agent_2.update_epsilon(ep)

model_list_2 = []
greedy_list_2 = []
base_list_2 = []

for day in test_days_2:
    
    # Compute single action using the model

    state, _ = runway_2.reset(day=day) # Start environment
    model_reward = 0 # Initialize total reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Determining the set of actions
        set_of_actions = runway_2.get_valid_actions()
    
        # Take an epsilon-greedy action over the environment
        action = agent_2.compute_single_action(state, set_of_actions)

        # Take the action and observe the state and reward
        state, R, done, truncated, _ = runway_2.step(action.ravel())
        
        # Updating return
        model_reward += R
    
    model_list_2.append(model_reward)
    
    # Compute single action using the model

    state, _ = runway_2.reset(day=day) # Start environment
    greedy_reward = 0 # Initialize total reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Determining the set of actions
        set_of_actions = runway_2.get_valid_actions()
    
        # Take an epsilon-greedy action over the environment
        action = runway_2.naive_action()

        # Take the action and observe the state and reward
        state, R, done, truncated, _ = runway_2.step(action.ravel())
        
        # Updating return
        greedy_reward += R
    
    greedy_list_2.append(greedy_reward)
        
    # Compute single action following a greedy policy of a naive agent

    state, _ = runway_2.reset(day=day) # Test for the same day
    base_reward = 0 # Initialize test reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Greedy policy of the naive agent
        action = runway_2.base_action()

        # Take the action and observe the state and reward
        state, reward, done, truncated, _ = runway_2.step(action)
        
        # Updating return
        base_reward += reward
    
    base_list_2.append(base_reward)
    
    print('Day: ' + day + 
          ', model reward: ' + str(model_reward) + 
          ', greedy reward: ' + str(greedy_reward) +
          ', baseline: ' + str(base_reward))

# Plotting results
plt.figure(dpi=1200)
plt.plot(greedy_list_2,'g')
plt.plot(model_list_2,'r')
plt.plot(base_list_2,'b')
# plt.xticks(rotation=45)
plt.legend(['Greedy','DQN','Baseline'])
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
x = ['Greedy Policy', 'DQN Policy', 'Base Policy']
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

# Saving DQN
dqn_model_2 = agent_2.get_dqn_model()