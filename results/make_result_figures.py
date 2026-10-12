import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
# Reproduces the result figures of the paper (Figs. 13-19) from the per-day
# total rewards stored in returns_209_test_days.csv and returns_7_test_days.csv
###############################################################################

wd = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(wd, 'figures')

# Fixed EPS creation date, so that regenerated figures are identical to the committed ones
os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
os.makedirs(out, exist_ok=True)

df_209 = pd.read_csv(os.path.join(wd, 'returns_209_test_days.csv'))
df_7 = pd.read_csv(os.path.join(wd, 'returns_7_test_days.csv'))

# Colors used in the paper
colors = {'GA': 'g', 'PPO': 'b', 'DQN': 'orange', 'Baseline': 'r'}

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(out, name + '.png'), dpi=300)
    plt.savefig(os.path.join(out, name + '.eps'), format='eps')
    plt.close()

def plot_lines(series, title, name):
    # Total reward per test day
    plt.figure()
    for label, values in series.items():
        plt.plot(np.asarray(values), color=colors[label], label=label)
    plt.legend()
    plt.xlabel('Test Days')
    plt.ylabel('Total reward')
    plt.title(title)
    save(name)

def plot_bars(series, title, name):
    # Mean total reward per policy, with value labels
    plt.figure()
    labels = list(series)
    means = [round(float(np.mean(series[k])), 2) for k in labels]
    plt.bar(labels, means, color=[colors[k] for k in labels])
    for i, m in enumerate(means):
        plt.text(i, m, m, ha='center', va='top')
    plt.ylabel('Mean total reward')
    plt.title(title)
    save(name)

###############################################################################
# Fig. 13 - DB Approach 1, 209 test days
###############################################################################

s = {'GA': df_209['GA'], 'PPO': df_209['PPO_DB1'],
     'DQN': df_209['DQN_DB1'], 'Baseline': df_209['Baseline']}
plot_lines(s, 'DB Approach 1 - 209 test days', 'fig13a_db1_209_days')
plot_bars(s, 'DB Approach 1 - 209 test days', 'fig13b_db1_209_days_mean')

###############################################################################
# Fig. 14 - DB Approach 1, 7 test days of DB/FB Approach 2
###############################################################################

s = {'GA': df_7['GA'], 'PPO': df_7['PPO_DB1'],
     'DQN': df_7['DQN_DB1'], 'Baseline': df_7['Baseline']}
plot_lines(s, 'DB Approach 1 - 7 test days', 'fig14a_db1_7_days')
plot_bars(s, 'DB Approach 1 - 7 test days', 'fig14b_db1_7_days_mean')

###############################################################################
# Fig. 15 - DB Approach 2
###############################################################################

s = {'GA': df_7['GA'], 'PPO': df_7['PPO_DB2'],
     'DQN': df_7['DQN_DB2'], 'Baseline': df_7['Baseline']}
plot_lines(s, 'DB Approach 2', 'fig15a_db2')
plot_bars(s, 'DB Approach 2', 'fig15b_db2_mean')

###############################################################################
# Fig. 16 - FB Approach 1 (PPO only)
###############################################################################

s = {'GA': df_7['GA'], 'PPO': df_7['PPO_FB1'], 'Baseline': df_7['Baseline']}
plot_lines(s, 'FB Approach 1', 'fig16a_fb1')
plot_bars(s, 'FB Approach 1', 'fig16b_fb1_mean')

###############################################################################
# Fig. 17 - FB Approach 2
###############################################################################

s = {'GA': df_7['GA'], 'PPO': df_7['PPO_FB2'],
     'DQN': df_7['DQN_FB2'], 'Baseline': df_7['Baseline']}
plot_lines(s, 'FB Approach 2', 'fig17a_fb2')
plot_bars(s, 'FB Approach 2', 'fig17b_fb2_mean')

###############################################################################
# Fig. 18 - DB Approach 1 without the baseline
###############################################################################

plot_lines({'GA': df_209['GA'], 'PPO': df_209['PPO_DB1'], 'DQN': df_209['DQN_DB1']},
           'DB Approach 1 - 209 test days', 'fig18a_db1_209_days_no_baseline')
plot_lines({'GA': df_7['GA'], 'PPO': df_7['PPO_DB1'], 'DQN': df_7['DQN_DB1']},
           'DB Approach 1 - 7 test days', 'fig18b_db1_7_days_no_baseline')

###############################################################################
# Fig. 19 - DB Approach 2 vs FB Approach 2 without the baseline
###############################################################################

plot_lines({'GA': df_7['GA'], 'PPO': df_7['PPO_DB2'], 'DQN': df_7['DQN_DB2']},
           'DB Approach 2', 'fig19a_db2_no_baseline')
plot_lines({'GA': df_7['GA'], 'PPO': df_7['PPO_FB2'], 'DQN': df_7['DQN_FB2']},
           'FB Approach 2', 'fig19b_fb2_no_baseline')

print('Figures saved in ' + out)
