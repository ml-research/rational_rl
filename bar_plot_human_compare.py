import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


sns.set_context('paper')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

path = os.getcwd()
df_human = pd.read_csv(path+'/scores_tables/scores_table_human_normalized.csv')
df_human = df_human[df_human.game != 'Skiing']
df_human["max_both"] = df_human[["SPAU", "PAUS"]].max(axis=1)
df_human = df_human.sort_values('max_both', ascending = False)
df_human.to_csv('scores/csv/human_normalized_final_scores.csv')
print(df_human)

sns.set(font_scale=2)
f, ax = plt.subplots(figsize = (7,11))
palette = sns.color_palette('deep')


sns.barplot(x = 'PAUS', y = 'game', data = df_human,
            label = 'RN', color = palette[1], edgecolor = 'w')
sns.barplot(x = 'SPAU', y = 'game', data = df_human,
            label = 'RRN', color = palette[0], edgecolor = 'w')
#sns.set_color_codes('muted')
chart = sns.barplot(x = 'LreLU', y = 'game', data = df_human,
            label = 'LReLU', color = palette[2], edgecolor = 'w')

ax.legend(ncol = 3, loc = 'upper left', bbox_to_anchor=(-0.01, 1.08), fontsize=20) #, bbox_to_anchor=(0., -1.11)

ax.axvline(100, ls='--', color="darkred")


intervall = len(ax.patches)//3
new_value= .5
for patch in ax.patches[-intervall*2:-intervall]:
    current_height = patch.get_height()
    diff = current_height - new_value

    # we change the bar width
    patch.set_height(new_value)

    # we recenter the bar
    patch.set_y(patch.get_y() + diff * .5)
new_value = .15
for patch in ax.patches[-intervall:]:
    current_height = patch.get_height()
    diff = current_height - new_value

    # we change the bar width
    patch.set_height(new_value)

    # we recenter the bar
    patch.set_y(patch.get_y() + diff * .5)
ax.set(xlabel='Human normalized score [\%]', ylabel='', xscale="log")
#sns.despine(left = True, bottom = True)
# plt.show()
plt.savefig(path+'/images/score_bar_plot_h.pdf', dpi=300, bbox_inches='tight')
