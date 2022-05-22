# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared_logic import *

random_seed = 1
np.random.seed(random_seed)

# %%
plt.style.use('seaborn-colorblind')
# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 11pt font in plots, to match 11pt font in document
    "axes.labelsize": 11,
    "font.size": 11
}
plt.rcParams.update(tex_fonts)

# %%
stations_df = pd.read_csv('./data/stations.csv')
stations_dict = stations_df.groupby(['common_id']).first().to_dict('index')

# %%
common_id = '36022-ie'
# common_id = 'auto-1003803'
tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
if not os.path.exists(tex_plots_path):
    os.makedirs(tex_plots_path)
if not os.path.exists(tex_table_path):
    os.makedirs(tex_table_path)

df = pd.read_parquet(
    f'./data/classified_raw/{common_id}_outliers_classified.parquet')

# %%
df.info()

# %%
print(df.describe().to_latex())
# pd.Styler.to_latex(df.describe())

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
bars = plt.bar(['regular', 'outlier'],
               df['is_outlier'].value_counts().to_numpy())
plt.margins(0.1, 0.15)
ax.bar_label(bars)
# plt.xlabel('Is an outlier?')
plt.ylabel('Count')
plt.title(f'Outlier class distribution of '
          f'{stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]}')
plt.grid(alpha=0.25, axis='y')
plt.savefig(f'{tex_plots_path}outlier_class_distribution_{common_id}.pdf',
            format='pdf', bbox_inches='tight')

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True,
                                    figsize=set_size('thesis'))
# fig.subplots_adjust(hspace=0.25)

ax1.set_title('All values')
ax1.boxplot(df['water_level'].to_numpy(), vert=True,
            flierprops={'marker': '.', 'markersize': 2})
ax1.set_ylabel('Water level')
# Hide x axis
ax1.get_xaxis().set_visible(False)
ax1.grid(alpha=0.25)

ax2.set_title('Regular values')
ax2.boxplot(df.loc[~df['is_outlier'], 'water_level'].to_numpy(), vert=True,
            flierprops={'marker': '.', 'markersize': 2})
ax2.set_ylabel('Water level')
# Hide x axis
ax2.get_xaxis().set_visible(False)
ax2.grid(alpha=0.25)

ax3.set_title('Outlier values')
ax3.boxplot(df.loc[df['is_outlier'], 'water_level'].to_numpy(), vert=True,
            flierprops={'marker': '.', 'markersize': 2})
ax3.set_ylabel('Water level')
# Hide x axis
ax3.get_xaxis().set_visible(False)
ax3.grid(alpha=0.25)

plt.suptitle(f'Boxplot of {stations_dict[common_id]["water_name"]} - '
             f'{stations_dict[common_id]["station_name"]}')
plt.setp((ax1, ax2, ax3), ylim=ax1.get_ylim())

plt.savefig(f'{tex_plots_path}boxplot_{common_id}.pdf', format='pdf',
            bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.boxplot(df['water_level'].to_numpy(), vert=False,
            flierprops={'marker': '.', 'markersize': 2})
plt.xlabel('Water level')
# Hide y axis
ax.get_yaxis().set_visible(False)
plt.title(f'Boxplot of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]} (all values)')
plt.grid(alpha=0.25)
plt.savefig(f'{tex_plots_path}boxplot_{common_id}_all.pdf', format='pdf',
            bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.boxplot(df.loc[~df['is_outlier'], 'water_level'].to_numpy(), vert=False,
            flierprops={'marker': '.', 'markersize': 2})
plt.xlabel('Water level')
# Hide y axis
ax.get_yaxis().set_visible(False)
plt.title(f'Boxplot of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]} (regular values)')
plt.grid(alpha=0.25)
plt.savefig(f'{tex_plots_path}boxplot_{common_id}_regular.pdf', format='pdf',
            bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.boxplot(df.loc[df['is_outlier'], 'water_level'].to_numpy(), vert=False,
            flierprops={'marker': '.', 'markersize': 2})
plt.xlabel('Water level')
# Hide y axis
ax.get_yaxis().set_visible(False)
plt.title(f'Boxplot of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]} (outlier values)')
plt.grid(alpha=0.25)
plt.savefig(f'{tex_plots_path}boxplot_{common_id}_outlier.pdf', format='pdf',
            bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.violinplot(df['water_level'].to_numpy(), vert=False, showmeans=True,
               showmedians=True, showextrema=True)
plt.xlabel('Water level')
# Hide y axis
ax.get_yaxis().set_visible(False)
plt.title(f'Violinplot of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]}')
plt.grid(alpha=0.25)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True,
                                    figsize=set_size('thesis', subplots=(3, 1)))
ax1.set_title('All Values')
ax1.hist(df['water_level'].to_numpy(), bins=100)
ax1.set_xlabel('Water level')
ax1.set_yscale('log')
ax1.set_ylabel('Count')
ax1.grid(alpha=0.25)

ax2.set_title('Regular values')
ax2.hist(df.loc[~df['is_outlier'], 'water_level'].to_numpy(), bins=100)
ax2.set_xlabel('Water level')
ax2.set_ylabel('Count')
ax2.set_yscale('log')
ax2.grid(alpha=0.25)

ax3.set_title('Outlier values')
ax3.hist(df.loc[df['is_outlier'], 'water_level'].to_numpy(), bins=100)
ax3.set_xlabel('Water level')
ax3.set_ylabel('Count')
ax3.set_yscale('log')
ax3.grid(alpha=0.25)

plt.suptitle(f'Histogram of the water Level of '
             f'{stations_dict[common_id]["water_name"]} - '
             f'{stations_dict[common_id]["station_name"]}')
plt.savefig(f'{tex_plots_path}water_level_histogram_{common_id}.pdf',
            format='pdf', bbox_inches='tight')

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.hist(df['water_level'].to_numpy(), bins=100)
plt.xlabel('Water level')
plt.ylabel('Count')
plt.title(f'Histogram of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]}')
plt.grid(alpha=0.25)
# plt.show()
plt.savefig(f'{tex_plots_path}histogram_{common_id}.pdf', format='pdf',
            bbox_inches='tight')

# %%
set_size('thesis', subplots=(3, 1))

# %%
df['water_level_diff'] = df['water_level'].diff()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True,
                                    figsize=set_size('thesis', subplots=(3, 1)))
ax1.set_title('All Values')
ax1.hist(df['water_level_diff'].to_numpy(), bins=100)
ax1.set_yscale('log')
ax1.set_xlabel('Water level delta to previous value')
ax1.set_ylabel('Count')
ax1.grid(alpha=0.25)

ax2.set_title('Regular values')
ax2.hist(df.loc[~df['is_outlier'], 'water_level'].diff().to_numpy(), bins=100)
ax2.set_yscale('log')
ax2.set_xlabel('Water level delta to previous value')
ax2.set_ylabel('Count')
ax2.grid(alpha=0.25)

ax3.set_title('Outlier values')
ax3.hist(df.loc[df['is_outlier'], 'water_level_diff'].to_numpy(), bins=100)
ax3.set_xlabel('Water level delta to previous value')
ax3.set_ylabel('Count')
ax3.set_yscale('log')
ax3.grid(alpha=0.25)

plt.suptitle(f'Histogram of the water level delta of '
             f'{stations_dict[common_id]["water_name"]} - '
             f'{stations_dict[common_id]["station_name"]}')
plt.savefig(f'{tex_plots_path}water_level_delta_histogram_{common_id}.pdf',
            format='pdf', bbox_inches='tight')

# %%
# get time difference between measurements in hours
df['timedelta'] = df['timestamp'].diff().astype('timedelta64[h]').astype(float)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True,
                                    figsize=set_size('thesis', subplots=(3, 1)))
ax1.set_title('All Values')
ax1.hist(df['timedelta'].to_numpy(), bins=100)
ax1.set_yscale('log')
ax1.set_xlabel('Timedelta to previous datapoint in hours')
ax1.set_ylabel('Count')
ax1.grid(alpha=0.25)

ax2.set_title('Regular values')
ax2.hist(df.loc[~df['is_outlier'], 'timedelta'].to_numpy(), bins=100)
ax2.set_yscale('log')
ax2.set_xlabel('Timedelta to previous datapoint in hours')
ax2.set_ylabel('Count')
ax2.grid(alpha=0.25)

ax3.set_title('Outlier values')
ax3.hist(df.loc[df['is_outlier'], 'timedelta'].to_numpy(), bins=100)
ax3.set_xlabel('Timedelta to previous datapoint in hours')
ax3.set_ylabel('Count')
ax3.grid(alpha=0.25)

plt.suptitle(f'Histogram of the timedelta between values of '
             f'{stations_dict[common_id]["water_name"]} - '
             f'{stations_dict[common_id]["station_name"]}')
plt.savefig(f'{tex_plots_path}time_delta_histogram_{common_id}.pdf',
            format='pdf', bbox_inches='tight')

# %%
from scipy.stats import kde

density = kde.gaussian_kde(df['water_level'].to_numpy())
x = np.linspace(np.floor(df['water_level'].min()),
                np.ceil(df['water_level'].max()), 300)
y = density(x)
plt.hist(df['water_level'], bins=100, density=True)
plt.plot(x, y)
plt.title("Density Plot of the data")
plt.show()

# %%
df.describe().to_latex(f'{tex_table_path}/{common_id}-7-number-summary-all.tex',
                       position='htp',
                       label=f'table:{common_id}-7-number-summary-all',
                       caption=f'Seven number summary of '
                               f'{stations_dict[common_id]["water_name"]} - '
                               f'{stations_dict[common_id]["station_name"]} '
                               f'(all values)')

# %%
df.loc[~df['is_outlier']].describe().to_latex(
    f'{tex_table_path}/{common_id}-7-number-summary-regular.tex',
    position='htp',
    label=f'table:{common_id}-7-number-summary-regular',
    caption=f'Seven number summary of {stations_dict[common_id]["water_name"]} '
            f'- {stations_dict[common_id]["station_name"]} (regular values)')

# %%
df.loc[df['is_outlier']].describe().to_latex(
    f'{tex_table_path}/{common_id}-7-number-summary-outlier.tex',
    position='htp',
    label=f'table:{common_id}-7-number-summary-outlier',
    caption=f'Seven number summary of {stations_dict[common_id]["water_name"]} '
            f'- {stations_dict[common_id]["station_name"]} (outlier values)')

# %%
df.loc[df['is_outlier']].describe().to_latex(
    f'{tex_table_path}/{common_id}-7-number-summary-outlier.tex',
    position='htp',
    label=f'table:{common_id}-7-number-summary-outlier',
    caption=f'Seven number summary of {stations_dict[common_id]["water_name"]} '
            f'- {stations_dict[common_id]["station_name"]} (outlier values)')

# %%
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.plot(df['timestamp'], df['water_level'], linewidth=0.5, zorder=-1)
# plt.scatter(df['timestamp'], df['water_level'], s=0.5)
plt.scatter(df.loc[~df['is_outlier'], 'timestamp'],
            df.loc[~df['is_outlier'], 'water_level'], s=0.5, c='C0',
            label='Regular')

plt.scatter(df.loc[df['is_outlier'], 'timestamp'],
            df.loc[df['is_outlier'], 'water_level'], s=0.5, c='C2',
            label='Outlier')

plt.gcf().autofmt_xdate()
plt.xlabel('Timestamp')
plt.ylabel('Water Level')
plt.title(f'Linechart of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]}')
plt.grid(alpha=0.25)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32))

# plt.show()
plt.savefig(f'{tex_plots_path}linechart_{common_id}.pdf', format='pdf',
            bbox_inches='tight')

# %%
from matplotlib.dates import DateFormatter

fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
start_date = '2019-10-16'
end_date = '2019-10-19'
df_slice = df.loc[
    (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
plt.plot(df_slice['timestamp'], df_slice['water_level'], linewidth=0.5,
         zorder=-1)
# plt.scatter(df['timestamp'], df['water_level'], s=0.5)
plt.scatter(df_slice.loc[~df_slice['is_outlier'], 'timestamp'],
            df_slice.loc[~df_slice['is_outlier'], 'water_level'], s=0.5, c='C0',
            label='Regular')

plt.scatter(df_slice.loc[df_slice['is_outlier'], 'timestamp'],
            df_slice.loc[df_slice['is_outlier'], 'water_level'], s=0.5, c='C2',
            label='Outlier')

ax.fmt_xdata = DateFormatter('%Y-%m-%d')
import matplotlib.dates as mdates

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.xlabel('Timestamp')
plt.ylabel('Water Level')
plt.title(f'Linechart of {stations_dict[common_id]["water_name"]} - '
          f'{stations_dict[common_id]["station_name"]}')
plt.grid(alpha=0.25)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.37))

# plt.show()
plt.savefig(f'{tex_plots_path}slice_linechart_{common_id}.pdf', format='pdf',
            bbox_inches='tight')
