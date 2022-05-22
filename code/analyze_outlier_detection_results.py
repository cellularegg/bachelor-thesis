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


def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


# %%


def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


# %%


def get_TP_TN_FP_FN(actual, predicted):
    if actual == True and predicted == True:
        return 'TP'
    elif actual == False and predicted == False:
        return 'TN'
    elif actual == False and predicted == True:
        return 'FP'
    elif actual == True and predicted == False:
        return 'FN'
    else:
        raise ValueError(
            f'Invalid actual and predicted values: {actual} {predicted}')


# %%


stations_df = pd.read_csv('./data/stations.csv')
stations_dict = stations_df.groupby(['common_id']).first().to_dict('index')
common_id = '36022-ie'

tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
if not os.path.exists(tex_plots_path):
    os.makedirs(tex_plots_path)
if not os.path.exists(tex_table_path):
    os.makedirs(tex_table_path)

prediction_summary_df = pd.read_parquet(
    './data/predictions/predictions_summary.parquet')
preprocessed_prediction_summary_df = pd.read_parquet(
    './data/predictions/predictions_preprocessed_summary.parquet')
prediction_summary_df = prediction_summary_df.loc[
    prediction_summary_df['model_type'] != 'delta-z-score']
prediction_summary_df['recall'] = prediction_summary_df.apply(
    lambda row: calculate_recall(row['tp'], row['fn']),
    axis=1)
prediction_summary_df['precision'] = prediction_summary_df.apply(
    lambda row: calculate_precision(row['tp'], row['fp']),
    axis=1)
preprocessed_prediction_summary_df[
    'recall'] = preprocessed_prediction_summary_df.apply(
    lambda row: calculate_recall(row['tp'], row['fn']), axis=1)
preprocessed_prediction_summary_df[
    'precision'] = preprocessed_prediction_summary_df.apply(
    lambda row: calculate_precision(row['tp'], row['fp']), axis=1)

# %%


prediction_summary_df.info()

# %%


prediction_summary_df['model_type'].unique()

# %%


preprocessed_prediction_summary_df.info()

# %%


prediction_summary_df.groupby(['common_id', 'model_type']).any().reset_index()[
    ['common_id', 'model_type']]

# %%


# get unique combination of common id and model type
# create a dictionary of dataframes for faster filtering
prediction_summaries_dict = {}
for idx, row in \
prediction_summary_df.groupby(['common_id', 'model_type']).any().reset_index()[
    ['common_id', 'model_type']].iterrows():
    prediction_summaries_dict[f'{row["common_id"]}'] = {
        'regular': prediction_summary_df.loc[
            prediction_summary_df['common_id'] == row['common_id']].copy(),
        'preprocessed': preprocessed_prediction_summary_df.loc[
            preprocessed_prediction_summary_df['common_id'] == row[
                'common_id']].copy()
    }

# %%


common_id = '36022-ie'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['regular'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['regular']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
combined_df[
    ['window_size', 'center_window', 'normalized', 'threshold', 'model_type',
     'f1_score']].to_latex(
    f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
    label=f'table:{common_id}-top-predictions-summary', index=False,
    caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '36022-ie'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['preprocessed'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['preprocessed']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
# combined_df[['window_size', 'center_window', 'normalized', 'threshold', 'model_type', 'f1_score']].to_latex(
#     f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
#     label=f'table:{common_id}-top-predictions-summary', index=False,
#     caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '2386-ch'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['regular'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['regular']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
combined_df[
    ['window_size', 'center_window', 'normalized', 'threshold', 'model_type',
     'f1_score']].to_latex(
    f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
    label=f'table:{common_id}-top-predictions-summary', index=False,
    caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '2386-ch'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['preprocessed'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['preprocessed']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
# combined_df[['window_size', 'center_window', 'normalized', 'threshold', 'model_type', 'f1_score']].to_latex(
#     f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
#     label=f'table:{common_id}-top-predictions-summary', index=False,
#     caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '2720050000-de'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['regular'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['regular']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
combined_df[
    ['window_size', 'center_window', 'normalized', 'threshold', 'model_type',
     'f1_score']].to_latex(
    f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
    label=f'table:{common_id}-top-predictions-summary', index=False,
    caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '2720050000-de'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['preprocessed'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['preprocessed']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
# combined_df[['window_size', 'center_window', 'normalized', 'threshold', 'model_type', 'f1_score']].to_latex(
#     f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
#     label=f'table:{common_id}-top-predictions-summary', index=False,
#     caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '39003-ie'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['regular'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['regular']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
combined_df[
    ['window_size', 'center_window', 'normalized', 'threshold', 'model_type',
     'f1_score']].to_latex(
    f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
    label=f'table:{common_id}-top-predictions-summary', index=False,
    caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '39003-ie'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['preprocessed'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['preprocessed']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
# combined_df[['window_size', 'center_window', 'normalized', 'threshold', 'model_type', 'f1_score']].to_latex(
#     f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
#     label=f'table:{common_id}-top-predictions-summary', index=False,
#     caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '42960105-de'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['regular'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['regular']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
combined_df[
    ['window_size', 'center_window', 'normalized', 'threshold', 'model_type',
     'f1_score']].to_latex(
    f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
    label=f'table:{common_id}-top-predictions-summary', index=False,
    caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


common_id = '42960105-de'
tex_table_path = f'../bachelor-thesis/tables/{common_id}/'
model_types = prediction_summaries_dict[common_id]['preprocessed'][
    'model_type'].unique()
combined_df_lst = []
for model_type in model_types:
    tmp_df = prediction_summaries_dict[common_id]['preprocessed']
    norm_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (tmp_df['normalized'])]
    combined_df_lst.append(
        norm_df.sort_values(by=['f1_score'], ascending=False).head(3))
    reg_df = tmp_df.loc[
        (tmp_df['model_type'] == model_type) & (~tmp_df['normalized'])]
    combined_df_lst.append(
        reg_df.sort_values(by=['f1_score'], ascending=False).head(3))
combined_df = pd.concat(combined_df_lst).sort_values(
    by=['f1_score', 'model_type'], ascending=False)
# combined_df[['window_size', 'center_window', 'normalized', 'threshold', 'model_type', 'f1_score']].to_latex(
#     f'{tex_table_path}/{common_id}-top-predictions-summary.tex', position='htp',
#     label=f'table:{common_id}-top-predictions-summary', index=False,
#     caption=f'Top predictions summary of {stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}')
combined_df

# %%


# create a dataframe with overall best parameters
combined_df = None
tex_table_path = f'../bachelor-thesis/tables/'
for common_id in prediction_summaries_dict.keys():
    if combined_df is None:
        combined_df = prediction_summaries_dict[common_id]['regular'][
            ['window_size', 'center_window', 'normalized', 'threshold',
             'model_type', 'f1_score', 'tn', 'fp', 'fn',
             'tp', 'precision', 'recall']].rename(
            columns={'f1_score': f'{common_id}-f1_score',
                     'tn': f'{common_id}-tn',
                     'fp': f'{common_id}-fp', 'fn': f'{common_id}-fn',
                     'tp': f'{common_id}-tp',
                     'precision': f'{common_id}-precision',
                     'recall': f'{common_id}-recall'})
    else:
        combined_df = combined_df.merge(
            prediction_summaries_dict[common_id]['regular'][
                ['window_size', 'center_window', 'normalized', 'threshold',
                 'model_type',
                 'f1_score', 'tn', 'fp', 'fn', 'tp', 'precision',
                 'recall']].rename(
                columns={'f1_score': f'{common_id}-f1_score',
                         'tn': f'{common_id}-tn',
                         'fp': f'{common_id}-fp', 'fn': f'{common_id}-fn',
                         'tp': f'{common_id}-tp',
                         'precision': f'{common_id}-precision',
                         'recall': f'{common_id}-recall'}),
            on=['window_size', 'center_window', 'normalized', 'threshold',
                'model_type'],
            how='outer')
    # break
combined_df['average_f1_score'] = (combined_df['2386-ch-f1_score'] +
                                   combined_df['2720050000-de-f1_score'] +
                                   combined_df['36022-ie-f1_score'] +
                                   combined_df['39003-ie-f1_score'] +
                                   combined_df[
                                       '42960105-de-f1_score']) / 5
combined_df['harmonic_mean_f1_score'] = 5 / (
            1 / combined_df['2386-ch-f1_score'] + 1 / combined_df[
        '2720050000-de-f1_score'] +
            1 / combined_df['36022-ie-f1_score'] + 1 / combined_df[
                '39003-ie-f1_score'] + 1 / combined_df[
                '42960105-de-f1_score'])

combined_df.sort_values(by=['average_f1_score'], ascending=False).head(5)[
    ['window_size', 'center_window', 'normalized', 'threshold', 'model_type',
     'average_f1_score']].to_latex(
    f'{tex_table_path}/top-avg-predictions-summary.tex', position='htp',
    label=f'table:top-avg-predictions-summary', index=False,
    caption=f'Best parameters of the average F1-score of all stations tested')
combined_df.sort_values(by=['average_f1_score'], ascending=False).head(5)
# combined_df.info()
# %%


best_pred = \
prediction_summary_df.sort_values(by=['f1_score'], ascending=False).iloc[0]
common_id = best_pred['common_id']
model_type = best_pred['model_type']
window_size = int(best_pred['window_size']) if best_pred[
                                                   'window_size'] is not None else None
center_window = 'cw' if best_pred['center_window'] else 'ncw'
normalized = 'normalized' if best_pred['normalized'] else 'regular'
threshold = best_pred['threshold']
best_pred_df = pd.read_parquet(
    f'./data/predictions/raw_preprocessed/{normalized}/{common_id}/{window_size}_{center_window}_{model_type}.parquet')
best_pred_df.info()
best_pred_df['y_pred'] = np.where(best_pred_df['result'] > threshold, 1, 0)
best_pred_df['pred_type'] = best_pred_df.apply(
    lambda row: get_TP_TN_FP_FN(row['is_outlier'], row['y_pred']), axis=1)
# best_pred_df = best_pred_df.loc[best_pred_df['timestamp'] <= '2019-06-01']
tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.plot(best_pred_df['timestamp'], best_pred_df['result'], linewidth=0.5,
         zorder=-1)
# plt.scatter(best_pred_df.loc[~best_pred_df['is_outlier'], 'timestamp'],
#             best_pred_df.loc[~best_pred_df['is_outlier'], 'result'], s=0.1, label='regular (ground truth)')
# plt.scatter(best_pred_df.loc[best_pred_df['is_outlier'], 'timestamp'],
#             best_pred_df.loc[best_pred_df['is_outlier'], 'result'], s=0.5, c='C2', label='outliers (ground truth)')
plt.scatter(best_pred_df.loc[best_pred_df['pred_type'] == 'TN', 'timestamp'],
            best_pred_df.loc[best_pred_df['pred_type'] == 'TN', 'result'],
            s=0.5, c='C0', label='True negative')
plt.scatter(best_pred_df.loc[best_pred_df['pred_type'] == 'TP', 'timestamp'],
            best_pred_df.loc[best_pred_df['pred_type'] == 'TP', 'result'],
            s=0.5, c='C1', label='True positive')
plt.scatter(best_pred_df.loc[best_pred_df['pred_type'] == 'FP', 'timestamp'],
            best_pred_df.loc[best_pred_df['pred_type'] == 'FP', 'result'],
            s=0.5, c='C2', label='False positive')
plt.scatter(best_pred_df.loc[best_pred_df['pred_type'] == 'FN', 'timestamp'],
            best_pred_df.loc[best_pred_df['pred_type'] == 'FN', 'result'],
            s=0.5, c='C3', label='False negative')

plt.suptitle(
    f'Result of {model_type} based outlier detection of '
    f'{stations_dict[common_id]["water_name"]} - {stations_dict[common_id]["station_name"]}',
    y=1.05)
plt.title(
    f'(window size={window_size}, {"no " if not best_pred["center_window"] else ""} center window, {"not " if best_pred["normalized"] else ""} normalized , threshold={round(threshold, 2)})')
plt.grid(alpha=0.25)
plt.gcf().autofmt_xdate()
plt.xlabel('Timestamp')
plt.ylabel('Result ($|x_t - \hat{x}_t|$)')
plt.axhline(y=threshold, color='C3', linestyle='--', linewidth=0.5,
            label='Threshold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32))
plt.savefig(f'{tex_plots_path}od_result_{model_type}_{common_id}_all.pdf',
            format='pdf',
            bbox_inches='tight')

# %%


worst_best_pred = None
for common_id in prediction_summaries_dict.keys():
    if worst_best_pred is None:
        worst_best_pred = \
        prediction_summaries_dict[common_id]['regular'].sort_values(
            by=['f1_score'], ascending=False).iloc[0]
    elif \
    prediction_summaries_dict[common_id]['regular'].sort_values(by=['f1_score'],
                                                                ascending=True).iloc[
        0]['f1_score'] >= worst_best_pred['f1_score']:
        worst_best_pred = \
        prediction_summaries_dict[common_id]['regular'].sort_values(
            by=['f1_score'], ascending=False).iloc[0]

common_id = worst_best_pred['common_id']
model_type = worst_best_pred['model_type']
window_size = int(worst_best_pred['window_size']) \
    if worst_best_pred['window_size'] is not None else None
center_window = 'cw' if worst_best_pred['center_window'] else 'nocw'
normalized = 'normalized' if worst_best_pred['normalized'] else 'regular'
threshold = worst_best_pred['threshold']
worst_best_df = pd.read_parquet(
    f'./data/predictions/raw_preprocessed/{normalized}/{common_id}/'
    f'{window_size}_{center_window}_{model_type}.parquet')
worst_best_df.info()
worst_best_df['y_pred'] = np.where(worst_best_df['result'] > threshold, 1, 0)
worst_best_df['pred_type'] = worst_best_df.apply(
    lambda row: get_TP_TN_FP_FN(row['is_outlier'], row['y_pred']), axis=1)

tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
plt.plot(worst_best_df['timestamp'], worst_best_df['result'], linewidth=0.5,
         zorder=-1)
plt.scatter(worst_best_df.loc[worst_best_df['pred_type'] == 'TN', 'timestamp'],
            worst_best_df.loc[worst_best_df['pred_type'] == 'TN', 'result'],
            s=0.5, c='C0', label='True negative')
plt.scatter(worst_best_df.loc[worst_best_df['pred_type'] == 'TP', 'timestamp'],
            worst_best_df.loc[worst_best_df['pred_type'] == 'TP', 'result'],
            s=0.5, c='C1', label='True positive')
plt.scatter(worst_best_df.loc[worst_best_df['pred_type'] == 'FP', 'timestamp'],
            worst_best_df.loc[worst_best_df['pred_type'] == 'FP', 'result'],
            s=0.5, c='C2', label='False positive')
plt.scatter(worst_best_df.loc[worst_best_df['pred_type'] == 'FN', 'timestamp'],
            worst_best_df.loc[worst_best_df['pred_type'] == 'FN', 'result'],
            s=0.5, c='C3', label='False negative')
plt.suptitle(f'Result of {model_type} based outlier detection of '
             f'{stations_dict[common_id]["water_name"]} - '
             f'{stations_dict[common_id]["station_name"]}', y=1.05)
plt.title(f'(window size={window_size}, '
          f'{"no " if not worst_best_pred["center_window"] else ""} center '
          f'window, {"not " if worst_best_pred["normalized"] else ""} '
          f'normalized , threshold={round(threshold, 2)})')
plt.grid(alpha=0.25)
plt.gcf().autofmt_xdate()
plt.xlabel('Timestamp')
plt.ylabel('Result ($|x_t - \hat{x}_t|$)')
plt.axhline(y=threshold, color='C3', linestyle='--', linewidth=0.5,
            label='Threshold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32))
plt.savefig(f'{tex_plots_path}od_result_{model_type}_{common_id}_all.pdf',
            format='pdf',
            bbox_inches='tight')

# %%
for mt in prediction_summary_df['model_type'].unique():
    best_pred_per_model = prediction_summary_df.loc[
        prediction_summary_df['model_type'] == mt].sort_values(by=['f1_score'],
                                                               ascending=False).iloc[
        0]
    common_id = best_pred_per_model['common_id']
    model_type = best_pred_per_model['model_type']
    window_size = int(best_pred_per_model['window_size']) if \
    best_pred_per_model['window_size'] is not None else None
    center_window = 'cw' if best_pred_per_model['center_window'] else 'nocw'
    normalized = 'normalized' if best_pred_per_model[
        'normalized'] else 'regular'
    threshold = best_pred_per_model['threshold']
    print(f'model type: {model_type}')
    print(f'window size: {window_size}')
    print(f'center window: {center_window}')
    print(f'normalized: {normalized}')
    print(f'threshold: {threshold}')
    print(f'f1 score: {best_pred_per_model["f1_score"]}')
    print(f'precision: {best_pred_per_model["precision"]}')
    print(f'recall: {best_pred_per_model["recall"]}')
    best_pred_per_model_df = pd.read_parquet(
        f'./data/predictions/raw_preprocessed/{normalized}/{common_id}/'
        f'{window_size}_{center_window}_{model_type}.parquet')
    best_pred_per_model_df.info()
    best_pred_per_model_df['y_pred'] = np.where(
        best_pred_per_model_df['result'] > threshold, 1, 0)
    best_pred_per_model_df['pred_type'] = best_pred_per_model_df.apply(
        lambda row: get_TP_TN_FP_FN(row['is_outlier'], row['y_pred']), axis=1)
    # best_pred_df = best_pred_df.loc[best_pred_df['timestamp'] <= '2019-06-01']
    tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
    fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
    plt.plot(best_pred_per_model_df['timestamp'],
             best_pred_per_model_df['water_level'], linewidth=0.5,
             zorder=-1)
    plt.scatter(best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'TN', 'timestamp'],
                best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'TN', 'water_level'],
                s=0.5, c='C0', label='True negative')
    plt.scatter(best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'TP', 'timestamp'],
                best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'TP', 'water_level'],
                s=0.5, c='C1', label='True positive')
    plt.scatter(best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'FP', 'timestamp'],
                best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'FP', 'water_level'],
                s=0.5, c='C2', label='False positive')
    plt.scatter(best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'FN', 'timestamp'],
                best_pred_per_model_df.loc[
                    best_pred_per_model_df['pred_type'] == 'FN', 'water_level'],
                s=0.5, c='C3', label='False negative')
    plt.suptitle(
        f'Outliers classified with {model_type} of '
        f'{stations_dict[common_id]["water_name"]} - '
        f'{stations_dict[common_id]["station_name"]}',
        y=1.05)
    plt.title(
        f'(window size={window_size}, '
        f'{"no " if not best_pred_per_model["center_window"] else ""} '
        f'center window, {"not " if best_pred_per_model["normalized"] else ""} '
        f'normalized , threshold={round(threshold, 2)})')
    plt.grid(alpha=0.25)
    plt.gcf().autofmt_xdate()
    plt.xlabel('Timestamp')
    plt.ylabel('Result ($|x_t - \hat{x}_t|$)')
    # plt.axhline(y=threshold, color='C3', linestyle='--', linewidth=0.5, label='Threshold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32),
              fancybox=True, shadow=True)
    plt.savefig(f'{tex_plots_path}od_{model_type}_{common_id}_all.pdf',
                format='pdf',
                bbox_inches='tight')
    print(f'{best_pred_per_model["f1_score"]}')
    plt.show()

# %%
for mt in prediction_summary_df['model_type'].unique():
    best_pred_per_model = prediction_summary_df.loc[
        prediction_summary_df['model_type'] == mt].sort_values(by=['f1_score'],
                                                               ascending=False).iloc[
        0]
    common_id = best_pred_per_model['common_id']
    model_type = best_pred_per_model['model_type']
    window_size = int(best_pred_per_model['window_size']) if \
    best_pred_per_model['window_size'] is not None else None
    center_window = 'cw' if best_pred_per_model['center_window'] else 'nocw'
    normalized = 'normalized' if best_pred_per_model[
        'normalized'] else 'regular'
    threshold = best_pred_per_model['threshold']
    print(f'model type: {model_type}')
    print(f'window size: {window_size}')
    print(f'center window: {center_window}')
    print(f'normalized: {normalized}')
    print(f'threshold: {threshold}')
    print(f'f1 score: {best_pred_per_model["f1_score"]}')
    print(f'precision: {best_pred_per_model["precision"]}')
    print(f'recall: {best_pred_per_model["recall"]}')
    best_pred_per_model_df = pd.read_parquet(
        f'./data/predictions/raw_preprocessed/{normalized}/{common_id}/'
        f'{window_size}_{center_window}_{model_type}.parquet')
    best_pred_per_model_df['y_pred'] = np.where(
        best_pred_per_model_df['result'] > threshold, 1, 0)
    best_pred_per_model_df['pred_type'] = best_pred_per_model_df.apply(
        lambda row: get_TP_TN_FP_FN(row['is_outlier'], row['y_pred']), axis=1)
    tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'
    min_timestamp = best_pred_per_model_df['timestamp'].min()
    max_timestamp = best_pred_per_model_df['timestamp'].max()
    timedelta = np.timedelta64(7, 'D')
    increment = np.timedelta64(1, 'D')
    lower_window = min_timestamp
    upper_window = min_timestamp + timedelta
    while upper_window < max_timestamp:
        df_slice = best_pred_per_model_df.loc[
            (best_pred_per_model_df['timestamp'] >= lower_window) & (
                        best_pred_per_model_df['timestamp'] < upper_window)]
        lower_window = lower_window + increment
        upper_window = upper_window + increment
        if len(df_slice['pred_type'].unique()) == 4:
            print(f'{lower_window} - {upper_window}')
            fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
            plt.plot(df_slice['timestamp'], df_slice['water_level'],
                     linewidth=0.5,
                     zorder=-1)
            plt.scatter(
                df_slice.loc[df_slice['pred_type'] == 'TN', 'timestamp'],
                df_slice.loc[df_slice['pred_type'] == 'TN', 'water_level'],
                s=0.5, c='C0', label='True negative')
            plt.scatter(
                df_slice.loc[df_slice['pred_type'] == 'TP', 'timestamp'],
                df_slice.loc[df_slice['pred_type'] == 'TP', 'water_level'],
                s=0.5, c='C1', label='True positive')
            plt.scatter(
                df_slice.loc[df_slice['pred_type'] == 'FP', 'timestamp'],
                df_slice.loc[df_slice['pred_type'] == 'FP', 'water_level'],
                s=0.5, c='C2', label='False positive')
            plt.scatter(
                df_slice.loc[df_slice['pred_type'] == 'FN', 'timestamp'],
                df_slice.loc[df_slice['pred_type'] == 'FN', 'water_level'],
                s=0.5, c='C3', label='False negative')
            plt.suptitle(
                f'Outliers classified with {model_type} of '
                f'{stations_dict[common_id]["water_name"]} - '
                f'{stations_dict[common_id]["station_name"]} zoomed in',
                y=1.05)
            plt.title(
                f'(window size={window_size}, '
                f'{"no " if not best_pred_per_model["center_window"] else ""} '
                f'center window, '
                f'{"not " if best_pred_per_model["normalized"] else ""} '
                f'normalized , threshold={round(threshold, 2)})')
            plt.grid(alpha=0.25)
            plt.gcf().autofmt_xdate()
            plt.xlabel('Timestamp')
            plt.ylabel('Result ($|x_t - \hat{x}_t|$)')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.37),
                      fancybox=True, shadow=True)
            plt.savefig(
                f'{tex_plots_path}od_{model_type}_{common_id}_zoomed.pdf',
                format='pdf',
                bbox_inches='tight')
            plt.show()
            break
