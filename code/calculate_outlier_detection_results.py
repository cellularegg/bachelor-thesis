# %%
import glob
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython.core.display_functions import display
import sklearn.metrics as metrics

random_seed = 1
np.random.seed(random_seed)

# %%
plt.style.use('seaborn-colorblind')

# from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 11pt font in plots, to match 11pt font in document
    "axes.labelsize": 11,
    "font.size": 11
}
plt.rcParams.update(tex_fonts)
# tex_plots_path = f'../bachelor-thesis/plots/pdfs/{common_id}/'


# %%

file_list = glob.glob('./data/predictions/raw/**/*.parquet', recursive=True)

predictions = []

for fp in file_list:
    file_name = fp.split('/')[-1]
    metadata = file_name.split('_')
    # remove .parquet
    metadata = [m.split('.')[0] for m in metadata]
    # df = pd.read_parquet(fp)
    metadata_dict = {
        'file_path': fp,
        'normalized': fp.split('/')[-3] == 'normalized',
        'window_size': None if metadata[0] == 'None' else int(metadata[0]),
        'center_window': metadata[1] == 'cw',
        'model_type': metadata[2],
        'common_id': fp.split('/')[-2]
    }
    predictions.append(metadata_dict)

# %%
len(predictions)


# %%
def get_prediction_summary(metadata_dict):
    predictions_summary = []
    pred_df = pd.read_parquet(metadata_dict['file_path'])
    threshold_min = 1
    threshold_max = 100
    threshold_steps = 300
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    y_true = pred_df['is_outlier'].astype(int).to_numpy()
    m = pred_df['result'].to_numpy()
    for threshold in thresholds:
        y_pred = np.where(m > threshold, 1, 0)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        f1_score = metrics.f1_score(y_true, y_pred, zero_division=0)

        predictions_summary.append({
            'common_id': metadata_dict['common_id'],
            'window_size': metadata_dict['window_size'],
            'center_window': metadata_dict['center_window'],
            'model_type': metadata_dict['model_type'],
            'normalized': metadata_dict['normalized'],
            'threshold': threshold,
            'f1_score': f1_score,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        })
    return predictions_summary


# predictions_summary_df = pd.DataFrame(predictions_summary)
# predictions_summary_df.info()

# %%
with mp.Pool(processes=12) as executor:
    results = executor.map(get_prediction_summary, predictions)
    result_lst = [item for sublist in results for item in sublist]

# %%
predictions_summary_df = pd.DataFrame(result_lst)

# %%
predictions_summary_df

# %%
predictions_summary_df.info()

# %%
predictions_summary_df.to_parquet(
    f'./data/predictions/predictions_summary.parquet')

# %%
for id in predictions_summary_df['common_id'].unique():
    print(id)
    df = predictions_summary_df[predictions_summary_df['common_id'] == id]
    df.to_csv(f'./data/predictions/predictions_preprocessed_summary/{id}.csv',
              index=False)
