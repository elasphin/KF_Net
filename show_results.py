"""Show the training and test results in one figure and one table.

    python show_results.py   (after train.py and test.py)  ->  outputs/results.png

Reads outputs/training_info.json, training_history.json, test_summary.json, test_epochs.csv.
"""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import settings as cfg

TRAIN_COLOR, VALIDATION_COLOR = '#1baf7a', '#4a3aa7'
METHOD_COLORS = {'masked_cla_kalmannet': '#2a78d6', 'traditional_ekf': '#eb6834'}

folder = cfg.OUTPUT_FOLDER
info = json.loads((folder / 'training_info.json').read_text())
history = json.loads((folder / 'training_history.json').read_text())
test = json.loads((folder / 'test_summary.json').read_text())
with open(folder / 'test_epochs.csv') as f:
    rows = list(csv.DictReader(f))

epochs = [h['epoch'] for h in history]
network, ekf = test['masked_cla_kalmannet'], test['traditional_ekf']

# --- Table ---------------------------------------------------------------------
lines = [
    'DATA',
    f"  training samples (epochs)   {info['training_samples']}",
    f"  validation samples          {info['validation_samples']}",
    f"  test samples                {network['epochs']}",
    f"  max measurements N_max      {info['max_measurements']}",
    'NETWORK',
    f"  input size                  {info['input_size']}",
    '  layers', *[f'    {layer}' for layer in info['network'].split(' -> ')],
    f"  trainable parameters        {info['trainable_parameters']:,}",
    'TRAINING',
    f"  learning rate               {info['learning_rate']}",
    f"  epochs run / max            {info['epochs_run']} / {info['max_epochs']}",
    f"  best epoch                  {info['best_epoch']}",
    f"  best validation loss        {info['best_validation_loss']:.4g}",
    f"  best validation RMSE        {info['best_validation_position_rmse_m']:.3f} m",
    f"  L2 weight / patience        {info['l2_weight']} / {info['early_stopping_patience']}",
    f"  training time               {info['training_time_s'] / 3600:.2f} h",
    'TEST                          network      EKF',
    f"  RMSE north [m]              {network['rmse_north_m']:7.2f}  {ekf['rmse_north_m']:7.2f}",
    f"  RMSE east [m]               {network['rmse_east_m']:7.2f}  {ekf['rmse_east_m']:7.2f}",
    f"  RMSE down [m]               {network['rmse_down_m']:7.2f}  {ekf['rmse_down_m']:7.2f}",
    f"  RMSE 3D [m]                 {network['rmse_3d_m']:7.2f}  {ekf['rmse_3d_m']:7.2f}",
    f"  improvement 3D              {100 * (1 - network['rmse_3d_m'] / ekf['rmse_3d_m']):6.1f} %",
    f"  horizontal NO (Stanford)    {network['stanford_horizontal_percent']['NO']:6.1f} %"
    f"  {ekf['stanford_horizontal_percent']['NO']:6.1f} %",
    f"  vertical NO (Stanford)      {network['stanford_vertical_percent']['NO']:6.1f} %"
    f"  {ekf['stanford_vertical_percent']['NO']:6.1f} %",
    f"  epochs with fault           {network['epochs_with_fault']:7d}  {ekf['epochs_with_fault']:7d}",
]
print('\n'.join(lines))

# --- Figure ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(epochs, [h['train_loss'] for h in history], color=TRAIN_COLOR, linewidth=2, label='train')
ax.plot(epochs, [h['validation_loss'] for h in history], color=VALIDATION_COLOR, linewidth=2, linestyle='--',
        label='validation')
ax.axvline(info['best_epoch'], color='#52514e', linewidth=1, linestyle=':', label=f"best epoch {info['best_epoch']}")
ax.set(yscale='log', xlabel='Epoch', ylabel='Loss, Eq. (32)', title='Loss')

ax = axes[0, 1]
ax.plot(epochs, [h['train_position_rmse_m'] for h in history], color=TRAIN_COLOR, linewidth=2, label='train')
ax.plot(epochs, [h['validation_position_rmse_m'] for h in history], color=VALIDATION_COLOR, linewidth=2,
        linestyle='--', label='validation')
ax.axvline(info['best_epoch'], color='#52514e', linewidth=1, linestyle=':')
ax.set(xlabel='Epoch', ylabel='Position RMSE [m]', title='Position RMSE (cf. paper Fig. 15)')

ax = axes[1, 0]
for method, color in METHOD_COLORS.items():
    error = np.sort([np.sqrt(float(r['north_error_m']) ** 2 + float(r['east_error_m']) ** 2
                             + float(r['down_error_m']) ** 2) for r in rows if r['method'] == method])
    ax.plot(error, np.arange(1, len(error) + 1) / len(error), color=color, linewidth=2, label=method)
ax.set(xlabel='3D position error [m]', ylabel='Cumulative probability', title='Test error CDF (cf. paper Fig. 19)')

for ax in axes.flat[:3]:
    ax.grid(color='#e5e5e3', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)

axes[1, 1].axis('off')
axes[1, 1].text(0.0, 1.0, '\n'.join(lines), family='monospace', fontsize=8.5, va='top')

fig.tight_layout()
fig.savefig(folder / 'results.png', dpi=150)
print(f"saved {folder / 'results.png'}")
