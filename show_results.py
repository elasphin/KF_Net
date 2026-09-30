"""Show the training and test results in one figure and one table.

    python show_results.py   (after train.py and test.py)  ->  outputs/results.png

Panels: loss and position RMSE per epoch (cf. paper Fig. 15), test errors over
time (Fig. 18), Stanford diagrams (Fig. 20), error CDF (Fig. 19), table.

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


def test_column(method, name):
    return np.array([float(r[name]) for r in rows if r['method'] == method])

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
fig, axes = plt.subplots(4, 2, figsize=(14, 18))

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

for method, color in METHOD_COLORS.items():
    seconds = test_column(method, 'time_gpst_s') - test_column(method, 'time_gpst_s')[0]
    horizontal = np.hypot(test_column(method, 'north_error_m'), test_column(method, 'east_error_m'))
    vertical = np.abs(test_column(method, 'down_error_m'))
    for column, error, pl_name in ((0, horizontal, 'horizontal_pl_m'), (1, vertical, 'vertical_pl_m')):
        axes[1, column].plot(seconds, error, color=color, linewidth=1.2, label=method)
        axes[2, column].scatter(error, test_column(method, pl_name), s=8, color=color, alpha=0.5, label=method)
    error_3d = np.sort(np.hypot(horizontal, vertical))
    axes[3, 0].plot(error_3d, np.arange(1, len(error_3d) + 1) / len(error_3d), color=color, linewidth=2,
                    label=method)

limit = 1.3 * cfg.ALERT_LIMIT
for column, name in ((0, 'Horizontal'), (1, 'Vertical')):
    axes[1, column].set(xlabel='Time [s]', ylabel=f'{name} error [m]', title=f'{name} test error (cf. paper Fig. 18)')
    ax = axes[2, column]
    ax.plot([0, limit], [0, limit], color='#52514e', linewidth=1)
    ax.axhline(cfg.ALERT_LIMIT, color='#52514e', linewidth=1, linestyle='--')
    ax.axvline(cfg.ALERT_LIMIT, color='#52514e', linewidth=1, linestyle='--')
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='Position error [m]', ylabel='Protection level [m]',
           title=f'{name} Stanford diagram, AL = {cfg.ALERT_LIMIT:g} m (cf. paper Fig. 20)')
axes[3, 0].set(xlabel='3D position error [m]', ylabel='Cumulative probability',
               title='Test error CDF (cf. paper Fig. 19)')

for ax in axes.flat[:7]:
    ax.grid(color='#e5e5e3', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)

axes[3, 1].axis('off')
axes[3, 1].text(0.0, 1.0, '\n'.join(lines), family='monospace', fontsize=8.5, va='top')

fig.tight_layout()
fig.savefig(folder / 'results.png', dpi=150)
print(f"saved {folder / 'results.png'}")
