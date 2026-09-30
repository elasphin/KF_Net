"""Recursive test of the trained network on Data02 (run after pipeline.py)."""
from __future__ import annotations

import config as cfg
from leo import load_cache
from mknet import load_checkpoint
from pipeline import DEVICE, prepare_fusion_epochs, run_recursive, save_test_results
from simulation_data import load_dataset


def main() -> None:
    if cfg.TEST_OUTPUT_DIR.exists() and any(cfg.TEST_OUTPUT_DIR.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output directory: {cfg.TEST_OUTPUT_DIR}')
    # N_max comes from the checkpoint: the compact Eq. (16) layout has no satellite slots.
    model, ckpt = load_checkpoint(cfg.CHECKPOINT_PATH, DEVICE)
    if model.gain_scale_mode != cfg.GAIN_SCALE_MODE:
        raise ValueError(f'checkpoint gain_scale_mode={model.gain_scale_mode!r} differs from '
                         f'config GAIN_SCALE_MODE={cfg.GAIN_SCALE_MODE!r}')
    ds = load_dataset('test')
    epochs = prepare_fusion_epochs(ds, load_cache(cfg.TEST_LEO_CACHE, ds).meas)
    run = run_recursive(model, epochs, ds)
    validation = {'best_epoch': int(ckpt['epoch']), 'state_loss_mean': float(ckpt['validation_state_loss']),
                  'position_rmse_m': float(ckpt['validation_position_rmse_m'])}
    save_test_results(cfg.TEST_OUTPUT_DIR, validation, run)


if __name__ == '__main__':
    main()
