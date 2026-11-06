"""Gradient-boosted models over the per-trade features (milestone 4e).

`features` turns each shown buy into one point-in-time feature row, and `study` trains
LightGBM walk-forward by year, compares it with the two-input score and with
shuffled-label placebos, and explains what it learned with SHAP values.
"""
