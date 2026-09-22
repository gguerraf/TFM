"""
23_run_ml_baselines.py
==================
Machine Learning baseline for the intra-ETF spillover model.

Fits a LightGBM regressor and a simple feed-forward neural network
on the same panel data used by the econometric model. Compares
predictive performance via temporal train/val/test splits.

Neural network follows supervisor guidance:
  - Fixed tanh activation (smooth, infinitely differentiable)
  - Modest hyperparameter grid: layers, nodes per layer
  - MSE loss, Adam optimizer

Requires:
    results/panel_improved.csv  (from 21_estimate_improved_model.py)
    processed/returns_clean.csv (for additional features)

Outputs:
    results/ml_results.txt
    results/lgbm_feature_importance.csv
    figures/shap_summary.png
    figures/lgbm_vs_ols.png
    figures/nn_training_curve.png
    results/nn_grid_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap
import json

# ================================================================================
# CONFIGURATION
# ================================================================================

BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR   = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR    = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Feature columns for ML models
FEATURE_COLS = [
    "Shock_i", "w_i", "Illiq_j", "Mispricing_k",
    "Similarity_ij", "Neg_e",
]

TARGET_COL = "AR_j"

# Temporal split dates
TRAIN_END  = "2021-12-31"
VAL_END    = "2023-12-31"
# Test: 2024-01-01 onward
NN_MAX_TRAIN_OBS = 80000
NN_MAX_EPOCHS = 40
NN_PATIENCE = 5
NN_CURVE_EPOCHS = 40

# ================================================================================
# STEP 0: LOAD AND PREPARE DATA
# ================================================================================

def load_and_prepare():
    """Load panel and prepare features for ML."""
    panel_path = OUTPUT_DIR / "panel_improved.csv"
    if not panel_path.exists():
        print(f"[ERROR] {panel_path} not found. Run 21_estimate_improved_model.py first.")
        exit(1)

    panel = pd.read_csv(panel_path, parse_dates=["t0"])
    print(f"Panel loaded: {panel.shape}")

    # Feature engineering: add extra features available from the panel
    panel["abs_shock"] = panel["Shock_i"].abs()
    panel["shock_x_weight"] = panel["Shock_i"] * panel["w_i"]

    # Add year, month features for seasonality
    panel["year"] = panel["t0"].dt.year
    panel["month"] = panel["t0"].dt.month

    # Extended feature set
    features = FEATURE_COLS + ["abs_shock", "shock_x_weight", "year", "month"]

    # Drop rows with NaN in features or target
    df = panel.dropna(subset=features + [TARGET_COL]).copy()
    print(f"After dropping NaN: {len(df):,} observations")

    return df, features


def temporal_split(df, features):
    """Split data temporally: train / validation / test."""
    train = df[df["t0"] <= TRAIN_END]
    val   = df[(df["t0"] > TRAIN_END) & (df["t0"] <= VAL_END)]
    test  = df[df["t0"] > VAL_END]

    print(f"\nTemporal split:")
    print(f"  Train : {len(train):,} obs ({train['t0'].min().date()} to "
          f"{train['t0'].max().date()})")
    print(f"  Val   : {len(val):,} obs ({val['t0'].min().date()} to "
          f"{val['t0'].max().date()})")
    print(f"  Test  : {len(test):,} obs ({test['t0'].min().date()} to "
          f"{test['t0'].max().date()})")

    X_train, y_train = train[features].values, train[TARGET_COL].values
    X_val,   y_val   = val[features].values,   val[TARGET_COL].values
    X_test,  y_test  = test[features].values,  test[TARGET_COL].values

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train, val, test, features)


def evaluate(y_true, y_pred, label):
    """Compute and print evaluation metrics."""
    r2  = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Directional accuracy: does the predicted sign match the actual sign?
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))

    print(f"  {label}:")
    print(f"    R2             : {r2:.6f}")
    print(f"    MSE            : {mse:.8f}")
    print(f"    MAE            : {mae:.6f}")
    print(f"    Dir. Accuracy  : {dir_acc:.4f}")

    return {"label": label, "R2": r2, "MSE": mse, "MAE": mae,
            "DirAcc": dir_acc}


# ================================================================================
# STEP 1: LIGHTGBM BASELINE
# ================================================================================

def run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test,
                 feature_names, log_lines):
    """Fit LightGBM and evaluate."""
    print("\n" + "=" * 60)
    print("LIGHTGBM BASELINE")
    print("=" * 60)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train,
                             feature_name=feature_names)
    val_data   = lgb.Dataset(X_val, label=y_val,
                             feature_name=feature_names, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Predictions
    pred_train = model.predict(X_train)
    pred_val   = model.predict(X_val)
    pred_test  = model.predict(X_test)

    print("\nLightGBM Performance:")
    r_train = evaluate(y_train, pred_train, "Train")
    r_val   = evaluate(y_val, pred_val, "Validation")
    r_test  = evaluate(y_test, pred_test, "Test")

    log_lines.append("\nLIGHTGBM RESULTS")
    log_lines.append("=" * 40)
    for r in [r_train, r_val, r_test]:
        log_lines.append(f"  {r['label']}: R2={r['R2']:.6f}, "
                         f"MAE={r['MAE']:.6f}, DirAcc={r['DirAcc']:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "lgbm_feature_importance.csv", index=False)
    print(f"\nFeature importance saved.")

    # SHAP analysis
    print("Computing SHAP values (this may take a minute)...")
    try:
        explainer = shap.TreeExplainer(model)
        # Use a sample for speed
        sample_size = min(5000, len(X_val))
        X_sample = X_val[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=feature_names,
                          show=False)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved.")
    except Exception as e:
        print(f"  [WARN] SHAP computation failed: {e}")

    return model, r_test


# ================================================================================
# STEP 2: NEURAL NETWORK (tanh activation, modest grid)
# ================================================================================

def run_neural_network(X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names, log_lines):
    """
    Feed-forward neural network with tanh activation.
    Grid search over: hidden layers (1, 2, 3) x nodes per layer (16, 32, 64).
    Following supervisor guidance: activation fixed to tanh.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("\n[WARN] PyTorch not installed. Skipping neural network.")
        print("  Install with: pip install torch")
        log_lines.append("\nNEURAL NETWORK: Skipped (PyTorch not installed)")
        return None, None

    print("\n" + "=" * 60)
    print("NEURAL NETWORK (tanh activation)")
    print("=" * 60)

    # Standardize features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    rng = np.random.default_rng(42)
    if len(X_tr) > NN_MAX_TRAIN_OBS:
        sample_idx = rng.choice(len(X_tr), size=NN_MAX_TRAIN_OBS, replace=False)
        X_fit = X_tr[sample_idx]
        y_fit = y_train[sample_idx]
        print(f"  NN grid uses a fixed training sample: {len(X_fit):,} obs")
    else:
        X_fit = X_tr
        y_fit = y_train

    X_fit_t = torch.FloatTensor(X_fit).to(device)
    y_fit_t = torch.FloatTensor(y_fit).unsqueeze(1).to(device)
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    y_va_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)

    input_dim = X_tr.shape[1]

    # Define model class
    class SpilloverNN(nn.Module):
        def __init__(self, input_dim, hidden_layers, nodes_per_layer,
                     dropout=0.0):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(prev_dim, nodes_per_layer))
                layers.append(nn.Tanh())   # Fixed activation per supervisor
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = nodes_per_layer
            layers.append(nn.Linear(prev_dim, 1))  # Output layer (linear)
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # Modest grid focused on the supervisor's suggested parameters.
    grid = {
        "hidden_layers": [1, 2, 3],
        "nodes_per_layer": [16, 32, 64],
    }
    fixed_params = {
        "learning_rate": 0.01,
        "dropout": 0.0,
        "batch_size": 2048,
    }

    from itertools import product as itertools_product
    configs_reduced = []
    for hidden_layers, nodes_per_layer in itertools_product(
            grid["hidden_layers"], grid["nodes_per_layer"]):
        configs_reduced.append({
            "hidden_layers": hidden_layers,
            "nodes_per_layer": nodes_per_layer,
            **fixed_params,
        })
    print(f"  Grid search: {len(configs_reduced)} configurations")
    print("  Activation is fixed to tanh; the grid varies hidden layers and nodes.")

    best_val_loss = float("inf")
    best_config = None
    best_model_state = None
    grid_results = []

    for cfg_idx, cfg in enumerate(configs_reduced):
        print(f"    Config {cfg_idx+1}/{len(configs_reduced)}: {cfg}", flush=True)

        model = SpilloverNN(input_dim, cfg["hidden_layers"],
                            cfg["nodes_per_layer"], cfg["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        criterion = nn.MSELoss()

        # Training
        dataset = TensorDataset(X_fit_t, y_fit_t)
        loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

        best_epoch_loss = float("inf")
        patience_counter = 0
        max_epochs = NN_MAX_EPOCHS
        patience = NN_PATIENCE

        for epoch in range(max_epochs):
            model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va_t)
                val_loss = criterion(val_pred, y_va_t).item()

            if val_loss < best_epoch_loss:
                best_epoch_loss = val_loss
                patience_counter = 0
                epoch_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Record this config's result
        model.load_state_dict(epoch_state)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_va_t).cpu().numpy().flatten()
        val_r2 = r2_score(y_val, val_preds)

        grid_results.append({
            **cfg,
            "val_loss": best_epoch_loss,
            "val_R2": val_r2,
            "epochs": epoch + 1,
        })
        print(f"      val_loss={best_epoch_loss:.8f}, val_R2={val_r2:.6f}, epochs={epoch + 1}", flush=True)

        if best_epoch_loss < best_val_loss:
            best_val_loss = best_epoch_loss
            best_config = cfg
            best_model_state = epoch_state

    # Save grid results
    grid_df = pd.DataFrame(grid_results).sort_values("val_loss")
    grid_df.to_csv(OUTPUT_DIR / "nn_grid_results.csv", index=False)
    print(f"\n  Grid search results saved.")
    print(f"  Best config: {best_config}")
    print(f"  Best val loss: {best_val_loss:.8f}")

    # Final evaluation on test set
    final_model = SpilloverNN(
        input_dim, best_config["hidden_layers"],
        best_config["nodes_per_layer"], best_config["dropout"]
    ).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    with torch.no_grad():
        test_preds = final_model(X_te_t).cpu().numpy().flatten()
        train_preds = final_model(X_tr_t).cpu().numpy().flatten()
        val_preds_final = final_model(X_va_t).cpu().numpy().flatten()

    print("\nNeural Network Performance (best config):")
    r_train = evaluate(y_train, train_preds, "Train")
    r_val   = evaluate(y_val, val_preds_final, "Validation")
    r_test  = evaluate(y_test, test_preds, "Test")

    log_lines.append(f"\nNEURAL NETWORK RESULTS (tanh activation)")
    log_lines.append("=" * 40)
    log_lines.append(f"  Best config: {best_config}")
    for r in [r_train, r_val, r_test]:
        log_lines.append(f"  {r['label']}: R2={r['R2']:.6f}, "
                         f"MAE={r['MAE']:.6f}, DirAcc={r['DirAcc']:.4f}")

    # Plot training curve for best config
    # (Re-train with recording)
    try:
        model2 = SpilloverNN(input_dim, best_config["hidden_layers"],
                             best_config["nodes_per_layer"],
                             best_config["dropout"]).to(device)
        optimizer2 = torch.optim.Adam(model2.parameters(),
                                      lr=best_config["learning_rate"])
        train_losses = []
        val_losses = []
        curve_dataset = TensorDataset(X_fit_t, y_fit_t)
        curve_loader = DataLoader(curve_dataset, batch_size=best_config["batch_size"], shuffle=True)
        for epoch in range(NN_CURVE_EPOCHS):
            model2.train()
            epoch_loss = 0
            n_batches = 0
            for batch_X, batch_y in curve_loader:
                optimizer2.zero_grad()
                pred = model2(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer2.step()
                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(epoch_loss / max(n_batches, 1))
            model2.eval()
            with torch.no_grad():
                vl = criterion(model2(X_va_t), y_va_t).item()
                val_losses.append(vl)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train", alpha=0.8)
        ax.plot(val_losses, label="Validation", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"NN Training Curve (best config: "
                     f"{best_config['hidden_layers']}L x "
                     f"{best_config['nodes_per_layer']}N)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "nn_training_curve.png", dpi=150)
        plt.close()
        print(f"  Training curve saved.")
    except Exception as e:
        print(f"  [WARN] Training curve plot failed: {e}")

    return final_model, r_test


# ================================================================================
# STEP 3: OLS BASELINE FOR COMPARISON
# ================================================================================

def ols_baseline(X_train, y_train, X_test, y_test, feature_names, log_lines):
    """Simple OLS baseline for comparison (no fixed effects)."""
    from sklearn.linear_model import LinearRegression

    print("\n" + "=" * 60)
    print("OLS BASELINE (no fixed effects, for ML comparison)")
    print("=" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)

    print("OLS Performance:")
    r_train = evaluate(y_train, pred_train, "Train")
    r_test  = evaluate(y_test, pred_test, "Test")

    log_lines.append("\nOLS BASELINE (no FE)")
    log_lines.append("=" * 40)
    for r in [r_train, r_test]:
        log_lines.append(f"  {r['label']}: R2={r['R2']:.6f}, "
                         f"MAE={r['MAE']:.6f}, DirAcc={r['DirAcc']:.4f}")

    return r_test


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    log_lines = []

    df, features = load_and_prepare()
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_df, val_df, test_df, feature_names) = temporal_split(df, features)

    # OLS baseline
    ols_result = ols_baseline(X_train, y_train, X_test, y_test,
                              feature_names, log_lines)

    # LightGBM
    lgbm_model, lgbm_result = run_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, log_lines)

    # Neural Network
    nn_model, nn_result = run_neural_network(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names, log_lines)

    # Comparison plot
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    models_compared = {"OLS (no FE)": ols_result}
    if lgbm_result:
        models_compared["LightGBM"] = lgbm_result
    if nn_result:
        models_compared["Neural Net (tanh)"] = nn_result

    comp_df = pd.DataFrame(models_compared).T
    print(comp_df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["R2", "MAE", "DirAcc"]
    titles  = ["R-squared (higher is better)",
               "Mean Absolute Error (lower is better)",
               "Directional Accuracy (higher is better)"]
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [comp_df.loc[m, metric] for m in comp_df.index]
        colors = ["#4A90D9", "#E24B4A", "#50C878"][:len(vals)]
        ax.bar(comp_df.index, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lgbm_vs_ols.png", dpi=150)
    plt.close()
    print(f"Comparison plot saved: {FIG_DIR / 'lgbm_vs_ols.png'}")

    # Save full log
    log_path = OUTPUT_DIR / "ml_results.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"ML results saved: {log_path}")

    # Decision: should we proceed to GNN?
    if lgbm_result and ols_result:
        r2_improvement = lgbm_result["R2"] - ols_result["R2"]
        dir_improvement = lgbm_result["DirAcc"] - ols_result["DirAcc"]
        print(f"\nLightGBM vs OLS:")
        print(f"  R2 improvement     : {r2_improvement:+.4f}")
        print(f"  DirAcc improvement : {dir_improvement:+.4f}")
        if r2_improvement > 0.02 or dir_improvement > 0.03:
            print("  ==> ML shows meaningful improvement. GNN may be justified.")
        else:
            print("  ==> ML shows limited improvement. GNN unlikely to add value.")





