#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAAC-ML (v15) : reproduce the full machine-learning analysis of metal-catalyzed
azide-alkyne cycloaddition (MAAC) directly from the curated DFT database.

Just run this one file:

    python MAAC_run.py

It does everything automatically:
  1. Installs any missing Python packages.
  2. Finds the database ("MAAC_database/" folder or "MAAC_database.zip"; on Google
     Colab it asks you to upload the zip if neither is present).
  3. Runs the analysis and writes all figures and tables to "MAAC_outputs/":
        - per-metal, per-step barrier statistics with bootstrap confidence intervals
        - reference baselines (global mean, (metal, step) cell means, one-hot linear,
          descriptor-only GP) and the six regression families (leave-one-out)
        - Gaussian-process model with calibrated uncertainty; feature ablation
        - single-environment subsets (gas phase only / C-PCM toluene only)
        - grouped cross-validation and leave-one-study-out (applicability domain)
        - SHAP design rules, uncertainty-aware virtual screening
        - energetic-span ranking (Kozuch-Shaik) with bootstrap intervals
        - closed-form / symbolic-regression baseline (reported honestly)
  4. On Google Colab, zips "MAAC_outputs/" and downloads it.

The database is read-only; no new quantum chemistry is performed.
"""

import os
import sys


# --------------------------------------------------------------------------- #
#  Environment helpers: package install, Colab detection, database location
# --------------------------------------------------------------------------- #
REQUIRED_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "scipy": "scipy", "xgboost": "xgboost",
    "shap": "shap", "gplearn": "gplearn",
}


def ensure_packages():
    """Install any missing third-party packages (useful on a fresh Colab/VM)."""
    import importlib.util
    import subprocess
    missing = [pip_name for module, pip_name in REQUIRED_PACKAGES.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing],
                       check=False)


def running_in_colab():
    """Return True when executing inside a Google Colab notebook."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def find_database_dir(preferred=None):
    """Return the folder that contains both structures.csv and barriers.csv."""
    for root in [preferred, "MAAC_database", "."]:
        if root and os.path.isdir(root):
            for current, _dirs, files in os.walk(root):
                if "structures.csv" in files and "barriers.csv" in files:
                    return current
    return None


def unzip(zip_path):
    """Extract a zip archive into the current directory (pure Python)."""
    import zipfile
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(".")


def get_database(preferred=None):
    """Locate the database, unzipping or prompting for MAAC_database.zip if needed."""
    database = find_database_dir(preferred)
    if database:
        return database
    if os.path.exists("MAAC_database.zip"):
        unzip("MAAC_database.zip")
        database = find_database_dir(preferred)
        if database:
            return database
    if running_in_colab():
        print("\nMAAC_database.zip was not found - please upload it now.")
        from google.colab import files
        files.upload()
        if os.path.exists("MAAC_database.zip"):
            unzip("MAAC_database.zip")
            database = find_database_dir(preferred)
            if database:
                return database
    sys.exit("ERROR: could not find the MAAC database. Put MAAC_database.zip next to "
             "this script (it will be unzipped automatically) and run again.")


def download_outputs(output_dir):
    """On Colab, zip and download the results; otherwise report the local path."""
    if not running_in_colab():
        print(f"\nAll results are in: {os.path.abspath(output_dir)}")
        return
    import shutil
    archive = shutil.make_archive(output_dir, "zip", output_dir)
    from google.colab import files
    print(f"\nDownloading {os.path.basename(archive)} ...")
    files.download(archive)


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #
ELECTRONIC_DESCRIPTORS = ["HOMO", "LUMO", "gap", "mu", "eta", "omega", "chi", "dNmax"]
HARTREE_TO_KCAL = 627.5095
METAL_COLOR = {"Cu": "#C0451B", "Ag": "#6F7F8E", "Au": "#C9951A", "Ru": "#2E8B57", "Rh": "#7E3CB5"}
METAL_ORDER = ["Cu", "Ag", "Au", "Ru", "Rh"]
SEED = 0


def run_analysis(database_dir, output_dir):
    """Reproduce every figure and table from the database."""
    import csv
    import json
    import warnings
    from collections import defaultdict
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from sklearn.model_selection import LeaveOneOut, GroupKFold, KFold, cross_val_predict
    from sklearn.linear_model import Ridge, LinearRegression, LassoCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Constant
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    from xgboost import XGBRegressor
    import shap

    plt.rcParams.update({"font.size": 11})
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    numbers = {}

    # ------------------------------------------------------------------ data
    structures = pd.read_csv(os.path.join(database_dir, "structures.csv"))
    barriers = pd.read_csv(os.path.join(database_dir, "barriers.csv"))
    S = structures.set_index("structure_id")
    print(f"loaded {len(structures)} structures, {len(barriers)} barriers from {database_dir}")
    rows = []
    for _, bar in barriers.iterrows():
        ref = S.loc[bar["reactant_id"]]
        d = dict(barrier_id=bar["barrier_id"], dG=float(bar["dG_act_kcal"]), step=int(bar["step"]), metal=bar["metal"],
                 nucl=bar["nuclearity"] if isinstance(bar["nuclearity"], str) else "mono",
                 system=bar["system_id"], pathway=bar["pathway"], paper=bar["paper_code"],
                 phase=bar["phase"], solvent=bar["solvent"], eps=float(bar["eps"]), charge=int(bar["charge"]))
        for key in ELECTRONIC_DESCRIPTORS:
            d[key] = float(ref[key])
        d["Vbur"] = float(ref["pct_Vbur"]); d["qM"] = float(ref["q_metal"])
        rows.append(d)
    data = pd.DataFrame(rows).dropna(subset=ELECTRONIC_DESCRIPTORS + ["Vbur", "qM", "dG"]).reset_index(drop=True)
    data.to_csv(f"{output_dir}/MAAC_modeling_table.csv", index=False)
    y = data["dG"].values
    n = len(y)
    numbers["n_barriers"] = int(n)
    numbers["n_gas"] = int((data.phase == "gas").sum()); numbers["n_cpcm"] = int((data.phase == "cpcm").sum())
    numbers["n_cation"] = int((data.charge == 1).sum()); numbers["n_neutral"] = int((data.charge == 0).sum())
    print(f"modeling set: n={n}  gas={numbers['n_gas']} C-PCM={numbers['n_cpcm']}  "
          f"neutral={numbers['n_neutral']} cationic={numbers['n_cation']}  "
          f"target {y.mean():.1f}+/-{y.std():.1f} kcal/mol (range {y.min():.1f}..{y.max():.1f})")
    numbers["dG_mean"] = round(float(y.mean()), 1); numbers["dG_min"] = round(float(y.min()), 1); numbers["dG_max"] = round(float(y.max()), 1)

    CONT = ELECTRONIC_DESCRIPTORS + ["Vbur", "qM"]
    def design(frame, cols_cont, categorical=("metal", "nucl"), extra=("step", "charge", "eps")):
        X = frame[list(cols_cont) + [c for c in extra if c]].copy()
        for c in categorical:
            X = pd.concat([X, pd.get_dummies(frame[c], prefix=c)], axis=1)
        return X.astype(float)
    X_full = design(data, CONT)                                       # 20 features (with charge + eps)
    X_old = design(data, CONT, extra=("step",))                       # original 18 features
    feature_names = list(X_full.columns)
    numbers["n_features"] = len(feature_names)
    print("features:", feature_names)

    # ------------------------------------------------------------------ helpers
    def make_kernel():
        return Constant(1.0, (1e-2, 1e2)) * RBF(5.0, (0.5, 50)) + WhiteKernel(1.0, (1e-3, 1e2))

    def gp_fit(Xtr, ytr):
        scaler = StandardScaler().fit(Xtr)
        gp = GaussianProcessRegressor(kernel=make_kernel(), normalize_y=True, alpha=1e-6, n_restarts_optimizer=0)
        gp.fit(scaler.transform(Xtr), ytr)
        return scaler, gp

    def gp_loo(X, yv):
        X = np.asarray(X, float); mean = np.zeros(len(yv)); std = np.zeros(len(yv))
        for train, test in LeaveOneOut().split(X):
            scaler, gp = gp_fit(X[train], yv[train])
            m, s = gp.predict(scaler.transform(X[test]), return_std=True)
            mean[test], std[test] = m[0], s[0]
        return mean, std

    def loo(model_factory, X, yv, scale=False):
        X = np.asarray(X, float); pred = np.zeros(len(yv))
        for train, test in LeaveOneOut().split(X):
            Xtr, Xte = X[train], X[test]
            if scale:
                scaler = StandardScaler().fit(Xtr); Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
            model = model_factory(); model.fit(Xtr, yv[train]); pred[test] = model.predict(Xte)[0]
        return pred

    def make_xgb():
        return XGBRegressor(n_estimators=160, max_depth=3, learning_rate=0.06, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=2, random_state=SEED)

    def boot_ci(values, B=4000, stat=np.mean):
        values = np.asarray(values, float)
        if len(values) < 2:
            return (float("nan"), float("nan"))
        idx = rng.integers(0, len(values), size=(B, len(values)))
        return tuple(float(v) for v in np.percentile(stat(values[idx], axis=1), [2.5, 97.5]))

    def boot_diff_ci(a, b_, B=4000):
        a = np.asarray(a, float); b_ = np.asarray(b_, float)
        if len(a) < 2 or len(b_) < 2:
            return (float("nan"), float("nan"))
        ia = rng.integers(0, len(a), size=(B, len(a))); ib = rng.integers(0, len(b_), size=(B, len(b_)))
        return tuple(float(v) for v in np.percentile(a[ia].mean(1) - b_[ib].mean(1), [2.5, 97.5]))

    def metrics(yv, pred):
        return dict(R2=round(float(r2_score(yv, pred)), 3), MAE=round(float(mean_absolute_error(yv, pred)), 2),
                    RMSE=round(float(np.sqrt(np.mean((yv - pred) ** 2))), 2))

    # ------------------------------------------------------------------ 1. per-metal, per-step statistics
    stats = []
    for step in (1, 2):
        for metal in METAL_ORDER:
            v = data[(data.step == step) & (data.metal == metal)].dG.values
            if len(v) == 0:
                continue
            lo, hi = boot_ci(v)
            stats.append(dict(step=step, metal=metal, n=len(v), mean=round(float(v.mean()), 1),
                              sd=round(float(v.std(ddof=1)), 1) if len(v) > 1 else float("nan"),
                              ci_low=round(lo, 1), ci_high=round(hi, 1), median=round(float(np.median(v)), 1)))
    pooled = []
    for metal in METAL_ORDER:
        v = data[data.metal == metal].dG.values; lo, hi = boot_ci(v)
        pooled.append(dict(step="pooled", metal=metal, n=len(v), mean=round(float(v.mean()), 1),
                           sd=round(float(v.std(ddof=1)), 1), ci_low=round(lo, 1), ci_high=round(hi, 1), median=round(float(np.median(v)), 1)))
    stat_table = pd.DataFrame(stats + pooled)
    stat_table.to_csv(f"{output_dir}/MAAC_metal_step_statistics.csv", index=False)
    print("\n[per-metal, per-step barrier statistics]")
    print(stat_table.to_string(index=False))
    numbers["metal_step"] = stats; numbers["metal_pooled"] = pooled
    # pairwise differences (Cu vs others) per step
    pairs = []
    for step in (1, 2):
        cu = data[(data.step == step) & (data.metal == "Cu")].dG.values
        for metal in ["Ag", "Au", "Ru", "Rh"]:
            v = data[(data.step == step) & (data.metal == metal)].dG.values
            if len(v) < 2:
                continue
            lo, hi = boot_diff_ci(v, cu)
            pairs.append(dict(step=step, comparison=f"{metal} - Cu", diff=round(float(v.mean() - cu.mean()), 1),
                              ci_low=round(lo, 1), ci_high=round(hi, 1), resolved=bool(lo > 0 or hi < 0)))
    pair_table = pd.DataFrame(pairs); pair_table.to_csv(f"{output_dir}/MAAC_metal_pairwise.csv", index=False)
    print("\n[pairwise differences vs Cu, 95% bootstrap CI]"); print(pair_table.to_string(index=False))
    numbers["pairwise"] = pairs

    # figure 1: per-step bars with 95% CI
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharey=False)
    for ax, step in zip(axes, (1, 2)):
        sub = stat_table[stat_table.step == step].copy()
        sub["metal"] = pd.Categorical(sub["metal"], METAL_ORDER); sub = sub.sort_values("metal")
        err = np.vstack([sub["mean"] - sub["ci_low"], sub["ci_high"] - sub["mean"]])
        ax.bar(sub.metal, sub["mean"], yerr=err, capsize=4, color=[METAL_COLOR[m] for m in sub.metal], edgecolor="k", lw=0.6, zorder=3)
        for i, (_, r) in enumerate(sub.iterrows()):
            top = r["ci_high"] if not np.isnan(r["ci_high"]) else r["mean"]
            ax.text(i, top + 0.8, f"n={int(r['n'])}", ha="center", fontsize=10)
        ax.set_title(f"step {step}: " + ("TS1 (first C–N bond)" if step == 1 else "TS2 (ring closure)"), fontsize=12)
        ax.set_xlabel("metal", fontsize=11); ax.set_ylabel(r"mean $\Delta G^{\ddag}$ (kcal mol$^{-1}$)", fontsize=11); ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.tick_params(labelsize=11); ax.set_ylim(0, np.nanmax(sub["ci_high"].fillna(sub["mean"])) * 1.28)
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_metal_barriers.png", dpi=200); plt.close()

    # ------------------------------------------------------------------ 2. baselines and model comparison (LOO)
    comparison = []
    # global mean
    pred = np.array([np.delete(y, i).mean() for i in range(n)])
    comparison.append(dict(model="Global mean (no information)", **metrics(y, pred)))
    # (metal, step) cell means
    cell_pred = np.zeros(n)
    for i in range(n):
        mask = ((data.metal == data.metal[i]) & (data.step == data.step[i])).values.copy(); mask[i] = False
        cell_pred[i] = y[mask].mean() if mask.sum() else np.delete(y, i).mean()
    comparison.append(dict(model="(metal, step) cell-mean lookup", **metrics(y, cell_pred)))
    # one-hot linear (metal + step, additive)
    X_onehot = pd.get_dummies(data[["metal"]]).astype(float); X_onehot["step"] = data["step"].astype(float)
    comparison.append(dict(model="Linear, one-hot metal + step (additive)", **metrics(y, loo(LinearRegression, X_onehot.values, y))))
    # descriptor-only GP (no metal / step / nuclearity / charge / eps)
    desc_mean, _ = gp_loo(data[CONT].values, y)
    comparison.append(dict(model="GP, 10 continuous descriptors only", **metrics(y, desc_mean)))
    desc_step_mean, _ = gp_loo(data[CONT + ["step"]].values, y)
    comparison.append(dict(model="GP, descriptors + step (no metal)", **metrics(y, desc_step_mean)))
    # the six families on the full design
    print("\n[model comparison - leave-one-out]")
    families = [
        ("Ridge regression (linear baseline)", lambda: Ridge(alpha=5), True),
        ("Random forest", lambda: RandomForestRegressor(400, max_depth=6, random_state=SEED), False),
        ("Gradient boosting", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED), False),
        ("XGBoost", make_xgb, False),
        ("Neural network (MLP)", lambda: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=2000, random_state=SEED), True),
    ]
    for name, factory, scale in families:
        pr = loo(factory, X_full.values, y, scale); comparison.append(dict(model=name, **metrics(y, pr)))
    gp_mean, gp_std = gp_loo(X_full.values, y)
    comparison.append(dict(model="Gaussian process (primary model)", **metrics(y, gp_mean)))
    comp_table = pd.DataFrame(comparison); comp_table.to_csv(f"{output_dir}/MAAC_model_comparison.csv", index=False)
    print(comp_table.to_string(index=False))
    numbers["model_comparison"] = comparison
    data["pred"] = gp_mean; data["sd"] = gp_std; data["cell_mean_pred"] = cell_pred

    # GP vs cell-mean: paired bootstrap on LOO residuals
    e_gp = np.abs(y - gp_mean); e_cm = np.abs(y - cell_pred)
    idx = rng.integers(0, n, size=(4000, n))
    d_mae = (e_cm[idx].mean(1) - e_gp[idx].mean(1))
    r2_gp = 1 - ((y[idx] - gp_mean[idx]) ** 2).sum(1) / ((y[idx] - y[idx].mean(1, keepdims=True)) ** 2).sum(1)
    r2_cm = 1 - ((y[idx] - cell_pred[idx]) ** 2).sum(1) / ((y[idx] - y[idx].mean(1, keepdims=True)) ** 2).sum(1)
    numbers["gp_vs_cellmean"] = dict(dMAE=round(float(e_cm.mean() - e_gp.mean()), 2), dMAE_ci=[round(float(v), 2) for v in np.percentile(d_mae, [2.5, 97.5])],
                                     dR2=round(float(r2_score(y, gp_mean) - r2_score(y, cell_pred)), 2), dR2_ci=[round(float(v), 2) for v in np.percentile(r2_gp - r2_cm, [2.5, 97.5])],
                                     frac_better=round(float((e_gp < e_cm).mean()), 2))
    print(f"\n[GP vs cell-mean] dMAE={numbers['gp_vs_cellmean']['dMAE']} CI {numbers['gp_vs_cellmean']['dMAE_ci']} ; "
          f"dR2={numbers['gp_vs_cellmean']['dR2']} CI {numbers['gp_vs_cellmean']['dR2_ci']} ; GP better on {numbers['gp_vs_cellmean']['frac_better']*100:.0f}% of barriers")

    # ------------------------------------------------------------------ 3. GP primary: calibration, ablation, subsets
    within_1 = float(np.mean(np.abs(y - gp_mean) <= gp_std) * 100); within_2 = float(np.mean(np.abs(y - gp_mean) <= 2 * gp_std) * 100)
    numbers["gp"] = dict(**metrics(y, gp_mean), within_1sigma=round(within_1), within_2sigma=round(within_2), mean_sigma=round(float(gp_std.mean()), 2))
    print(f"\n[Gaussian process - primary] R2={numbers['gp']['R2']:.2f} MAE={numbers['gp']['MAE']:.2f} RMSE={numbers['gp']['RMSE']:.2f} "
          f"mean_sigma={gp_std.mean():.2f}; {within_1:.0f}% within 1 sigma, {within_2:.0f}% within 2 sigma")
    scaler_full, gp_full = gp_fit(X_full.values, y); k = gp_full.kernel_
    numbers["gp_kernel"] = dict(sigma_f2=round(float(k.k1.k1.constant_value), 2), length_scale=round(float(k.k1.k2.length_scale), 2),
                                sigma_n2=round(float(k.k2.noise_level), 3), signal_kcal=round(float(np.sqrt(k.k1.k1.constant_value) * y.std()), 1),
                                noise_kcal=round(float(np.sqrt(k.k2.noise_level) * y.std()), 1))
    print("  fitted kernel:", numbers["gp_kernel"])
    # ablation: original 18 features (no charge / eps)
    old_mean, old_std = gp_loo(X_old.values, y)
    numbers["gp_old_features"] = dict(**metrics(y, old_mean), within_1sigma=round(float(np.mean(np.abs(y - old_mean) <= old_std) * 100)))
    print(f"  ablation (without charge and eps): R2={numbers['gp_old_features']['R2']:.2f} MAE={numbers['gp_old_features']['MAE']:.2f}")
    # single-environment subsets
    subsets = []
    for label, mask in [("gas phase only", (data.phase == "gas").values), ("C-PCM toluene only", (data.phase == "cpcm").values),
                        ("neutral only", (data.charge == 0).values), ("cationic only", (data.charge == 1).values)]:
        ys = y[mask]; Xs = X_full.values[mask]
        if mask.sum() < 15:
            continue
        m_, s_ = gp_loo(Xs, ys)
        cm = np.zeros(len(ys)); sub = data[mask].reset_index(drop=True)
        for i in range(len(ys)):
            mk = ((sub.metal == sub.metal[i]) & (sub.step == sub.step[i])).values.copy(); mk[i] = False
            cm[i] = ys[mk].mean() if mk.sum() else np.delete(ys, i).mean()
        subsets.append(dict(subset=label, n=int(mask.sum()), **{f"GP_{k_}": v for k_, v in metrics(ys, m_).items()},
                            GP_within_1sigma=round(float(np.mean(np.abs(ys - m_) <= s_) * 100)),
                            **{f"cellmean_{k_}": v for k_, v in metrics(ys, cm).items()}))
    subset_table = pd.DataFrame(subsets); subset_table.to_csv(f"{output_dir}/MAAC_subsets.csv", index=False)
    print("\n[single-environment subsets]"); print(subset_table.to_string(index=False)); numbers["subsets"] = subsets

    # ------------------------------------------------------------------ 4. grouped CV and leave-one-study-out
    groups = data["paper"].values
    grouped_prediction = np.zeros(n)
    for train, test in GroupKFold(min(6, len(set(groups)))).split(X_full.values, y, groups):
        grouped_prediction[test] = make_xgb().fit(X_full.values[train], y[train]).predict(X_full.values[test])
    numbers["grouped_cv_xgb"] = metrics(y, grouped_prediction)
    loso_mean = np.zeros(n); loso_std = np.zeros(n); loso_rows = []
    for study in sorted(set(groups)):
        train = groups != study; test = groups == study
        scaler, gp = gp_fit(X_full.values[train], y[train]); m_, s_ = gp.predict(scaler.transform(X_full.values[test]), return_std=True)
        loso_mean[test], loso_std[test] = m_, s_; err = m_ - y[test]
        loso_rows.append(dict(study=study, n=int(test.sum()), MAE=round(float(np.abs(err).mean()), 2), bias=round(float(err.mean()), 2),
                              within_1sigma_pct=round(float(np.mean(np.abs(err) <= s_) * 100)), within_2sigma_pct=round(float(np.mean(np.abs(err) <= 2 * s_) * 100)),
                              mean_sigma=round(float(s_.mean()), 2)))
    err = loso_mean - y
    loso_rows.append(dict(study="ALL (pooled)", n=n, MAE=round(float(np.abs(err).mean()), 2), bias=round(float(err.mean()), 2),
                          within_1sigma_pct=round(float(np.mean(np.abs(err) <= loso_std) * 100)), within_2sigma_pct=round(float(np.mean(np.abs(err) <= 2 * loso_std) * 100)),
                          mean_sigma=round(float(loso_std.mean()), 2)))
    loso_table = pd.DataFrame(loso_rows); loso_table.to_csv(f"{output_dir}/MAAC_leave_one_study_out.csv", index=False)
    numbers["loso"] = dict(rows=loso_rows, R2=round(float(r2_score(y, loso_mean)), 2))
    print(f"\n[grouped-CV, XGBoost] R2={numbers['grouped_cv_xgb']['R2']:.2f}   [leave-one-study-out, GP] R2={numbers['loso']['R2']:.2f}")
    print(loso_table.to_string(index=False))
    # sigma as applicability-domain flag: error binned by predicted sigma (LOO + LOSO pooled)
    all_err = np.concatenate([np.abs(y - gp_mean), np.abs(err)]); all_sd = np.concatenate([gp_std, loso_std])
    bins = [0, 4, 5, 6, 8, 100]; ad = []
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        mk = (all_sd >= lo_) & (all_sd < hi_)
        if mk.sum():
            ad.append(dict(sigma_bin=f"{lo_}-{hi_ if hi_ < 100 else 'inf'}", n=int(mk.sum()), MAE=round(float(all_err[mk].mean()), 2),
                           within_2sigma_pct=round(float(np.mean(all_err[mk] <= 2 * all_sd[mk]) * 100))))
    pd.DataFrame(ad).to_csv(f"{output_dir}/MAAC_sigma_bins.csv", index=False); numbers["sigma_bins"] = ad
    print("[error by predicted sigma]"); print(pd.DataFrame(ad).to_string(index=False))

    # ------------------------------------------------------------------ 5. figures: parity, calibration, per-metal
    step_color = {1: "#2b6cb0", 2: "#dd6b20"}
    limits = [min(y.min(), gp_mean.min()) - 3, max(y.max(), gp_mean.max()) + 3]
    plt.figure(figsize=(5.4, 5.2))
    for step in (1, 2):
        mask = data["step"] == step
        plt.errorbar(data.dG[mask], data.pred[mask], yerr=data.sd[mask], fmt="o", ms=6, mfc=step_color[step], mec="w",
                     ecolor=step_color[step], elinewidth=1, capsize=2, alpha=.85, label=f"step {step}", zorder=3)
    plt.plot(limits, limits, "--", c="gray"); plt.xlim(limits); plt.ylim(limits); plt.legend(frameon=False)
    plt.xlabel(r"DFT $\Delta G^{\ddag}$ (kcal mol$^{-1}$)"); plt.ylabel(r"GP prediction $\pm\sigma$ (LOO)")
    plt.title(f"Gaussian process: R²={numbers['gp']['R2']:.2f}, MAE={numbers['gp']['MAE']:.1f} (n={n})")
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_parity_gp.png", dpi=200); plt.close()
    levels = np.linspace(0.05, 0.99, 30)
    observed = [np.mean(np.abs(y - gp_mean) <= norm.ppf((1 + p) / 2) * gp_std) for p in levels]
    obs_loso = [np.mean(np.abs(err) <= norm.ppf((1 + p) / 2) * loso_std) for p in levels]
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "--", c="gray", label="perfect")
    plt.plot(levels, observed, "-o", ms=4, c="#2b6cb0", label="GP, leave-one-out"); plt.plot(levels, obs_loso, "-s", ms=4, c="#dd6b20", label="GP, leave-one-study-out")
    plt.xlabel("expected confidence"); plt.ylabel("observed coverage"); plt.title("Uncertainty calibration"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_calibration.png", dpi=200); plt.close()
    plt.figure(figsize=(5.4, 5.2)); per_metal = []
    for metal in METAL_ORDER:
        g = data[data.metal == metal]
        if len(g) == 0:
            continue
        r2 = r2_score(g.dG, g.pred) if len(g) > 2 else float("nan"); mae = mean_absolute_error(g.dG, g.pred)
        per_metal.append(dict(metal=metal, n=len(g), R2=round(float(r2), 2), MAE=round(float(mae), 2)))
        plt.scatter(g.dG, g.pred, s=46, edgecolor="w", color=METAL_COLOR[metal], label=f"{metal} (n={len(g)}, R²={r2:.2f})", zorder=3)
    plt.plot(limits, limits, "--", c="gray"); plt.xlim(limits); plt.ylim(limits); plt.legend(frameon=False, fontsize=9, loc="upper left")
    plt.xlabel(r"DFT $\Delta G^{\ddag}$ (kcal mol$^{-1}$)"); plt.ylabel("GP prediction (LOO)"); plt.title("Per-metal reliability (applicability domain)")
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_parity_by_metal.png", dpi=200); plt.close()
    numbers["per_metal"] = per_metal; print("\n[per-metal LOO]", per_metal)

    # ------------------------------------------------------------------ 6. SHAP design rules
    model = make_xgb().fit(X_full, y); shap_values = shap.TreeExplainer(model).shap_values(X_full)
    plt.figure(); shap.summary_plot(shap_values, X_full, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_shap_summary.png", dpi=200, bbox_inches="tight"); plt.close()
    importance = pd.Series(np.abs(shap_values).mean(0), index=feature_names).sort_values(ascending=False)
    importance.to_csv(f"{output_dir}/MAAC_shap_importance.csv", header=["mean_abs_shap"])
    numbers["shap_top"] = [(k_, round(float(v), 2)) for k_, v in importance.head(8).items()]
    print("\n[SHAP] top features:", numbers["shap_top"])
    continuous = [c for c in feature_names if c in CONT]
    top2 = [c for c in importance.index if c in continuous][:2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    for ax, fc in zip(axes, top2):
        i = feature_names.index(fc)
        sc_ = ax.scatter(X_full[fc], shap_values[:, i], c=data["dG"], cmap="coolwarm", s=30, edgecolor="w")
        ax.axhline(0, c="gray", lw=.8); ax.set_xlabel(fc); ax.set_ylabel(f"SHAP ({fc})"); ax.set_title(f"design rule: {fc}")
    fig.colorbar(sc_, ax=axes, label=r"DFT $\Delta G^{\ddag}$ (kcal mol$^{-1}$)", shrink=0.9, pad=0.02)
    plt.savefig(f"{output_dir}/fig_design_rules.png", dpi=200, bbox_inches="tight"); plt.close()
    numbers["design_rule_features"] = top2

    # ------------------------------------------------------------------ 7. uncertainty-aware screening
    threshold = np.quantile(y, 0.25); active = y <= threshold; order = np.argsort(gp_mean)
    found = np.cumsum(active[order]) / active.sum(); fraction = np.arange(1, n + 1) / n
    k20 = max(1, int(0.2 * n)); ef20 = (active[order][:k20].sum() / k20) / active.mean()
    cm_order = np.argsort(cell_pred); ef20_cm = (active[cm_order][:k20].sum() / k20) / active.mean()
    numbers["screening"] = dict(EF20=round(float(ef20), 1), recovered20=round(float(found[k20 - 1] * 100)), EF20_cellmean=round(float(ef20_cm), 1))
    print(f"\n[screening] EF@20%={ef20:.1f}x (cell-mean ranking {ef20_cm:.1f}x); top-20% recovers {found[k20-1]*100:.0f}% of the most active")
    data["priority"] = (-(gp_mean - gp_mean.mean()) / gp_mean.std()) + (gp_std - gp_std.mean()) / gp_std.std()
    data.sort_values("pred")[["barrier_id", "system", "metal", "nucl", "pathway", "step", "phase", "charge", "pred", "sd", "dG"]].to_csv(f"{output_dir}/MAAC_screening_ranking.csv", index=False)
    plt.figure(figsize=(5.2, 4.6)); plt.plot(fraction * 100, found * 100, "-", c="#2b6cb0", lw=2, label="GP ranking")
    found_cm = np.cumsum(active[cm_order]) / active.sum(); plt.plot(fraction * 100, found_cm * 100, "-", c="#888", lw=1.5, label="(metal, step) lookup")
    plt.plot([0, 100], [0, 100], "--", c="gray", label="random"); plt.plot(fraction * 100, np.minimum(fraction * n / active.sum(), 1) * 100, ":", c="green", label="ideal")
    plt.xlabel(r"% screened (by predicted $\Delta G^{\ddag}$)"); plt.ylabel("% of most-active found"); plt.title(f"Uncertainty-aware screening (EF@20%={ef20:.1f}×)"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_screening_enrichment.png", dpi=200); plt.close()

    # ------------------------------------------------------------------ 8. energetic span (Kozuch-Shaik)
    # same system, pathway, environment AND stoichiometry (formula); the acetylide (different formula) is excluded
    def cycle_index(record):
        role = record["role"]
        if role == "TS":
            return 2 if str(record["step"]) in ("1", "1.0") else 4
        return {"RC": 1, "IC": 3, "product": 5}.get(role)
    grouped_states = defaultdict(list)
    for _, record in structures.iterrows():
        if record["calc_type"] != "opt+freq" or pd.isna(record["G"]):
            continue
        index = cycle_index(record)
        if index is None:
            continue
        key = (record["system_id"], record["pathway"], record["phase"], record["solvent"], record["formula"])
        grouped_states[key].append((index, float(record["G"]), record))
    spans = []
    for (system_id, pathway, phase, solvent, formula), states in grouped_states.items():
        minima = {}
        for i, g, _ in states:
            if i in (1, 3, 5):
                minima[i] = min(g, minima.get(i, g))
        tss = {}
        for i, g, _ in states:
            if i in (2, 4):
                tss[i] = min(g, tss.get(i, g))
        if not minima or not tss:
            continue
        candidates = [(tg - min(g for i, g in minima.items() if i < ti)) * HARTREE_TO_KCAL
                      for ti, tg in tss.items() if [g for i, g in minima.items() if i < ti]]
        if candidates and 0 <= max(candidates) <= 60:
            complete = (set(tss) == {2, 4}) and (set(minima) >= {1, 3})
            spans.append(dict(system=system_id, metal=states[0][2]["metal"], pathway=pathway, phase=phase, solvent=solvent,
                              dE=round(max(candidates), 2), complete_cycle=complete, n_TS=len(tss), roles=f"{sorted(minima)}|{sorted(tss)}"))
    span_table = pd.DataFrame(spans)
    span_stats = []
    if len(span_table):
        span_table.sort_values("dE").to_csv(f"{output_dir}/MAAC_energetic_span.csv", index=False)
        comp = span_table[span_table.complete_cycle]
        numbers["span_n_complete"] = int(len(comp)); numbers["span_n_partial"] = int((~span_table.complete_cycle).sum())
        for metal in METAL_ORDER:
            v = comp[comp.metal == metal].dE.values; vall = span_table[span_table.metal == metal].dE.values
            if len(v) == 0:
                continue
            lo, hi = boot_ci(v)
            span_stats.append(dict(metal=metal, n=len(v), mean=round(float(v.mean()), 1), sd=round(float(v.std(ddof=1)), 1) if len(v) > 1 else float("nan"),
                                   ci_low=round(lo, 1), ci_high=round(hi, 1), median=round(float(np.median(v)), 1),
                                   n_incl_partial=len(vall), mean_incl_partial=round(float(vall.mean()), 1)))
        cu = comp[comp.metal == "Cu"].dE.values
        for s_ in span_stats:
            if s_["metal"] != "Cu":
                v = comp[comp.metal == s_["metal"]].dE.values; lo, hi = boot_diff_ci(v, cu)
                s_["diff_vs_Cu"] = round(float(v.mean() - cu.mean()), 1); s_["diff_ci"] = [round(lo, 1), round(hi, 1)]
        pd.DataFrame(span_stats).to_csv(f"{output_dir}/MAAC_energetic_span_by_metal.csv", index=False)
        print(f"\n[energetic span] complete cycles: {len(comp)} (partial profiles, lower bounds only: {int((~span_table.complete_cycle).sum())})")
        print(pd.DataFrame(span_stats).to_string(index=False))
        plt.figure(figsize=(5.2, 3.9)); st_ = pd.DataFrame(span_stats)
        err = np.vstack([st_["mean"] - st_["ci_low"], st_["ci_high"] - st_["mean"]]); err = np.nan_to_num(err)
        plt.bar(st_.metal, st_["mean"], yerr=err, capsize=4, color=[METAL_COLOR[m] for m in st_.metal], edgecolor="k", lw=0.6, zorder=3)
        for i, r in st_.iterrows():
            top = r["ci_high"] if not np.isnan(r["ci_high"]) else r["mean"]
            plt.text(i, top + 0.6, f"n={int(r['n'])}", ha="center", fontsize=9)
        plt.ylabel(r"apparent energetic span $\delta E$ (kcal mol$^{-1}$)"); plt.xlabel("metal")
        plt.title("Turnover-determining span, complete cycles (95% CI)", fontsize=10.5)
        plt.grid(axis="y", alpha=0.25, zorder=0); plt.tight_layout(); plt.savefig(f"{output_dir}/fig_energetic_span.png", dpi=200); plt.close()
    numbers["span"] = span_stats

    # ------------------------------------------------------------------ 9. closed-form baseline (honest)
    linear_features = ["qM", "dNmax", "gap", "omega", "Vbur", "HOMO", "eta", "charge", "eps"]
    scaled = StandardScaler().fit_transform(data[linear_features].values)
    lin_pred = cross_val_predict(LassoCV(cv=5, random_state=SEED), scaled, y, cv=5)
    numbers["lfer_R2"] = round(float(r2_score(y, lin_pred)), 2)
    print(f"\n[closed-form] linear LFER 5-fold R2={numbers['lfer_R2']:.2f}")
    try:
        from gplearn.genetic import SymbolicRegressor
        sym_pred = np.zeros(n)
        for train, test in KFold(5, shuffle=True, random_state=SEED).split(scaled):
            reg = SymbolicRegressor(population_size=600, generations=10, function_set=("add", "sub", "mul", "div"),
                                    parsimony_coefficient=0.02, random_state=SEED).fit(scaled[train], y[train])
            sym_pred[test] = reg.predict(scaled[test])
        numbers["symbolic_R2"] = round(float(r2_score(y, sym_pred)), 2)
        print(f"              symbolic regression 5-fold R2={numbers['symbolic_R2']:.2f}")
    except Exception:
        numbers["symbolic_R2"] = None

    json.dump(numbers, open(f"{output_dir}/MAAC_numbers.json", "w"), indent=1, default=float)
    print("\n" + "=" * 60); print(f"DONE - all results are in {output_dir}/"); print("=" * 60)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def main():
    ensure_packages()
    preferred = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
    database_dir = get_database(preferred)
    run_analysis(database_dir, "MAAC_outputs")
    download_outputs("MAAC_outputs")


if __name__ == "__main__":
    main()
