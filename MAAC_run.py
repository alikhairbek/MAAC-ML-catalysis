#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAAC-ML : reproduce the full machine-learning analysis of metal-catalyzed
azide-alkyne cycloaddition (MAAC) directly from the curated DFT database.

Just run this one file:

    python MAAC_run.py

It does everything automatically:
  1. Installs any missing Python packages.
  2. Finds the database. If no unzipped "MAAC_database/" folder is present it
     looks for "MAAC_database.zip" and unzips it; if the zip is missing and the
     script runs on Google Colab, it asks you to upload "MAAC_database.zip".
  3. Runs the analysis and writes all figures and tables to "MAAC_outputs/":
        - model comparison (leave-one-out)
        - Gaussian-process model with calibrated uncertainty
        - grouped cross-validation (applicability domain)
        - SHAP design rules and mean barrier per metal
        - uncertainty-aware virtual screening
        - energetic-span ranking (Kozuch-Shaik)
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

    # A local MAAC_database.zip next to the script
    if os.path.exists("MAAC_database.zip"):
        unzip("MAAC_database.zip")
        database = find_database_dir(preferred)
        if database:
            return database

    # On Colab, ask the user to upload the archive
    if running_in_colab():
        print("\nMAAC_database.zip was not found - please upload it now.")
        from google.colab import files
        files.upload()                      # the user selects MAAC_database.zip
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


def run_analysis(database_dir, output_dir):
    """Reproduce every figure and table from the database."""
    import csv
    import warnings
    from collections import defaultdict

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})

    from sklearn.model_selection import (LeaveOneOut, GroupKFold,
                                         cross_val_predict, KFold)
    from sklearn.linear_model import Ridge, LassoCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF, WhiteKernel,
                                                  ConstantKernel as Constant)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import norm
    from xgboost import XGBRegressor
    import shap

    os.makedirs(output_dir, exist_ok=True)

    def to_float(record, key):
        try:
            return float(record[key])
        except (KeyError, ValueError, TypeError):
            return np.nan

    # ---- load the database ----
    with open(os.path.join(database_dir, "structures.csv")) as handle:
        structures = {row["structure_id"]: row for row in csv.DictReader(handle)}
    with open(os.path.join(database_dir, "barriers.csv")) as handle:
        barriers = list(csv.DictReader(handle))
    print(f"loaded {len(structures)} structures, {len(barriers)} barriers from {database_dir}")

    # ---- assemble the modeling table ----
    rows = []
    for barrier in barriers:
        reactant = structures.get(barrier["reactant_id"])
        if not reactant:
            continue
        row = dict(dG=float(barrier["dG_act_kcal"]), step=int(float(barrier["step"])),
                   metal=barrier["metal"], nucl=barrier["nuclearity"] or "mono",
                   system=barrier["system_id"], pathway=barrier["pathway"])
        for key in ELECTRONIC_DESCRIPTORS:
            row[key] = to_float(reactant, key)
        row["Vbur"] = to_float(reactant, "pct_Vbur")
        row["qM"] = to_float(reactant, "q_metal")
        rows.append(row)

    data = (pd.DataFrame(rows)
            .dropna(subset=ELECTRONIC_DESCRIPTORS + ["Vbur", "qM", "dG"])
            .reset_index(drop=True))
    data.to_csv(os.path.join(output_dir, "MAAC_modeling_table.csv"), index=False)

    y = data["dG"].values
    feature_cols = ELECTRONIC_DESCRIPTORS + ["Vbur", "qM", "step", "metal", "nucl"]
    X = pd.get_dummies(data[feature_cols], columns=["metal", "nucl"]).astype(float)
    X_values = X.values
    print(f"modeling set: n={len(y)} features={X_values.shape[1]} "
          f"target {y.mean():.1f}+/-{y.std():.1f} kcal/mol")

    def make_xgb():
        return XGBRegressor(n_estimators=160, max_depth=3, learning_rate=0.06,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2,
                            random_state=0)

    def leave_one_out(make_model, scale=False):
        prediction = np.zeros(len(y))
        for train, test in LeaveOneOut().split(X_values):
            x_train, x_test = X_values[train], X_values[test]
            if scale:
                scaler = StandardScaler().fit(x_train)
                x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
            model = make_model().fit(x_train, y[train])
            prediction[test] = model.predict(x_test)[0]
        return prediction

    # ---- model comparison (leave-one-out) ----
    print("\n[model comparison - leave-one-out]")
    print(f"  {'model':22s}{'MAE':>7s}{'R2':>7s}")
    comparison = [
        ("Ridge (linear)", lambda: Ridge(alpha=5), True),
        ("RandomForest", lambda: RandomForestRegressor(400, max_depth=6, random_state=0), False),
        ("GradientBoosting", lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
            random_state=0), False),
        ("XGBoost", make_xgb, False),
        ("Neural Net (MLP)", lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=2000, random_state=0), True),
    ]
    for name, make_model, scale in comparison:
        try:
            prediction = leave_one_out(make_model, scale)
            print(f"  {name:22s}{mean_absolute_error(y, prediction):7.2f}"
                  f"{r2_score(y, prediction):7.2f}")
        except Exception as error:
            print(f"  {name:22s}  skipped ({str(error)[:30]})")

    # ---- primary Gaussian-process model with calibrated uncertainty ----
    def make_kernel():
        return (Constant(1.0, (1e-2, 1e2)) * RBF(5.0, (0.5, 50))
                + WhiteKernel(1.0, (1e-3, 1e2)))

    gp_mean = np.zeros(len(y))
    gp_std = np.zeros(len(y))
    for train, test in LeaveOneOut().split(X_values):
        scaler = StandardScaler().fit(X_values[train])
        gp = GaussianProcessRegressor(kernel=make_kernel(), normalize_y=True,
                                      alpha=1e-6, n_restarts_optimizer=0)
        gp.fit(scaler.transform(X_values[train]), y[train])
        mean, std = gp.predict(scaler.transform(X_values[test]), return_std=True)
        gp_mean[test], gp_std[test] = mean[0], std[0]
    data["pred"] = gp_mean
    data["sd"] = gp_std

    print(f"\n[Gaussian process - primary] R2={r2_score(y, gp_mean):.2f}  "
          f"MAE={mean_absolute_error(y, gp_mean):.2f}  mean_sigma={gp_std.mean():.2f}")
    within_1sigma = np.mean(np.abs(y - gp_mean) <= gp_std) * 100
    within_2sigma = np.mean(np.abs(y - gp_mean) <= 2 * gp_std) * 100
    print(f"  calibration: {within_1sigma:.0f}% within 1 sigma (ideal 68), "
          f"{within_2sigma:.0f}% within 2 sigma (ideal 95)")

    # ---- grouped cross-validation (extrapolation to unseen families) ----
    groups = np.array([system.split("_")[0] for system in data["system"]])
    grouped_prediction = np.zeros(len(y))
    n_splits = min(6, len(set(groups)))
    for train, test in GroupKFold(n_splits).split(X_values, y, groups):
        model = make_xgb().fit(X_values[train], y[train])
        grouped_prediction[test] = model.predict(X_values[test])
    print(f"  grouped-CV (unseen families): R2={r2_score(y, grouped_prediction):.2f}")

    # ---- figures: parity, calibration, per-metal ----
    step_color = {1: "#2b6cb0", 2: "#dd6b20"}
    limits = [min(y.min(), gp_mean.min()) - 3, max(y.max(), gp_mean.max()) + 3]

    plt.figure(figsize=(5.4, 5.2))
    for step in (1, 2):
        mask = data["step"] == step
        plt.errorbar(data["dG"][mask], data["pred"][mask], yerr=data["sd"][mask],
                     fmt="o", ms=6, mfc=step_color[step], mec="w",
                     ecolor=step_color[step], elinewidth=1, capsize=2, alpha=0.85,
                     label=f"step {step}", zorder=3)
    plt.plot(limits, limits, "--", c="gray")
    plt.xlim(limits); plt.ylim(limits); plt.legend(frameon=False)
    plt.xlabel(r"DFT $\Delta G^{\ddagger}$ (kcal mol$^{-1}$)")
    plt.ylabel(r"GP prediction $\pm\,\sigma$ (LOO)")
    plt.title(f"Gaussian process: $R^2$={r2_score(y, gp_mean):.2f}, "
              f"MAE={mean_absolute_error(y, gp_mean):.1f} (n={len(y)})")
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_parity_gp.png", dpi=200); plt.close()

    levels = np.linspace(0.05, 0.99, 30)
    coverage = [np.mean(np.abs(y - gp_mean) <= norm.ppf((1 + level) / 2) * gp_std)
                for level in levels]
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", c="gray", label="perfect")
    plt.plot(levels, coverage, "-o", ms=4, c="#2b6cb0", label="GP")
    plt.xlabel("expected confidence"); plt.ylabel("observed coverage")
    plt.title("Uncertainty calibration"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_calibration.png", dpi=200); plt.close()

    metal_color = {"Cu": "#b8860b", "Ag": "#708090", "Au": "#daa520",
                   "Ru": "#2e8b57", "Rh": "#c0392b", "Ni": "#555", "Pd": "#999"}
    plt.figure(figsize=(5.4, 5.2))
    for metal, group in data.groupby("metal"):
        r2 = r2_score(group["dG"], group["pred"]) if len(group) > 2 else float("nan")
        plt.scatter(group["dG"], group["pred"], s=46, edgecolor="w",
                    color=metal_color.get(metal, "#444"),
                    label=f"{metal} (n={len(group)}, $R^2$={r2:.2f})", zorder=3)
    plt.plot(limits, limits, "--", c="gray")
    plt.xlim(limits); plt.ylim(limits)
    plt.legend(frameon=False, fontsize=9, loc="upper left")
    plt.xlabel(r"DFT $\Delta G^{\ddagger}$ (kcal mol$^{-1}$)")
    plt.ylabel("GP prediction (LOO)")
    plt.title("Per-metal reliability (applicability domain)")
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_parity_by_metal.png", dpi=200); plt.close()

    # ---- SHAP design rules and mean barrier per metal ----
    model = make_xgb().fit(X, y)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    continuous = [c for c in X.columns
                  if c in ELECTRONIC_DESCRIPTORS + ["Vbur", "qM", "step"]]
    importance = np.abs(shap_values).mean(0)
    top_two = sorted(continuous,
                     key=lambda col: importance[list(X.columns).index(col)],
                     reverse=True)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, feature in zip(axes, top_two):
        index = list(X.columns).index(feature)
        axis.scatter(X[feature], shap_values[:, index], c=data["dG"],
                     cmap="coolwarm", s=30, edgecolor="w")
        axis.axhline(0, c="gray", lw=0.8)
        axis.set_xlabel(feature); axis.set_ylabel(f"SHAP ({feature})")
        axis.set_title(f"design rule: {feature}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_design_rules.png", dpi=200, bbox_inches="tight")
    plt.close()

    per_metal = (data.groupby("metal")["dG"]
                 .agg(["mean", "std", "count"]).sort_values("mean"))
    plt.figure(figsize=(5, 3.8))
    plt.bar(per_metal.index, per_metal["mean"], yerr=per_metal["std"].fillna(0),
            capsize=4, color="#4a5568", edgecolor="k")
    for position, (_metal, record) in enumerate(per_metal.iterrows()):
        plt.text(position, record["mean"] + 0.4, f"n={int(record['count'])}",
                 ha="center", fontsize=9)
    plt.ylabel(r"mean $\Delta G^{\ddagger}$ (kcal mol$^{-1}$)"); plt.xlabel("metal")
    plt.title("Mean activation barrier per metal")
    plt.tight_layout(); plt.savefig(f"{output_dir}/fig_metal_barriers.png", dpi=200); plt.close()
    print(f"\n[interpretation] top design-rule descriptors: {top_two}")

    # ---- uncertainty-aware virtual screening ----
    threshold = np.quantile(y, 0.25)
    active = y <= threshold
    order = np.argsort(gp_mean)
    found = np.cumsum(active[order]) / active.sum()
    fraction = np.arange(1, len(y) + 1) / len(y)
    top_20 = max(1, int(0.2 * len(y)))
    enrichment = (active[order][:top_20].sum() / top_20) / active.mean()
    print(f"\n[screening] enrichment@20%={enrichment:.1f}x ; "
          f"top-20% recovers {found[top_20 - 1] * 100:.0f}% of the most-active systems")

    data["priority"] = (-(gp_mean - gp_mean.mean()) / gp_mean.std()
                        + (gp_std - gp_std.mean()) / gp_std.std())
    (data.sort_values("pred")[["system", "metal", "nucl", "pathway",
                               "step", "pred", "sd", "dG"]]
     .to_csv(f"{output_dir}/MAAC_screening_ranking.csv", index=False))

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fraction * 100, found * 100, "-", c="#2b6cb0", lw=2, label="GP ranking")
    plt.plot([0, 100], [0, 100], "--", c="gray", label="random")
    plt.plot(fraction * 100,
             np.minimum(fraction * len(y) / active.sum(), 1) * 100,
             ":", c="green", label="ideal")
    plt.xlabel(r"% screened (by predicted $\Delta G^{\ddagger}$)")
    plt.ylabel("% of most-active found")
    plt.title(f"Uncertainty-aware screening (EF@20%={enrichment:.1f}x)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_screening_enrichment.png", dpi=200)
    plt.close()

    # ---- energetic-span ranking (Kozuch-Shaik) ----
    def cycle_index(record):
        if record["role"] == "TS":
            return 2 if str(record["step"]) == "1" else 4
        return {"acetylide": 0, "RC": 1, "IC": 3, "product": 5}.get(record["role"])

    grouped_states = defaultdict(list)
    for record in structures.values():
        energy = to_float(record, "G")
        index = cycle_index(record)
        if not np.isnan(energy) and index is not None:
            key = (record["system_id"], record["pathway"], record["solvent"])
            grouped_states[key].append((index, energy, record))

    spans = []
    for (system_id, pathway, solvent), states in grouped_states.items():
        minima = [(i, g) for i, g, _ in states if i in (0, 1, 3, 5)]
        transition_states = [(i, g) for i, g, _ in states if i in (2, 4)]
        if not minima or not transition_states:
            continue
        candidates = [(ts_g - min(g for i, g in minima if i < ts_i)) * HARTREE_TO_KCAL
                      for ts_i, ts_g in transition_states
                      if [g for i, g in minima if i < ts_i]]
        if candidates and 0 <= max(candidates) <= 60:
            spans.append(dict(system=system_id, metal=states[0][2]["metal"],
                              pathway=pathway, solvent=solvent,
                              dE=round(max(candidates), 2)))

    span_table = pd.DataFrame(spans)
    if len(span_table):
        span_table.sort_values("dE").to_csv(
            f"{output_dir}/MAAC_energetic_span.csv", index=False)
        by_metal = (span_table.groupby("metal")["dE"]
                    .agg(["count", "mean", "std"]).sort_values("mean"))
        print("\n[energetic span] apparent dE per metal (lower = faster turnover):")
        for metal, record in by_metal.iterrows():
            print(f"   {metal}: n={int(record['count'])}  dE={record['mean']:.1f} kcal/mol")
        plt.figure(figsize=(5, 3.9))
        plt.bar(by_metal.index, by_metal["mean"], yerr=by_metal["std"].fillna(0),
                capsize=4, color="#4a5568", edgecolor="k")
        for position, (_metal, record) in enumerate(by_metal.iterrows()):
            plt.text(position, record["mean"] + 0.4, f"n={int(record['count'])}",
                     ha="center", fontsize=9)
        plt.ylabel(r"apparent energetic span $\delta E$ (kcal mol$^{-1}$)")
        plt.xlabel("metal")
        plt.title("TOF-determining span per metal")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_energetic_span.png", dpi=200)
        plt.close()

    # ---- closed-form / symbolic baseline (reported honestly) ----
    linear_features = ["qM", "dNmax", "gap", "omega", "Vbur", "HOMO", "eta"]
    scaled = StandardScaler().fit_transform(data[linear_features].values)
    linear_prediction = cross_val_predict(LassoCV(cv=5, random_state=0), scaled, y, cv=5)
    lasso = LassoCV(cv=5, random_state=0).fit(scaled, y)
    terms = [f"{coefficient:+.2f}*{name}"
             for coefficient, name in zip(lasso.coef_, linear_features)
             if abs(coefficient) > 0.05]
    print(f"\n[closed-form] linear LFER: dG-act ~ {y.mean():.1f} {' '.join(terms)} | "
          f"5-fold R2={r2_score(y, linear_prediction):.2f} (weak -> GP preferred)")
    try:
        from gplearn.genetic import SymbolicRegressor
        symbolic_prediction = np.zeros(len(y))
        for train, test in KFold(5, shuffle=True, random_state=0).split(scaled):
            regressor = SymbolicRegressor(
                population_size=1000, generations=12,
                function_set=("add", "sub", "mul", "div"),
                parsimony_coefficient=0.02, random_state=0).fit(scaled[train], y[train])
            symbolic_prediction[test] = regressor.predict(scaled[test])
        print(f"              symbolic regression: 5-fold "
              f"R2={r2_score(y, symbolic_prediction):.2f} (also weak - multivariate)")
    except Exception:
        print("              (gplearn not available - symbolic step skipped)")

    print("\n" + "=" * 60)
    print(f"DONE - all results are in {output_dir}/ (8 figures + 3 tables)")
    print("=" * 60)


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
