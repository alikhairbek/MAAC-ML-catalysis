[README.md](https://github.com/user-attachments/files/28648887/README.md)
# MAAC-ML: Machine Learning of Metal-Catalyzed Azide–Alkyne Cycloaddition

A compact, fully reproducible package containing **(i) a consistent DFT database** of metal-catalyzed azide–alkyne ("click") cycloaddition (MAAC) and **(ii) a single script** that rebuilds the complete machine-learning analysis — calibrated activation-barrier prediction, mechanistic design rules, uncertainty-aware screening, and an energetic-span ranking — directly from that database.

This repository accompanies the article *"Machine Learning of Metal-Catalyzed Azide–Alkyne Cycloaddition from a Consistent DFT Dataset: Calibrated Activation-Barrier Prediction, Mechanistic Design Rules, and Uncertainty-Aware Screening"* (Khairbek et al.).

---

## What this is

Click chemistry — the copper- and metal-catalyzed azide–alkyne cycloaddition that builds 1,2,3-triazoles — is one of the most widely used reactions in chemistry. Its activation barriers, however, are scattered across many independent computational studies that cannot be compared on a common footing. This project fixes that:

- A **consistent dataset** assembled by reprocessing twelve published DFT studies onto a *single* level of theory (**MN12-L/def2-SVP, def2-TZVP for metals, continuum solvation, Gaussian 16**), with **no new calculations**.
- An **interpretable Gaussian-process model** that predicts the activation free energy **with calibrated uncertainty**, valid inside a clearly defined applicability domain.
- Honest validation: an optimistic interpolation metric (leave-one-out) **and** a stringent extrapolation metric (grouped cross-validation by source study).

**Scope:** restricted to metal-catalyzed azide–alkyne **[3+2] cycloaddition** across five metals (Cu, Ag, Au, Ru, Rh). It is *not* a universal reactivity predictor.

---

## Repository contents

```
.
├── README.md
├── MAAC_run.py                  ← the single script — just run it; everything is automatic
├── MAAC_database.zip            ← the consistent MAAC DFT database (the script unzips it for you)
├── requirements.txt
└── LICENSE                      ← MIT (code) + CC BY 4.0 (data)
```

### Inside `MAAC_database.zip`

| File | Description |
|------|-------------|
| `structures.csv` | 847 fully characterized stationary points: energies (G/H/SCF), conceptual-DFT/Koopmans electronic descriptors, `%Vbur`, metal partial charge, imaginary-frequency count, geometry |
| `barriers.csv` | 127 mechanism-resolved elementary-step activation free energies, ΔG‡₁ = G(TS1)−G(RC) and ΔG‡₂ = G(TS2)−G(IC) (same-stoichiometry referencing) |
| `systems.csv` | 128 catalytic systems |
| `papers.csv` | 11 source DFT studies with DOIs |
| `MAAC_catalysis.sqlite` | the same data as a relational SQLite database (4 linked tables) |
| `schema.sql`, `README.md` | schema and data dictionary |
| `geometries/` | 847 optimized structures (`.xyz`) |

Dataset composition: Cu (592), Ag (111), Ru (51), Rh (48), Au (35) structures; 199 are confirmed transition states. Every record carries its source paper and DOI, so the dataset is **F**indable, **A**ccessible, **I**nteroperable and **R**eusable (FAIR).

> **Note on approximations (stated transparently):** metal partial charges are Mulliken (the original logs lack NBO populations) and SMILES/InChI are perceived from the optimized geometries (approximate for organometallics). Neither affects the DFT energies, which are the learning target.

---

## How to run

The script does everything for you — installs packages, locates and unzips the database, runs the analysis, and (on Colab) downloads the results. It needs Python 3.9+.

### Locally (≈2–3 minutes on a normal CPU)

Put `MAAC_run.py` and `MAAC_database.zip` in the same folder, then:

```bash
python MAAC_run.py
```

It installs any missing packages, unzips `MAAC_database.zip` automatically, and writes every figure and table to `MAAC_outputs/`. A `requirements.txt` is included for reference (`numpy, pandas, scikit-learn, matplotlib, scipy, xgboost, shap, gplearn`); no GPU, internet, `openbabel`, or `morfeus` is needed because every descriptor is already stored in the database.

### Google Colab

Upload `MAAC_run.py` (drag it into the file panel) and run a single cell:

```python
!python MAAC_run.py
```

If `MAAC_database.zip` is not already present, the script prompts you to upload it, then continues automatically and downloads `MAAC_outputs.zip` at the end. (You can also paste the contents of `MAAC_run.py` straight into a cell and run it.)

---

## Outputs

All results are written to a new `MAAC_outputs/` folder.

**Figures (200 dpi PNG):**

| File | Content |
|------|---------|
| `fig_metal_barriers.png` | Mean activation barrier per metal |
| `fig_parity_gp.png` | Gaussian-process parity plot with ±1σ error bars (leave-one-out) |
| `fig_calibration.png` | Uncertainty calibration (observed vs expected coverage) |
| `fig_parity_by_metal.png` | Per-metal reliability (applicability domain) |
| `fig_shap_summary.png` | SHAP descriptor-importance ranking |
| `fig_design_rules.png` | SHAP dependence plots → design rules |
| `fig_screening_enrichment.png` | Uncertainty-aware virtual-screening enrichment curve |
| `fig_energetic_span.png` | Apparent energetic span (δE) per metal |

**Tables (CSV):** `MAAC_modeling_table.csv` (the assembled feature/target table), `MAAC_screening_ranking.csv` (systems ranked by predicted barrier with uncertainty), `MAAC_energetic_span.csv` (per-system δE).

**Console summary:** model-comparison metrics, GP accuracy and calibration, grouped-CV R², SHAP top features, screening enrichment factor, and the per-metal energetic-span ranking.

---

## Key results (reproduced by the script)

- **Mean elementary barrier by metal:** Cu 8.4 < Ru 9.3 < Au 13.8 < Ag 17.8 < Rh 18.5 kcal mol⁻¹ — copper is the most efficient, consistent with experiment.
- **Model comparison (leave-one-out):** Ridge 0.34 ≪ {RandomForest 0.68, GradientBoosting 0.66, XGBoost 0.69, MLP 0.77} ≈ **Gaussian process 0.76** → accuracy is *data-limited*, not model-limited.
- **Primary GP model:** R² = 0.76, MAE = 3.22 kcal mol⁻¹, with **calibrated uncertainty** (73% of points within ±1σ, 94% within ±2σ).
- **Applicability domain (per-metal LOO):** Cu R² = 0.78 (MAE 2.29), Ag 0.75, Au 0.60, Rh 0.54, Ru 0.05; grouped cross-validation across unseen families R² = 0.21.
- **Design rules (SHAP):** dominated by reaction step, metal partial charge, charge-transfer index ΔN_max, and metal identity.
- **Uncertainty-aware screening:** 3.1× enrichment of low-barrier systems in the top 20% (no transition-state search needed).
- **Energetic span (Kozuch–Shaik):** apparent δE Cu 10.6 < Ag 22.0 ≈ Au 22.5 < Rh 30.4 kcal mol⁻¹.

---

## What the script does (pipeline)

1. Loads `structures.csv` + `barriers.csv` and assembles an 18-feature modeling table (8 electronic descriptors + `%Vbur` + metal charge + reaction step + metal + nuclearity).
2. Benchmarks six regression families under leave-one-out cross-validation.
3. Fits the primary Gaussian-process model, produces leave-one-out predictions **with uncertainty**, and tests calibration.
4. Runs grouped cross-validation by source study (honest extrapolation limit).
5. Computes SHAP values and design-rule dependence plots.
6. Performs uncertainty-aware virtual screening (enrichment + ranking).
7. Computes the energetic span per system and aggregates by metal.

---

## License & citation

- **Code** (`MAAC_run.py`): MIT License.
- **Data** (`MAAC_database.zip` and all contents): Creative Commons Attribution 4.0 (CC BY 4.0).

If you use this dataset or code, please cite the associated article (citation details to be completed upon publication).

## Contact

Ali A. Khairbek — alikhairbek@gmail.com
