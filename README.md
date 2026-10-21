# MAAC-ML: a consistent DFT dataset and machine-learning benchmark for metal-catalyzed azide–alkyne cycloaddition

A compact, fully reproducible package containing **(1) a consistent DFT database** of metal-catalyzed azide–alkyne ("click") cycloaddition (MAAC) and **(2) a single script** that rebuilds the complete machine-learning analysis — reference baselines, calibrated activation-barrier prediction, step-resolved metal statistics, mechanistic design rules, uncertainty-aware screening, and an energetic-span ranking — directly from that database.

This repository accompanies the article *"A Consistent Quantum-Chemical Dataset of Metal-Catalyzed Azide–Alkyne Cycloaddition: Step-Resolved Metal Ordering and Calibrated Machine Learning Benchmarked against Metal- and Step-Specific Baselines"* (Khairbek et al.).

> **Database version 15 (revision of October 2026).** The solvation labels of v12 were wrong, 37 duplicate files were present, the dicopper catalysts of ref. 18 were labelled mononuclear, two rhodium systems had their 1,4/1,5 channels mixed, five "transition states" did not describe the reaction, and the ref. 19 archive contained nine systems that are not part of the published study (see *Changelog* below). Every structure now carries its true computational environment, every stationary point enters once and is traceable to a published paper, every transition state was verified from its imaginary normal mode, and every catalyst and intermediate drawn in the paper was verified against its geometry (750 structures, 91 barriers). Please use v15.

---

## What this is

Click chemistry — the copper- and metal-catalyzed azide–alkyne cycloaddition that builds 1,2,3-triazoles — is one of the most widely used reactions in chemistry. Its activation barriers, however, are scattered across many independent computational studies that cannot be compared on a common footing. This project addresses that:

- A **consistent dataset** assembled by reprocessing twelve published DFT studies of the same model reaction (benzyl azide + phenylacetylene) computed with **one functional and basis set (MN12-L/def2-SVP, def2-TZVP for metals, Gaussian 16)**, with **no new calculations**. The solvation environment is *not* uniform across the source studies (gas-phase optimizations or C-PCM in toluene/water/THF/methanol/acetonitrile); it is recorded for every entry and carried as a feature.
- An **interpretable Gaussian-process model** that predicts the activation free energy **with calibrated uncertainty**, compared against **reference baselines** (global mean, (metal, step) lookup, one-hot linear) so that the contribution of the descriptors is stated honestly.
- Honest validation: an optimistic interpolation metric (leave-one-out), a stringent extrapolation metric (grouped cross-validation by source study), a leave-one-study-out transferability map, and single-environment refits.

**Scope:** restricted to metal-catalyzed azide–alkyne **[3+2] cycloaddition** across five metals (Cu, Ag, Au, Ru, Rh). It is *not* a universal reactivity predictor; only copper is predicted quantitatively (see *Key results*).

---

## Repository contents

```
.
├── README.md
├── MAAC_run.py                  ← the single analysis script — just run it; everything is automatic
├── MAAC_database.zip            ← the MAAC DFT database, v15 (the script unzips it for you)
├── verification/                ← the verification tables behind Tables S4, S6, S7 and S8 of the paper (data, not code)
│   ├── MAAC_ts_characterization.csv      ← formed/forming N–C distances, imaginary frequency and mode criteria of every barrier (Table S6)
│   ├── MAAC_rc_ic_survey.csv             ← metal–azide / metal–carbon distances of every reactant complex and metallacycle (Table S4)
│   ├── MAAC_rc_ic_summary.json           ← the same, summarized per metal and channel
│   ├── scheme1_verification.csv          ← every structure drawn in Scheme 1 / Figure 1 vs. the deposited geometry (Table S8)
│   └── MAAC_SI_coordinate_matching.csv   ← match of the coordinates printed in the source-study SI to the database (Table S7)
├── requirements.txt
└── LICENSE                      ← MIT (code) + CC BY 4.0 (data)
```

The database was derived from the raw Gaussian output files of the source studies by the authors' processing scripts (log parsing, de-duplication, transition-state verification from the imaginary normal mode, structure drawing and verification); the raw outputs and those scripts are not distributed here (available from the authors on request). `MAAC_run.py` needs only the database and regenerates every number, figure and table of the paper.

### Inside `MAAC_database.zip`

| File | Description |
|------|-------------|
| `structures.csv` | 750 stationary points: energies (G/H/SCF), conceptual-DFT/Koopmans electronic descriptors, `%Vbur`, metal partial charge, **phase / solvent / dielectric constant (`eps`) / calculation type**, charge and multiplicity, number and value of imaginary frequencies, geometry file, source file, and (where matched) the label of the same structure in the Supporting Information of the source study (`si_label`) |
| `barriers.csv` | **91** mechanism-resolved elementary-step activation free energies, ΔG‡₁ = G(TS1)−G(RC) and ΔG‡₂ = G(TS2)−G(IC) (same-stoichiometry, same-channel referencing), each with its environment, charge, imaginary frequency, per-step reaction free energy, and the transition-state verification (formed/forming N–C distances, N···C amplitude of the imaginary mode and its ratio to the largest heavy-atom pair displacement) |
| `barriers_excluded.csv` | the 16 candidate barriers that fail the criteria (environment, frequencies, duplicate, geometry, imaginary mode), with the reason |
| `ts_verification.csv` | the geometric and normal-mode verification of every barrier (atom indices of N1/N3 and of the alkyne carbons, formed and forming distances, mode criteria, verdict) |
| `duplicates_removed.csv` | the 37 duplicate output files removed in v14, with the entry that was kept |
| `systems.csv` | 118 catalytic systems with chemical description of the catalyst, ligand family, substrates, environments and barrier count |
| `removed_unpublished.csv` | the 9 systems (60 structures) of the ref. 19 archive that are not part of the published study and were removed in v15 |
| `papers.csv` | the 12 source DFT studies with DOIs, reference number in the paper, functional/basis, environments, charge states and contributions |
| `MAAC_catalysis.sqlite`, `MAAC_database.json` | the same data as a relational SQLite database (8 tables) and as nested JSON |
| `schema.sql`, `README.md` | schema and data dictionary |
| `geometries/` | 750 optimized structures (`.xyz`) |

Dataset composition: Cu (497), Ag (109), Ru (49), Rh (46), Au (35) structures plus 14 metal-free species; 493 are full optimizations with frequencies and 257 are C-PCM single points on gas-phase geometries (no free energy). Barrier set: Cu 51, Rh 14, Ag 13, Ru 8, Au 5 (56 gas-phase, 35 C-PCM-toluene; 52 neutral, 39 cationic). Role labels: `acetylide` (Cu, Ag, Au), `alkyne complex` (π-bound intact alkyne: refs. 21, 22, 24, 27), `RC`, `TS`, `IC`, `product`, free `azide`/`alkyne`, `catalyst`, `other`. Every record carries its source paper and DOI, so the dataset is **F**indable, **A**ccessible, **I**nteroperable and **R**eusable (FAIR).

> **Note on approximations (stated transparently):** metal partial charges are Mulliken (the original outputs lack NBO populations; for dinuclear species `q_metal` and `pct_Vbur` are the mean over the two metal centres, the first-atom values of v13 are kept in `q_metal_v13_first_atom` / `pct_Vbur_v13_first_atom`) and SMILES/InChI are perceived from the optimized geometries (approximate for organometallics). Neither affects the DFT energies, which are the learning target. The column `dG_to_product_kcal` in `barriers.csv` is the free energy of the final product relative to the precursor of the step (kept from earlier versions); the per-step reaction free energy is `dG_step_rxn_kcal`.

---

## How to run

The script does everything for you — installs packages, locates and unzips the database, runs the analysis, and (on Colab) downloads the results. It needs Python 3.9+.

### Locally (≈4–5 minutes on a normal CPU)

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

If `MAAC_database.zip` is not already present, the script prompts you to upload it, then continues automatically and downloads `MAAC_outputs.zip` at the end.

---

## Outputs

All results are written to a new `MAAC_outputs/` folder.

**Figures (200 dpi PNG):**

| File | Content |
|------|---------|
| `fig_metal_barriers.png` | Mean activation barrier per metal, resolved by step, with 95% bootstrap CIs (Figure 2) |
| `fig_parity_gp.png` | Gaussian-process parity plot with ±1σ error bars, leave-one-out (Figure 3) |
| `fig_calibration.png` | Uncertainty calibration under leave-one-out and leave-one-study-out (Figure 4) |
| `fig_parity_by_metal.png` | Per-metal reliability / applicability domain (Figure 5) |
| `fig_shap_summary.png` | SHAP descriptor-importance ranking (Figure 6) |
| `fig_design_rules.png` | SHAP dependence plots → design rules (Figure 7) |
| `fig_screening_enrichment.png` | Uncertainty-aware virtual-screening enrichment curve, with the lookup baseline (Figure 8) |
| `fig_energetic_span.png` | Apparent energetic span δE per metal, complete cycles, 95% CIs (Figure 9) |

**Tables (CSV):** `MAAC_modeling_table.csv` (feature/target table), `MAAC_model_comparison.csv` (baselines + six models, Table 4), `MAAC_metal_step_statistics.csv` and `MAAC_metal_pairwise.csv` (Tables S9–S10), `MAAC_subsets.csv` (Table S11), `MAAC_sigma_bins.csv` (Table S12), `MAAC_leave_one_study_out.csv` (Table 5), `MAAC_screening_ranking.csv` (Table S14), `MAAC_energetic_span.csv` and `MAAC_energetic_span_by_metal.csv` (Table S15, Figure 9), `MAAC_shap_importance.csv` (Table S13), and `MAAC_numbers.json` (every number quoted in the paper). `MAAC_rc_ic_survey.csv` (metal–azide / metal–carbon distances of every RC and IC, Table S4) is provided in `verification/`.

---

## Key results (reproduced by the script, database v15)

- **Step-resolved mean barriers (kcal mol⁻¹, 95% bootstrap CI):** TS1 — Ru 6.5 < Rh 11.5 < Cu 15.0 < Ag 21.9 ≈ Au 22.5; TS2 — Cu 2.2 (Ag 1.5 and Au −1.3, one verified ring closure each) ≪ Ru 12.1 < Rh 18.3 (the 1,4 ring closures on Rh cost 34–36). Copper is moderate in both steps; pooled means are composition-weighted and are not a ranking.
- **Reference baselines (leave-one-out):** (metal, step) cell-mean lookup R² = 0.50, MAE = 4.53; additive one-hot linear model R² = 0.30; GP on the ten continuous descriptors only R² = 0.56; GP on descriptors + step (no metal indicators) R² = 0.75.
- **Model comparison (leave-one-out):** Ridge 0.32 ≪ {RandomForest 0.62, GradientBoosting 0.61, XGBoost 0.67, MLP 0.66} < **Gaussian process 0.70** (MAE 3.75, RMSE 5.08). The GP improves on the lookup by ΔR² = 0.20 (95% CI 0.04 to 0.36) and ΔMAE = 0.78 kcal mol⁻¹ (CI −0.02 to 1.64) — a modest increment, resolved in R² but not in MAE.
- **Calibration:** 71% of points within ±1σ, 96% within ±2σ (mean σ 5.27). Fitted kernel on the full set: σ_f² = 1.23, ℓ = 2.49, σ_n² = 0.14 (standardized-target units). Adding charge and ε as features changes the accuracy only within its uncertainty (R² 0.74 without them).
- **Applicability domain:** per-metal LOO — Cu R² = 0.77 (MAE 2.71, n = 51); Ag 0.28, Au −0.09, Ru 0.05, Rh 0.68 (n = 5–14). Grouped CV R² = 0.18; leave-one-study-out GP R² = 0.18, MAE = 5.54, with calibrated coverage (81% / 96% within 1σ / 2σ, mean σ 7.75).
- **Reactant complexes:** the azide is coordinated through N1 (M–N1 < 1.15 × Σ covalent radii) in 42 of 52 Cu, all Ru and Rh, 1 of 14 Ag and 0 of 4 Au reactant complexes (`MAAC_rc_ic_survey.csv`); in the Ru/Rh 1,5 metallacycles the vinylic C–H carbon stays bound to the metal (η²-vinyl, M–C 2.07–2.12 Å).
- **Design rules (SHAP):** reaction step, HOMO–LUMO gap of the precursor, metal partial charge, ΔN_max, %Vbur, copper identity, HOMO energy.
- **Uncertainty-aware screening:** 3.5× enrichment of low-barrier systems in the top 20% (lookup: 2.9×).
- **Energetic span (Kozuch–Shaik), complete cycles only:** Cu 12.0 (n = 15) ≈ Ru 12.9 (n = 4) < Ag 20.5 (n = 1), Rh 22.5 (n = 8), Au 22.7 (n = 3) kcal mol⁻¹; Cu and Ru are not resolved from each other.

---

## Changelog

**v15 (October 2026)**
- The ref. 19 (NJC) archive contained nine catalytic systems that are not part of the published study (copper with thioanisole (L6) and HCN (L8) as ligands, and 4-substituted benzyl azides / phenylacetylenes with L8): 60 structures and 18 candidate barriers (16 of which had entered the v14 modelling set). They were removed so that every entry is traceable to a published paper; they are listed in `removed_unpublished.csv`. 810 → 750 structures, 107 → 91 barriers, 127 → 118 systems. The `published_in_source` flag of v14 is therefore gone.
- Role label `alkyne complex` for the 32 catalyst–substrate species in which the alkyne is π-bound and intact (refs. 21, 22, 24, 27); they had inherited the label `acetylide`.
- `verification/MAAC_rc_ic_survey.csv`: metal–azide and metal–carbon distances of every reactant complex and metallacycle from the deposited geometries (basis of the RC definition in the paper).
- Scheme 1 and Figure 1 redrawn from SMILES with a purpose-written renderer; every drawing verified against the deposited geometry by fragment graph (element + hydrogen count) and by metal coordination (`verification/scheme1_verification.csv`).

**v14 (October 2026)**
- 37 exact duplicate output files removed (same calculation stored in two folders or used by two systems; e.g. the ref. 25 IMe–Cu calculations also stored as "L7" in the ref. 19 archive). Listed in `duplicates_removed.csv`; 847 → 810 structures.
- Nuclearity of the Com-X catalysts of ref. 18 corrected to dinuclear (two Cu atoms, one CX₂-bridged bis(α-diimine) ligand); system ids `AOC_Cu_bi_*`.
- The CTC (ref. 21) structures split into the Cp*RuCl and CpRuCl systems; metal-free species (free azides, alkynes, triazoles) carry no metal label.
- Rh/Cp^SO₃H and Cp^SiF₃ (ref. 24): 1,4 / 1,5 channel of every RC, IC, product and TS2 read from the file names; the three transition states that had been referenced to the intermediate of the other channel are now referenced to their own channel.
- Transition-state verification from the imaginary normal mode of every TS (`ts_verification.csv`): the N···C displacement must be ≥ 0.15 of the normalized mode and ≥ 0.4 of the largest heavy-atom pair displacement, and the geometry must match the step. Five barriers excluded (four ring-closure saddle points of 6–30i cm⁻¹ with methyl-rotation modes; one "TS2" that is a first-bond-formation geometry), plus five duplicate barriers. 117 → 107 barriers.
- `q_metal` and `pct_Vbur` are the mean over the metal centres for dinuclear species (v13 used the first metal atom of the input; those values are kept in `*_v13_first_atom`).
- `systems.csv`: `published_in_source` flag (superseded in v15, where the flagged systems are removed); exact chemical descriptions of every catalyst, verified against the geometries.

**v13 (October 2026)**
- Solvent labels re-derived from the polarizable-continuum section of every Gaussian output. The former label `implicit` was an extraction artifact (triggered by the internal `SCRF=Check` route of the frequency step) and had been assigned to gas-phase calculations. True distribution: 429 gas-phase, 418 C-PCM (toluene 285, water 69, THF 26, methanol 24, acetonitrile 14), of which 281 are single points.
- New structure columns: `phase`, `eps`, `calc_type`, `imag_freq_cm`, `si_label`, `si_match_dev_A`; `G`/`H` blanked for single points; 16 metal-containing species formerly labelled `alkyne` relabelled `acetylide`.
- Barrier set rebuilt under strict criteria (same environment for precursor and TS; TS with exactly one imaginary frequency; precursor with none; free energies available): 117 barriers kept, 10 excluded and listed in `barriers_excluded.csv`. Barrier values were unchanged in v13; new columns `phase`, `solvent`, `eps`, `charge`, `imag_freq_cm`, `dG_step_rxn_kcal`; former `dG_rxn_kcal` renamed `dG_to_product_kcal`.
- `systems.csv` / `papers.csv`: chemical descriptions of catalysts, ligand families, substrates, environments, charge states, reference numbers.
- `MAAC_run.py` extended: reference baselines, paired bootstrap, per-metal/per-step statistics with CIs, subset refits, σ-binned errors, stoichiometry-checked energetic spans on complete cycles, `MAAC_numbers.json`.

**v12 (June 2026)** — added the halogen-substituted dicopper study (ref. 18); 847 structures.

---

## License & citation

- **Code** (`MAAC_run.py`): MIT License.
- **Data** (`MAAC_database.zip` and all contents): Creative Commons Attribution 4.0 (CC BY 4.0).

If you use this dataset or code, please cite the associated article (citation details to be completed upon publication).

## Contact

Ali A. Khairbek — alikhairbek@gmail.com
