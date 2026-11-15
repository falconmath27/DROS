# Integration changes (15 November 2026)

Assembly and verification of the final report and slides on top of `d18771b`
("P5 task done!"). No model code, hyperparameters, seeds, experiment configs or files
under `results/` were edited by hand. Nothing is committed.

## Environment (not repository files)

- Installed Tectonic 0.17.0 to `C:\Users\anama\tools\tectonic` to compile the PDFs.
- Installed the new `requirements.txt` packages (yfinance, requests, lxml, matplotlib, jinja2).
- Set `core.autocrlf=false` in this clone's `.git/config` and re-checked out the files, so
  LF files are byte-identical to the commit (see open issue 1).

## Regeneration check

All table scripts were re-run. `experiments.summarize`, `experiments.paired`,
`backtest.summarize` and `experiments.benchmarks` reproduce their committed outputs exactly.
`experiments.backtest.summarize --n-boot 2000` reproduces every table the report uses
(`primary.tex`, `costs.tex`, `regimes.tex`, `markets.tex`, `interpretation.tex`); the other
outputs differ only by about 1e-17 in two interval columns and by machine-specific paths in
`manifest.json`, so the committed versions were kept.

## Files

### New
- `experiments/report_tables.py`: writes the P3 fBm and E4 tables and the P5 hold comparison
  (`results/report/*.tex`) from the CSVs written by the P3 and P5 scripts, so no table in the
  report is typed by hand.
- `results/report/fbm_table.tex`, `e4_table.tex`, `p5_means.tex`, `p5_vs_hold.tex`: its output.
- `CHANGES.md`: this file.

### Scripts
- `experiments/summarize.py`: adds a note under the E4 and E5synthetic headings of
  `summary.md` (earlier exact-Cholesky simulator; synthetic data); writes LF line endings.
- `results/summary.md`: regenerated; only the two notes are new.

### Report (`report/`)
- `main.tex`: author block with roles (Mridul Kwatra P3, Vraj Shah P4), course, BITS Pilani K K Birla Goa Campus,
  15 November 2026; new abstract; "Experiments and results" label and lead-in; two references
  added (Gatheral-Jaisson-Rosenbaum 2018, Bennedsen-Lunde-Pakkanen 2017).
- `intro.tex`: cites Gatheral et al. for empirical roughness (was Bayer-Friz-Gatheral, the
  pricing paper); adds a section roadmap.
- `theory.tex`: reward written $R_j$ (was $g_j$) to match the code, P3 and P4 sections; payoff
  $g(S)$ and the conventions ($R$ discounted to date 0, $H$ horizon, $h$ Hurst) stated once;
  section label.
- `method.tex`: $g_j \to R_j$; section label; "Section 2" now a `\ref`.
- `results_replication.tex`: "Section 2" now a `\ref`; removed a stale comment about the old
  stand-in simulator.
- `rough_vol.tex`: the fBm and E4 tables now `\input` generated files (all 15 rows identical
  to the hand-typed ones); states the shortened training budgets actually used
  (`run_fbm.py --medium`, `run_e4.py`); replaces the outdated "71-test suite" with a
  description of the actual P3 test (it checks Var of the Volterra process, not $E[v_t]$);
  cites the hybrid scheme.
- `data_backtest.tex`: adds the walk-forward test periods and fold counts from `folds.json`;
  replaces a role code ("P5") in the prose with a section reference.
- `backtest_results.tex`: heading without "(P5)"; tables and figure placed in captioned floats
  that state the metric (costs and regimes tables are economic net return); new paragraph and
  two generated tables comparing every method with holding to the deadline; "Task 5 ...
  remain outstanding" restated as a limitation; removed "P4 methodology review is pending"
  (P4 review not required).
- `conclusion.tex`: rewritten "What we found" plus new "Does memory help?" and "Does learned
  exit timing pay?" paragraphs from the final results; limitations for the reduced P3 runs
  and the market study; removed placeholders.
- `slides.tex`: title slide with names, roles, campus and date; outline slide; takeaways slide.
- `slides_p1_p2.tex`: $g \to R$; 7 frames condensed to 4 (baselines and implementation fixes
  folded into other frames, tree/PDE table replaced by its one-line result).
- `slides_p3.tex`: 5 frames to 3; tables now `\input` the generated files.
- `slides_p4.tex`: 3 frames to 2; removed "71 tests passed"; fold dates added; role codes
  removed from frame titles.
- `slides_p5.tex`: new "Nothing beats holding" frame; persistence-plot frame dropped for
  length; "Task 5 diagnostics and P4 review pending" restated; primary table scaled to fit.
- `main.pdf` (14 pages) and `slides.pdf` (15 slides): rebuilt, no LaTeX warnings.

### README
- Team table filled with the roster given (names, roles, final status); stand-in and
  skeleton labels removed; 82 tests; E4/E5 descriptions match the code; old E4 and
  E5synthetic rows labelled; real-market E5 headline added; regeneration commands for the
  P5 summary and report tables; line-ending note.

### One cohesive report (second pass)
- `main.tex`: names appear once, in a Name / ID number table on the title page (IDs from the
  project proposal); role footnotes removed; "Project Report" added to the course line.
- `experiments/report_tables.py`: three more generated tables, `p5_primary.tex`,
  `p5_costs.tex`, `p5_regimes.tex` (readable headers, percentages), replacing P5's raw
  `primary.tex`, `costs.tex`, `regimes.tex` and the twelve generated sentences of
  `interpretation.tex` in the report.
- `backtest_results.tex`: rewritten in the report's voice with the same content; states the
  15/60-day block sensitivity exactly (7 of 120 base-cost intervals change, none involving
  holding or the primary contrast).
- `results_replication.tex`, `rough_vol.tex`, `data_backtest.tex`: removed repository file
  names, function names, "JSON overrides", "harness", "smoke run", "handoff" and similar
  internal wording.
- `main.pdf`: rebuilt, 13 pages, no LaTeX warnings.
- `slides.tex`: title slide shows the same Name / ID list as the report, without roles;
  `slides_p5.tex` uses the generated primary-contrast table. `slides.pdf` rebuilt.

### Repository cleanup
- Removed `results/p5/grid_smoke/` (37 MB) and `results/p5/one_fold/` (74 MB): smoke and
  one-fold integration outputs that no script, test or report reads; both are now in
  `.gitignore` and are regenerated by `experiments/backtest/full_grid.py --smoke` and
  `experiments/backtest/run.py`.
- Removed `report/P4_REVIEW_REQUEST.md` (P4 review not required) and its mentions in
  `experiments/backtest/README.md`, which now gives a generic Tectonic command instead of a
  teammate's local path and the `experiments.report_tables` step.
- Kept `results/p4/model_handoff_smoke/` and `results/p4/offline_replay/` (0.9 MB, cited as
  P4 validation evidence) and `results/archive_v1/` (first replication run, cited in README).

### Four-page report (replaces the 13-page version)
- `report/main.tex` is now a single file: 4 pages, 11pt, one column, including references.
  Sections: introduction; optimal stopping and OSPG (with the PDE ground truth in one
  sentence); replication (Table 1); fBm and rough Bergomi (compact side-by-side tables); the
  real-market backtest, given the most space (data and design, mean returns, contrasts with
  holding, cost sensitivity, volatility-persistence figure); conclusion with limitations.
- Removed the per-section files `intro.tex`, `theory.tex`, `method.tex`,
  `results_replication.tex`, `rough_vol.tex`, `data_backtest.tex`, `backtest_results.tex`,
  `conclusion.tex` (the long version stays in git history, commit `5c977a7`).
- `experiments/report_tables.py`: adds `fbm_compact.tex`, `e4_compact.tex` and
  `persistence_corr.tex` (Spearman correlation of volatility persistence with the GRU-minus-MLP
  return: +0.10 linear, 0.00 exponential).
- README team table points to report sections instead of the removed files.

# Open issues

1. **Hash checks break on Windows checkouts.** `experiments/backtest/summarize.py` compares
   SHA-256 hashes of `config.json` (frozen with LF endings) and the three P4
   `clean_prices.csv` files (frozen with CRLF endings). With `core.autocrlf=true` (the
   Windows default) the summary refuses to run. I ran it after restoring LF files and
   temporarily converting the three CSVs to CRLF, then restored them. A `.gitattributes`
   (e.g. `* -text` for `results/` and the config) or hashing normalised text would fix this;
   that is a decision for P4/P5.
2. **European-call validation cannot be verified.** `rough_vol.tex` and the P3 slide quote
   $0.07886 \pm 0.00061$ against a "published reference value" $0.0791$. No script or results
   file in the repository produces the estimate, and the reference has no citation. Kept as
   written; P3 should add the script and the source.
3. **Old E4 numbers do not match the current code.** `results/summary.md` and the README's E4
   row come from the earlier exact-Cholesky simulator (100,000 paths, full training).
   `rbergomi_put_problem` now uses the hybrid scheme, so `run_all.py E4` would give different
   numbers. Labelled, not re-run (as agreed). The report uses P3's reduced sweep only.
4. **P3 runs are pilots.** The E4 sweep uses 3,000 paths, one split and training budgets
   about 16 times shorter than E1-E3 (OSPG 500 steps, DOS 150); the fBm table uses two splits
   and shortened budgets. The report now says so, but conclusions about memory under rough
   volatility are weak. `results/fbm_raw.jsonl` also mixes smoke rows (800 training paths)
   with the medium run used in the table.
5. **`experiments/rough_vol/run_fbm.py` runs everything twice.** It ends with two
   `if __name__ == "__main__": main()` blocks. Not changed (P3's experiment script).
6. **Hurst diagnostics not done.** `analysis/hurst.py` is still a stub; the report states this
   as a limitation.
7. **Section order.** The brief lists Intro, Method, Theory; the report keeps Intro, Theory,
   Method, because the Method section uses the Snell envelope, continuation values and exact
   prices defined in Theory and refers back to them. Swap if you prefer the brief's order.
8. **Citations to confirm.** Gatheral-Jaisson-Rosenbaum (Quantitative Finance 18(6), 2018)
    and Bennedsen-Lunde-Pakkanen (Finance and Stochastics 21(4), 2017) were added for claims
    the text already made; please confirm the hybrid scheme in `rbergomi.py` follows the
    latter. Bibliography entries give journal, volume and year but no page numbers.
9. **Repository size.** After the cleanup `results/` tracks about 607 MB, mostly the frozen
   P5 grid (531 MB) that the summary statistics are computed from. Deleted files remain in
   git history, so the clone itself does not shrink unless history is rewritten.
