# REPO CLEANUP + PUBLISH PREP

Actions taken to prepare the repository for publication. Standing rules honored:
numbers quoted from frozen files (sources linked); nothing under `results/` or `data/`
was modified in content (see the one transient exception under History purge).

## 0. Security finding (ACTION REQUIRED BY MAINTAINER)

`origin` had a GitHub personal-access token embedded in its URL
(`https://ghp_…@github.com/IsItHM/DisasterShield.git`). It surfaced during the read-only
audit (`git remote -v`) and lived in `.git/config` (never tracked, so never committed).

- `git filter-repo` removed the `origin` remote as a side effect, scrubbing the token
  from `.git/config`.
- **The maintainer must still revoke that token** at GitHub → Settings → Developer
  settings, since it was exposed in tooling logs. Re-add a tokenless remote before pushing.

## 1. Audit (read-only)

- Branch `main`; 48 tracked files; existing `.gitignore` (standard Python template +
  `data/Raw_data/`, `*.tif`, `*.aux.xml`).
- Untracked working tree carried large Phase-2 assets: `.venv-dsx/` 2.2 GB,
  `data/processed/` 2.8 GB (`.npy`), `data/labels_unosat/` 622 MB, `results/` 266 MB
  (mostly `.keras`). `data/Feni_2024_10m/` (1.8 GB) was already covered by `*.tif`.
- Tracked history already carried ~235 MB of deprecated Phase-1 binaries.

## 2. History purge (git-filter-repo 2.47.0)

Chosen approach: purge the deprecated bloat from **all** history.

- **Safety net first:** `git bundle --all` → `pre-purge-backup.bundle` (185 MB, verified
  `okay`) in the session scratchpad; the pre-purge files were copied aside (234 MB); and
  `origin` still held the pre-rewrite history until any force-push.
- **Purged paths** (`--invert-paths`): `outputs/`, `data reports/`, `data/Raw_data/`, and
  the root-level `2016.pdf` / `2017.pdf` / `2018.pdf` — early-history copies of the same
  PDFs (blob `588534be…`) that a dry-run revealed would otherwise survive under an old
  path.
- A `--dry-run` was inspected before executing: the filtered history's surviving file set
  was exactly `.gitignore, LICENSE, README.md, data/Data_script_19_23/…,
  models/flood_model_2019_2023.ipynb, papers/knn_flood.pdf`.
- **Result:** `.git` shrank from ~235 MB of bloat to **1.5 MB**; 0 bloat paths remain
  tracked; history rewritten (HEAD hash changed); `origin` removed.
- **Disk copies restored:** `outputs/`, `data reports/`, `data/Raw_data/` were re-copied
  from the backup after filter-repo's `reset --hard` (net-unchanged on disk, now untracked
  and git-ignored). File counts verified equal to backup (10 / 12 / 20).

## 3. .gitignore additions

Appended a Phase-2 block: `.venv-dsx/`, `data/Feni_2024_10m/`, `data/labels_unosat/`,
`data/processed/*.npy`, `outputs/`, `data reports/`, `results/**/demo_build/*.png`, and
`results/**/*.keras` with a negation `!results/20260703T054304Z/feni_unet_best.keras` so
the single release checkpoint is kept. (`*.tif`, `*.log`, `__pycache__/` were already
ignored.)

> A first draft used **inline** comments after patterns; git only honors `#` at the start
> of a line, so those patterns broke and a dry-run staged 3+ GB. Caught before commit and
> fixed by moving comments to their own lines. Verified with `git check-ignore` and a
> `git add -A -n` dry-run: 69 keep-set files, largest 23.7 MB (final `.keras`), none > 50 MB.

## 4. README, requirements, preview (STEP 3)

- `README.md` rewritten: honest, concrete, no marketing language; all figures quoted from
  frozen files with source links/comments. Canonical result **U-Net test IoU 0.7216 /
  F1 0.8383** (`results/20260703T165223Z/threshold_fair_v3.csv`), a "What this is and
  isn't" section, repo map, reproduction steps, roadmap.
- `requirements.txt` generated from `.venv-dsx` (`pip freeze`), trimmed to actually-imported
  packages + geopandas/folium runtime engines.
- `docs/demo_preview.png` composed from the frozen demo overlays
  (`results/20260703T202117Z/demo_build/overlay_s2.png` + `overlay_agreement.png`).

## 5. Commit + push

- Commit (this cleanup): message
  `Phase 2: independent-label SAR flood pipeline, fair benchmarks, demo (canonical: v2 U-Net 0.7216 test IoU)`.
- **Not pushed.** History was rewritten, so publishing requires a **force-push** on a
  re-added tokenless remote — left to the maintainer (see §0). Recovery is possible from
  `pre-purge-backup.bundle` or the un-rewritten `origin` until that force-push.
