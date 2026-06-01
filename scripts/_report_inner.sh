#!/usr/bin/env bash
set -euo pipefail

python analyze_muon_atlas.py --runs_dir runs --report_dir report
python generate_report_assets.py --runs_dir runs --report_dir report

cd report
mkdir -p compiled
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  cp main.pdf compiled/muon_routing_atlas_report.pdf
elif command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
  cp main.pdf compiled/muon_routing_atlas_report.pdf
else
  echo "PDF compilation skipped because latexmk/tectonic was not found."
  cat > compiled/PDF_NOT_BUILT.txt <<'TXT'
PDF compilation skipped because latexmk/tectonic was not found.

Install one of:
  sudo apt-get install latexmk texlive-latex-extra texlive-fonts-recommended
  or
  https://tectonic-typesetting.github.io/
TXT
fi
