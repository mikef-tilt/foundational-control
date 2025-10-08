#!/bin/bash
python summary_plots.py --input cache/first_loans.parquet --output weekly_loans.png
python lko.py --input cache/first_loans.parquet --histogram_output weekly_growth_histogram.png
