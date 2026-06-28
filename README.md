# README: AI Student Impact Analysis Tool

## Overview

**"Is AI a Tutor or a Cheat Code?"** This project provides a professional-grade, robust, and interactive pipeline to clean, analyze, and visualize a dataset of 50,000 student records regarding Generative AI usage, academic performance (GPA), and burnout risks. This tool is designed to move beyond raw data, helping researchers and educators identify whether AI integration correlates with improved learning outcomes or hinders skill retention.

---

## Features

* **Automated Data Cleaning:** * Handles missing values via median/mode imputation.
* Standardizes inconsistent categorical text.
* Enforces logical bounds (e.g., GPA range, weekly hours).
* Flags statistical outliers for transparency without deleting valuable extreme-use cases.


* **Deep Statistical Analysis:** * Computes Pearson correlation, ANOVA, t-tests, and regression.
* Quantifies the relationship between AI usage, exam anxiety, and GPA changes.


* **Visualizations:** Automatically generates and saves high-quality PNG charts (Bar, Pie, Line, and Distribution plots).
* **Interactive Interface:** A user-friendly, menu-driven terminal tool for real-time data exploration.
* **Robustness:** Engineered with error handling for every major step; the script degrades gracefully if optional libraries are missing.

---

## Requirements

The script is built to be efficient. While it runs on base Python, the following libraries are recommended for full functionality:

* **Core:** `pandas`, `numpy`
* **Optional (Recommended):** `matplotlib`, `seaborn`, `scipy`

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scipy

```

---

## How to Run

1. **Clone or Download** this repository.
2. **Place your CSV** file in the directory (default: `ai_student_impact_dataset.csv`).
3. **Execute the script:**
```bash
python ai_student_impact_analysis.py

```



### Command-Line Arguments

* `--file [path]`: Specify a custom path to your dataset.
* `--no-interactive`: Run in background/auto-pilot mode.
* `--no-show`: Prevent charts from popping up (useful for headless servers).

---

## Methodology & Visuals

The analysis identifies trends by categorizing students and comparing outcomes across groups. For instance, visualizing the relationship between weekly AI hours and GPA allows us to differentiate between "Tutor-like" support and "Cheat Code" reliance.

> **Note:** The script automatically creates an `ai_student_impact_charts/` directory to store all generated visual reports for your review.

---

## License

This tool is open-source and intended for educational research. Please cite the dataset sources if used in professional or academic publications.
