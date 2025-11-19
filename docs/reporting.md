# Reporting

This page explains what you get in each output format

---

## 1. Text format (`-f text`)

- **Shows only problems by default**. Passed checks are hidden.
- Ends with a short summary (e.g. `file.nc has X issues`).

**Example snippet (terminal):**

<p align="center">
  <a href="https://raw.githubusercontent.com/ESGF/esgf-qc/Ipsl-Develop/docs/img/Capture1.png" target="_blank">
    <img src="https://raw.githubusercontent.com/ESGF/esgf-qc/Ipsl-Develop/docs/img/Capture1.png" width="900" alt="Text report example">
  </a>
</p>

## 2. JSON formats (`-f json`, `-f json_new`)

**Contains every check** (PASS or FAIL) with its details and adds two global counters:

  - `scored_points` = Σ(passed_assertions × weight)
  - `possible_points` = Σ(total_assertions × weight)
    - **Weights**: High = 3, Medium = 2, Low = 1.

**Severity lists:**
  
  - `high_priorities`, `medium_priorities`, `low_priorities` → only the severities active for the chosen criteria.
  - `all_priorities` → shows all priorities related to the chosen criteria.


**Criteria effect (-c)**:

  - `strict`: High + Medium + Low count in the decision.
  - `normal` (default): High + Medium count.
  - `lenient`: High only counts.

**Example snippet (terminal):**

<p align="center">
  <a href="https://raw.githubusercontent.com/ESGF/esgf-qc/Ipsl-Develop/docs/img/Capture2.png" target="_blank">
    <img src="https://raw.githubusercontent.com/ESGF/esgf-qc/Ipsl-Develop/docs/img/Capture2.png" width="900" alt="Text report example">
  </a>
</p>