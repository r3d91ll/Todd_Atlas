# Experiments

This directory contains both archived and active experiments for the Atlas project.

## Structure

```text
experiments/
├── archived/           # Completed or discontinued experiments
│   ├── 001_*/          # Numbered for chronological ordering
│   ├── 002_*/
│   └── ...
└── active/             # Currently running experiments
    └── */
```

## Archived Experiments

Each archived experiment contains:
- `README.md` - Hypothesis, approach, results, and lessons learned
- `code/` - Scripts and configurations used
- `results/` - Outputs, metrics, and artifacts (if preserved)

Archived experiments are valuable documentation of the research journey, including "failed" attempts that informed subsequent design decisions.

## Active Experiments

Experiments currently in progress. Once complete (successful or not), they should be moved to `archived/` with proper documentation.

## Naming Convention

- Archived: `NNN_descriptive_name/` (e.g., `001_titans_original/`)
- Active: `descriptive_name/` (e.g., `shakespeare_kakeya/`)
