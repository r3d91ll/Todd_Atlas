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
    ├── shakespeare_kakeya/
    └── dumas_kakeya/
```

## Active Experiments

### shakespeare_kakeya
- 10M Atlas trained on Shakespeare (English)
- Grokking detection + Kakeya geometry study
- See: `active/shakespeare_kakeya/README.md`

### dumas_kakeya
- 10M Atlas trained on Dumas (French)
- Controlled comparison with Shakespeare experiment
- See: `active/dumas_kakeya/README.md`

## Archived Experiments

Each archived experiment contains:
- `README.md` - Hypothesis, approach, results, and lessons learned
- `code/` - Scripts and configurations used
- `results/` - Outputs, metrics, and artifacts (if preserved)

Archived experiments are valuable documentation of the research journey, including "failed" attempts that informed subsequent design decisions.

## Naming Convention

- Archived: `NNN_descriptive_name/` (e.g., `001_titans_original/`)
- Active: `descriptive_name/` (e.g., `shakespeare_kakeya/`)
