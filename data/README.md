# Data layout *(placeholder)*

**Do not commit** identifiable patient data or institution-internal paths.

For sharing policy, cohorts, and what this repository does **not** distribute, read [**availability_statement.md**](availability_statement.md).

Suggested layout:

```
data/
├── raw/           # local only — never committed
├── processed/     # tiles, manifests — policy TBD
└── splits/        # train/val/test CSV or JSON manifests
```

**TODO:** Document approved naming, label formats, and how to obtain data under IRB / DUA.
