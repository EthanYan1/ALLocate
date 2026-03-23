# Data layout *(placeholder)*

**Do not commit** identifiable patient data or institution-internal paths.

Suggested layout:

```
data/
├── raw/           # local only — never committed
├── processed/     # tiles, manifests — policy TBD
└── splits/        # train/val/test CSV or JSON manifests
```

**TODO:** Document approved naming, label formats, and how to obtain data under IRB / DUA.
