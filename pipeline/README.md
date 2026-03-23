# End-to-end pipeline *(skeleton)*

**Planned flow**

1. Ingest tiles or WSI-derived chunks  
2. **Stage 1** — score / filter regions  
3. **Stage 2** — detect cells, aggregate blast fraction  
4. Slide-level report vs. thresholds (e.g. ≤5% / ≥20%)  

Add CLI: `python -m pipeline.run --config configs/default.yaml` *(placeholder)*.
