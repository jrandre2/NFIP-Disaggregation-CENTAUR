# Project-Local Skill Adapters

Thin wrappers that adapt reusable global academic-manuscript skills to this
repository (the NFIP claim-disaggregation paper). Each adapter loads its global
base skill first, then applies only project-specific files, terminology, counts,
and high-risk failure modes. Keep shared rules in the base skills; keep project
facts here.

| Adapter | Base skill | Use for |
|---|---|---|
| `nfip-manuscript-audit` | `academic-methods-manuscript-audit` (in-session: `manuscript-audit`) | Fact-checking numbers/tables/figures, internal consistency, claim support, verification against `data_work/` |
| `nfip-manuscript-writing` | `academic-manuscript-writing-editing` | Copyediting and writing prose/captions/abstract in standalone academic voice with project terminology |

Source of truth for the manuscript: `manuscript_quarto/index.qmd`.
Render with `cd manuscript_quarto && ./render_all.sh`.
