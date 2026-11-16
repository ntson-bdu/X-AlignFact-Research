# Licences for the X-AlignFact v2 release

| Item | Licence | Why |
|---|---|---|
| Source code (`xalignfact/`, `scripts/`, `tests/`, notebook) | MIT | Our own code; permissive, standard for research code |
| `data/parallel_en_vi.json` (FEVER claims/evidence + Vietnamese MT) | CC BY-SA 4.0 | Derived from FEVER, distributed under CC BY-SA (share-alike): adaptations must keep a share-alike licence; BY-SA 3.0 adaptations may use BY-SA 4.0 |
| `flagged_pairs_annotated.csv` | CC BY-SA 4.0 | Contains FEVER claims and their translations |
| Split manifests, hashes, per-run logs, predictions (labels/indices only) | CC BY 4.0 | Our own outputs; no third-party text |
| ViFactCheck text, Vietnamese error examples, `vifc_test_vi2en.csv` | **not redistributed** | News text owned by the original outlets; rebuild with `scripts/prepare_data.py` and `scripts/ood_consistency.py --translate` |

## Attribution to include in README / Zenodo description
- FEVER: Thorne et al., NAACL 2018 (data under CC BY-SA; Wikipedia-derived).
- NLI-style FEVER: Nie, Chen and Bansal, AAAI 2019 (`pietrolesci/nli_fever`).
- ViFactCheck: Hoa et al., AAAI 2025 (`tranthaihoa/vifactcheck`).
- Machine translation: Google Translate via `deep-translator` (EN→VI parallel corpus);
  VinAI Translate `vinai-translate-vi2en-v2` (VI→EN out-of-domain set; model under AGPL-3.0, used
  unmodified through `transformers`; no VinAI code is included in this repository).

## Zenodo
Archived at https://doi.org/10.5281/zenodo.23850412 (new version of the v1 record
10.5281/zenodo.19682099, whose content — 4,400 hard negatives, old splits — differs from v2 and
should not be used to reproduce the article). The record lists both licences: MIT for the code
archive and CC BY-SA 4.0 for the data archive.
