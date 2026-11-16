"""Paper figures from outputs that already exist (no training).

    python scripts/make_figures.py --root $ROOT            # all figures that have inputs
    python scripts/make_figures.py --root $ROOT --only fig3 fig4

Inputs (all written by earlier steps):
  results/per_run.csv, runs/<variant>/seed*/preds_*.npz        (scripts/aggregate.py, run.py)
  results/comparisons_vs_full.csv                              (scripts/aggregate.py)
  results/analysis_full_seed42/gradient_cosines.json           (scripts/run_analysis.py)
  runs_ood/<variant>/seed42/best.pt, data/splits/par_test.jsonl (EXP-9 --save-best)
Outputs: results/figures/<name>.pdf (+ .png preview) and, for the IG example,
results/figures/fig7_attribution_example.json with the selection record.

Figures (numbers as in the manuscript):
  fig3_accuracy_consistency  VI macro-F1 vs CLCS, every run of the two-head systems
  fig4_agreement             EN-head x VI-head label on the 1,000 parallel test pairs
  fig5_ablation              Delta to the full model with 95% paired-bootstrap intervals
  fig6_gradients             gradient cosines between loss terms (one run, descriptive)
  fig7_attribution           Integrated Gradients on one parallel pair, two systems
  figA_confusion             confusion matrices of X-AlignFact (appendix)
  figA_tsne                  t-SNE of z for X-AlignFact, prototypes in the same fit (appendix)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from xalignfact.config import DEFAULT_SEEDS  # noqa: E402

# ---------------------------------------------------------------------------
# style: IEEE column = 3.5 in, double column = 7.16 in; 8 pt text
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.dpi": 300, "pdf.fonttype": 42, "text.color": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
COL1, COL2 = 3.5, 7.16
INK2 = "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
DIVERGE = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f0efec", "#eb6834"])
LAB = ["S", "R", "NEI"]

# display names = names used in the manuscript tables
NAMES = {"full": "X-AlignFact", "mt_xlmr": "MT-XLM-R", "mt_xlmr_vi": "MT-XLM-R",
         "translate_train": "Translate-train", "no_parallel": "w/o parallel data",
         "no_cons": "w/o $\\mathcal{L}_{cons}$", "no_repr": "w/o $\\mathcal{L}_{repr}$",
         "no_proto": "w/o $\\mathcal{L}_{proto}$", "cons_only": "$\\mathcal{L}_{cons}$ only",
         "translate_train_cons": "Translate-train + $\\mathcal{L}_{cons}$",
         "flat_curriculum": "Flat schedule", "fixed_ratio": "Fixed ratio",
         "shared_head": "Shared head", "plus_infonce": "+ InfoNCE",
         "plus_infonce_labelaware": "+ label-aware InfoNCE"}
ABLATION_ORDER = ["no_cons", "no_repr", "no_proto", "no_parallel", "translate_train", "cons_only",
                  "translate_train_cons", "flat_curriculum", "fixed_ratio", "shared_head",
                  "plus_infonce", "plus_infonce_labelaware"]


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}/{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {out}/{name}.pdf")


def runs_ok(root, variant, sub="runs"):
    """Seeds of a variant that are in DEFAULT_SEEDS and not diverged (per_run.csv rule)."""
    pr = f"{root}/results/per_run.csv"
    if os.path.exists(pr) and sub == "runs":
        df = pd.read_csv(pr)
        g = df[(df.variant == variant) & df.seed.isin(DEFAULT_SEEDS) & ~df.diverged.astype(bool)]
        return sorted(g.seed.tolist())
    return [s for s in DEFAULT_SEEDS if os.path.exists(f"{root}/{sub}/{variant}/seed{s}/results.json")]


def mt_name(root):
    """MT-XLM-R as reported in the manuscript: selected on Vietnamese dev F1 (mt_xlmr_vi)."""
    return "mt_xlmr_vi"


# ---------------------------------------------------------------------------
def fig3(root, out):
    df = pd.read_csv(f"{root}/results/per_run.csv")
    df = df[df.seed.isin(DEFAULT_SEEDS) & ~df.diverged.astype(bool)]
    systems = [mt_name(root), "no_parallel", "no_cons", "translate_train", "cons_only", "full"]
    markers = ["s", "v", "D", "^", "P", "o"]
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for k, (v, mk) in enumerate(zip(systems, markers)):
        g = df[df.variant == v]
        if g.empty or "par_clcs" not in g:
            continue
        c = SERIES[k]
        ax.scatter(g.vi_test_f1, g.par_clcs, s=12, marker=mk, color=c, alpha=0.45, linewidths=0)
        mx, my = g.vi_test_f1.mean(), g.par_clcs.mean()
        ax.errorbar(mx, my, xerr=g.vi_test_f1.std(ddof=1), yerr=g.par_clcs.std(ddof=1), fmt=mk,
                    ms=6, color=c, mec="#fcfcfb", mew=1.0, elinewidth=1.0, capsize=0,
                    label=f"{NAMES[v]} (n={len(g)})")
    ax.set_xlabel("Vietnamese macro-F1")
    ax.set_ylabel("CLCS (1,000 parallel test pairs)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    save(fig, out, "fig3_accuracy_consistency")


def _agreement(root, v, sub="runs", tag="par_test"):
    M = np.zeros((3, 3))
    seeds = runs_ok(root, v, sub)
    for s in seeds:
        d = np.load(f"{root}/{sub}/{v}/seed{s}/preds_{tag}.npz")
        for e, w in zip(d["en_preds"], d["vi_preds"]):
            M[int(e), int(w)] += 1
    return 100 * M / M.sum(), len(seeds)


def fig4(root, out):
    systems = [mt_name(root), "full"]
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.9))
    for ax, v in zip(axes, systems):
        M, n = _agreement(root, v)
        ax.imshow(M, cmap=BLUES, vmin=0, vmax=max(40, M.max()))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="#fcfcfb" if M[i, j] > 25 else "#0b0b0b",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3), LAB); ax.set_yticks(range(3), LAB)
        ax.set_xlabel("VI head (VI side)"); ax.grid(False)
        ax.set_title(f"{NAMES[v]}\nagreement {np.trace(M):.1f}%", fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("EN head (EN side)")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig4_agreement")


def fig5(root, out):
    cmp = pd.read_csv(f"{root}/results/comparisons_vs_full.csv")
    panels = [("vi_test", "macro_f1", "$\\Delta$ VI macro-F1 (points)"),
              ("par_test", "clcs", "$\\Delta$ CLCS (points)"),
              ("par_test", "joint_accuracy", "$\\Delta$ joint accuracy (points)")]
    order = [v for v in ABLATION_ORDER if v in set(cmp.variant)]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 0.22 * len(order) + 0.7), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (tag, metric, title) in zip(axes, panels):
        g = cmp[(cmp.set == tag) & (cmp.metric == metric)].set_index("variant")
        for yi, v in zip(y, order):
            if v not in g.index:
                continue
            r = g.loc[v]
            sig = r.p_holm < 0.05
            c = SERIES[0] if sig else "#9b9a95"
            ax.plot([100 * r.ci_low, 100 * r.ci_high], [yi, yi], color=c, lw=1.4, solid_capstyle="round")
            ax.plot(100 * r["diff"], yi, "o", ms=4.5, mfc=c if sig else "#fcfcfb", mec=c, mew=1.0)
        ax.axvline(0, color=INK2, lw=0.7)
        ax.set_xlabel(title); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [NAMES[v] for v in order])
    axes[0].text(0, 1.01, "filled: $p_{Holm}<0.05$; open: n.s.", transform=axes[0].transAxes,
                 fontsize=6.5, color=INK2, va="bottom")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig5_ablation")


def fig6(root, out, seed=42):
    p = f"{root}/results/analysis_full_seed{seed}/gradient_cosines.json"
    g = json.load(open(p))
    terms = ["L_EN", "L_VI", "L_cons", "L_repr", "L_proto"]
    tex = {"L_EN": "$\\mathcal{L}_{EN}$", "L_VI": "$\\mathcal{L}_{VI}$", "L_cons": "$\\mathcal{L}_{cons}$",
           "L_repr": "$\\mathcal{L}_{repr}$", "L_proto": "$\\mathcal{L}_{proto}$"}
    phases = list(g)
    fig, axes = plt.subplots(1, len(phases), figsize=(COL1, 1.8))
    axes = np.atleast_1d(axes)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for ax, ph in zip(axes, phases):
        present = [t for t in terms if any(t in k.split("|") for k in g[ph])]
        n = len(present)
        M = np.full((n, n), np.nan)
        for i, a in enumerate(present):
            M[i, i] = 1.0
            for j, b in enumerate(present):
                for key in (f"{a}|{b}", f"{b}|{a}"):
                    if key in g[ph]:
                        M[i, j] = g[ph][key]["mean"]
        Ml = np.where(np.tril(np.ones_like(M), -1) > 0, M, np.nan)[1:, :-1]   # strict lower triangle
        im = ax.imshow(Ml, cmap=DIVERGE, norm=norm)
        for i in range(n - 1):
            for j in range(i + 1):
                if not np.isnan(Ml[i, j]):
                    ax.text(j, i, f"{Ml[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="#fcfcfb" if abs(Ml[i, j]) > 0.6 else "#0b0b0b")
        ax.set_xticks(range(n - 1), [tex[t] for t in present[:-1]], rotation=45, ha="right")
        ax.set_yticks(range(n - 1), [tex[t] for t in present[1:]])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_title("End of " + ph.replace("phase", "Phase ").replace("_", " ").strip(), fontsize=7); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    cb.set_label("cosine", fontsize=6.5)
    save(fig, out, "fig6_gradients")


# ---------------------------------------------------------------------------
def _clean(t):
    return t.replace("▁", " ")


def _draw_tokens(ax, toks, attr, title, width=1.0, fs=6.3):
    """Tokens as text with a background shade = signed attribution (orange supports the
    predicted class, blue opposes it), normalised by the largest |value| in this row."""
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=6.8, pad=2)
    r = ax.figure.canvas.get_renderer()
    m = np.max(np.abs(attr)) + 1e-12
    x, yline, lh = 0.0, 0.80, None
    for t, a in zip(toks, attr):
        word = _clean(t)
        if word.startswith(" ") and x > 0:
            x += 0.006
        word = word.strip() or "·"
        w = a / m
        c = (np.array([0.922, 0.408, 0.204]) if w > 0 else np.array([0.165, 0.471, 0.839]))
        bg = 1 - abs(w) * (1 - c)                          # blend toward the pole colour
        txt = ax.text(x, yline, word, fontsize=fs, va="top", ha="left", transform=ax.transAxes,
                      bbox=dict(boxstyle="square,pad=0.08", fc=bg, ec="none"))
        bb = txt.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        lh = lh or 1.45 * bb.height
        if bb.x1 > width and x > 0:                       # wrap
            txt.remove()
            x, yline = 0.0, yline - lh
            txt = ax.text(x, yline, word, fontsize=fs, va="top", ha="left", transform=ax.transAxes,
                          bbox=dict(boxstyle="square,pad=0.08", fc=bg, ec="none"))
            bb = txt.get_window_extent(renderer=r).transformed(ax.transAxes.inverted())
        x = bb.x1 + 0.004


def fig7(root, out, seed=42, max_len=96):
    import torch

    from xalignfact.analysis import load_model
    from xalignfact.config import make_config
    from xalignfact.data import get_tokenizer, load_tensors
    from xalignfact.explain import SpecialIds, integrated_gradients, segments
    from xalignfact.model import EN, VI
    from xalignfact.utils import load_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    systems = ["full", "mt_xlmr_vi"]
    preds = {v: np.load(f"{root}/runs_ood/{v}/seed{seed}/preds_par_test.npz") for v in systems}
    cfg0 = make_config("full", seed=seed, root=root, runs_subdir="runs_ood")
    par = load_tensors(cfg0, ["par_test"])["par_test"]
    tok = get_tokenizer(cfg0)
    sp = SpecialIds.from_tokenizer(tok)
    y = preds["full"]["labels"]
    f, b = preds["full"], preds["mt_xlmr_vi"]
    # Selection rule (reported in the caption): the lowest-index pair on which X-AlignFact is
    # correct on both sides, MT-XLM-R disagrees across languages, the claim contains a digit,
    # and both sides fit in max_len tokens.
    chosen = None
    for i in range(len(y)):
        if not (f["en_preds"][i] == y[i] == f["vi_preds"][i] and b["en_preds"][i] != b["vi_preds"][i]):
            continue
        L = [int(par[f"{s}_attention_mask"][i].sum()) for s in ("en", "vi")]
        if max(L) > max_len:
            continue
        ids = par["en_input_ids"][i].long()
        seg = segments(ids, par["en_attention_mask"][i].long(), sp)
        claim = tok.convert_ids_to_tokens(ids[seg["claim"]].tolist())
        if any(ch.isdigit() for t in claim for ch in t):
            chosen = i
            break
    if chosen is None:
        print("  fig7: no pair meets the selection rule; relax --max-len")
        return
    vocab = load_json(f"{cfg0.data_dir}/manifest.json")["vocab_size"]
    rows, record = [], {"pair_index": int(chosen), "label": int(y[chosen]), "seed": seed,
                        "rule": "lowest index: X-AlignFact correct on both sides, MT-XLM-R "
                                "disagrees, digit in claim, <= %d tokens" % max_len}
    for v in systems:
        cfg = make_config(v, seed=seed, root=root, runs_subdir="runs_ood")
        model = load_model(cfg, f"{cfg.run_dir}/best.pt", vocab, device)
        for side, lang in (("en", EN), ("vi", VI)):
            ids = par[f"{side}_input_ids"][chosen].long()
            mask = par[f"{side}_attention_mask"][chosen].long()
            with torch.no_grad():
                yhat = int(model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), lang)["logits"].argmax(-1))
            ig = integrated_gradients(model, ids, mask, lang, yhat, sp, 32)
            seg = segments(ids, mask, sp)
            pos = seg["claim"] + [seg["first_sep"]] + seg["evidence"]
            toks = tok.convert_ids_to_tokens(ids[pos].tolist())
            att = np.array([ig[p] for p in pos])
            att[len(seg["claim"])] = 0.0                     # separator shown as a divider
            toks[len(seg["claim"])] = " ||"
            ev = np.abs(att[len(seg["claim"]) + 1:]).sum() / (np.abs(att).sum() + 1e-12)
            rows.append((v, side, yhat, toks, att, ev))
            record[f"{v}_{side}"] = {"pred": yhat, "evidence_share": float(ev),
                                     "tokens": toks, "ig": att.round(5).tolist()}
        del model
    fig, axes = plt.subplots(4, 1, figsize=(COL2, 4.6))
    for ax, (v, side, yhat, toks, att, ev) in zip(axes, rows):
        ok = "correct" if yhat == y[chosen] else "wrong"
        _draw_tokens(ax, toks, att, f"{NAMES[v]}, {side.upper()} head: {LAB[yhat]} ({ok}); "
                                    f"evidence share of |IG| {ev:.2f}")
    fig.text(0.0, 0.0, f"Gold label: {LAB[int(y[chosen])]}.  Shading: orange = supports the predicted "
             "label, blue = opposes it (per row, scaled to the largest |IG|).  || separates claim and evidence.",
             fontsize=6.3, color=INK2)
    save(fig, out, "fig7_attribution")
    json.dump(record, open(f"{out}/fig7_attribution_example.json", "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
def figA_confusion(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.9))
    for ax, tag, title in zip(axes, ("vi_test", "en_test"), ("Vietnamese test", "English test")):
        M = np.zeros((3, 3))
        seeds = runs_ok(root, "full")
        for s in seeds:
            d = np.load(f"{root}/runs/full/seed{s}/preds_{tag}.npz")
            for t, p in zip(d["labels"], d["preds"]):
                M[int(t), int(p)] += 1
        R = 100 * M / M.sum(1, keepdims=True)
        ax.imshow(R, cmap=BLUES, vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                        color="#fcfcfb" if R[i, j] > 55 else "#0b0b0b")
        ax.set_xticks(range(3), LAB); ax.set_yticks(range(3), LAB); ax.grid(False)
        ax.set_xlabel("Predicted"); ax.set_title(f"{title} ({len(seeds)} seeds)", fontsize=7)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
    axes[0].set_ylabel("Gold (row %)")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "figA_confusion")


def figA_tsne(root, out, seed=42):
    import torch

    from xalignfact.analysis import load_model, tsne_plot
    from xalignfact.config import make_config
    from xalignfact.data import load_tensors
    from xalignfact.utils import load_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for v in ("full",):          # MT-XLM-R has no prototype loss, so no trained prototypes
        cfg = make_config(v, seed=seed, root=root, runs_subdir="runs_ood")
        data = load_tensors(cfg, ["en_test", "vi_test"])
        vocab = load_json(f"{cfg.data_dir}/manifest.json")["vocab_size"]
        model = load_model(cfg, f"{cfg.run_dir}/best.pt", vocab, device)
        os.makedirs(out, exist_ok=True)
        tsne_plot(model, data["en_test"], data["vi_test"], f"{out}/figA_tsne_{v}.pdf", device)
        print(f"  wrote {out}/figA_tsne_{v}.pdf")
        del model


ALL = {"fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7,
       "figA_confusion": figA_confusion, "figA_tsne": figA_tsne}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", nargs="*", default=list(ALL))
    a = ap.parse_args()
    out = f"{a.root}/results/figures"
    for k in a.only:
        print(k)
        try:
            ALL[k](a.root, out)
        except FileNotFoundError as e:
            print(f"  skipped: missing input {e.filename}")
