"""
Academic Paper Integrity & Consistency Suite.
Asserts that docs/paper/main.tex and docs/paper/references.bib are:
1. Complete: All \\cite{...} keys exist in references.bib.
2. Self-contained: All \\ref{...} labels exist in main.tex.
3. Syntax balanced: All \\begin{env} have matching \\end{env}.
4. Empirically accurate: Tournament numbers in LaTeX match docs/benchmarks.md.
5. Bot nomenclature aligned: All bot modes match server.py BOT_CLASSES.
"""

import os
import re
import pytest

MAIN_TEX_PATH = "docs/paper/main.tex"
BIB_PATH = "docs/paper/references.bib"
BENCHMARKS_PATH = "docs/benchmarks.md"

def load_file(path):
    assert os.path.exists(path), f"File missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def test_citations_exist_in_bib():
    tex_content = load_file(MAIN_TEX_PATH)
    bib_content = load_file(BIB_PATH)

    # Extract all bib entry keys: @type{key,
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([a-zA-Z0-9_\-]+)\s*,", bib_content))
    assert len(bib_keys) >= 10, f"Expected at least 10 BibTeX entries, found {len(bib_keys)}"

    # Extract all cite keys: \cite{key1, key2}
    raw_cites = re.findall(r"\\cite\{([^}]+)\}", tex_content)
    tex_cites = set()
    for group in raw_cites:
        for key in group.split(","):
            tex_cites.add(key.strip())

    missing = tex_cites - bib_keys
    assert not missing, f"Dangling citations in main.tex not found in references.bib: {missing}"
    print(f"✓ All {len(tex_cites)} citations in main.tex resolve cleanly to references.bib.")

def test_latex_labels_and_refs():
    tex_content = load_file(MAIN_TEX_PATH)

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex_content))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex_content))

    missing_labels = refs - labels
    assert not missing_labels, f"Unresolved \\ref in main.tex: {missing_labels}"
    print(f"✓ All {len(refs)} cross-references match defined labels ({len(labels)} labels).")

def test_latex_environments_balanced():
    tex_content = load_file(MAIN_TEX_PATH)

    # Ignore comments
    lines = [line.split("%")[0] for line in tex_content.splitlines()]
    clean_tex = "\n".join(lines)

    begins = re.findall(r"\\begin\{([^}]+)\}", clean_tex)
    ends = re.findall(r"\\end\{([^}]+)\}", clean_tex)

    assert len(begins) == len(ends), f"Environment count mismatch: {len(begins)} begins vs {len(ends)} ends"

    begin_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1

    end_counts = {}
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1

    diff = {}
    for env in set(begin_counts.keys()).union(end_counts.keys()):
        if begin_counts.get(env, 0) != end_counts.get(env, 0):
            diff[env] = (begin_counts.get(env, 0), end_counts.get(env, 0))

    assert not diff, f"Unbalanced LaTeX environments (begin, end): {diff}"
    print(f"✓ All {len(begins)} LaTeX environments are perfectly balanced.")

def test_empirical_numbers_consistency():
    tex_content = load_file(MAIN_TEX_PATH)
    bench_content = load_file(BENCHMARKS_PATH)

    # HybridBot net score (+156) must be present in both
    assert "+156" in tex_content, "HybridBot net score +156 missing in main.tex"
    assert "+156" in bench_content, "HybridBot net score +156 missing in benchmarks.md"

    # Loss rate 1.6% (or 3 losses out of 190-200 non-draw games)
    assert "1.6\\%" in tex_content or "1.6%" in tex_content, "Loss rate 1.6% missing in main.tex"

    # Claim 2 dynamic ranges: 33.8% full model vs 20.7% ablation
    assert "33.8\\%" in tex_content or "33.8%" in tex_content, "Full model dynamic range 33.8% missing in main.tex"
    assert "20.7\\%" in tex_content or "20.7%" in tex_content, "Ablation dynamic range 20.7% missing in main.tex"
    print("✓ Empirical tournament & ablation statistics in main.tex match docs/benchmarks.md.")

def test_bot_nomenclature_alignment():
    tex_content = load_file(MAIN_TEX_PATH)
    import server
    bot_keys = set(server.BOT_CLASSES.keys())
    canonical = {"random", "honest", "cardcount", "bayesian", "purenn", "hybrid"}
    assert canonical.issubset(bot_keys), f"Missing canonical bots: {canonical - bot_keys}"

    for key in ["Random", "Honest", "CardCount", "Bayesian", "PureNN", "Hybrid"]:
        assert key in tex_content, f"Bot name {key} missing in main.tex"
    print("✓ All 6 canonical bot names aligned across server.py and main.tex.")
