"""The LaTeX bridge.

The write-up quotes numbers that the pipeline produces. Those two halves are
joined by a naming convention, and a naming convention with nothing checking it
is a naming convention that will drift. These tests are that check.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from gwglitch.latex import (
    code,
    escape,
    fmt,
    interval,
    macro_name,
    write_macros,
    write_table_body,
)

#: Keys that scripts/06_population_comparison.py emits into numbers.tex.
#: Kept here rather than imported so that a change to the script has to be a
#: deliberate change to this list too.
EMITTED_KEYS = [
    "strongest_feature",
    "strongest_delta",
    "strongest_channel",
    "strongest_rho",
    "quartile_ratio",
    "lag_peak",
    "classifier_auc",
    "classifier_efficiency",
    "coverage_o4a",
    "rate_ratio_h1",
    "rate_ratio_l1",
    "slow_shift_h1",
    "slow_shift_l1",
    "slow_fraction_o3_h1",
    "slow_fraction_o3_l1",
    "slow_fraction_o4a_h1",
    "slow_fraction_o4a_l1",
]


# ---------------------------------------------------------------------------
# Macro naming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("quartile_ratio", "ResultQuartileRatio"),
        ("rate_ratio_h1", "ResultRateRatioHOne"),
        ("slow_fraction_o3_l1", "ResultSlowFractionOThreeLOne"),
        ("slow_fraction_o4a_h1", "ResultSlowFractionOFouraHOne"),
        ("coverage_o4a", "ResultCoverageOFoura"),
    ],
)
def test_macro_name(key, expected):
    assert macro_name(key) == expected


def test_macro_names_are_legal_control_sequences():
    """TeX allows only letters in a control sequence name."""
    for key in EMITTED_KEYS:
        name = macro_name(key)
        assert re.fullmatch(r"[A-Za-z]+", name), f"{key} -> {name} is not legal"


def test_every_emitted_key_is_declared_in_main_tex(cfg):
    """A number the pipeline emits must have a placeholder to overwrite.

    ``numbers.tex`` uses ``\\renewcommand``, so an undeclared macro is a hard
    LaTeX error rather than a silently missing value -- but only if this stays
    in step, which is what the test enforces.
    """
    main = (cfg.root / "paper" / "main.tex").read_text(encoding="utf-8")
    declared = set(re.findall(r"\\newcommand\{\\(Result[A-Za-z]+)\}", main))

    missing = [k for k in EMITTED_KEYS if macro_name(k) not in declared]
    assert not missing, (
        "paper/main.tex is missing \\newcommand declarations for: "
        + ", ".join(f"{k} -> \\{macro_name(k)}" for k in missing)
    )


def test_declared_macros_are_all_used_or_emitted(cfg):
    """No orphan placeholders left behind in the preamble."""
    paper = cfg.root / "paper"
    main = (paper / "main.tex").read_text(encoding="utf-8")
    declared = set(re.findall(r"\\newcommand\{\\(Result[A-Za-z]+)\}", main))

    body = "".join(
        p.read_text(encoding="utf-8") for p in (paper / "sections").glob("*.tex")
    )
    used = set(re.findall(r"\\(Result[A-Za-z]+)", body))
    emitted = {macro_name(k) for k in EMITTED_KEYS}

    orphans = declared - used - emitted
    assert not orphans, f"declared but never used or emitted: {sorted(orphans)}"


def test_every_used_macro_is_declared(cfg):
    paper = cfg.root / "paper"
    main = (paper / "main.tex").read_text(encoding="utf-8")
    declared = set(re.findall(r"\\newcommand\{\\(Result[A-Za-z]+)\}", main))

    body = "".join(
        p.read_text(encoding="utf-8") for p in (paper / "sections").glob("*.tex")
    )
    used = set(re.findall(r"\\(Result[A-Za-z]+)", body))

    undeclared = used - declared
    assert not undeclared, f"used in a section but never declared: {sorted(undeclared)}"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_fmt_handles_missing_values():
    assert fmt(np.nan) == "---"
    assert fmt(None) == "---"
    assert fmt(np.inf) == "---"


def test_fmt_variants():
    assert fmt(0.1234, precision=2) == "0.12"
    assert fmt(0.25, percent=True, precision=2) == r"\SI{25}{\percent}"
    assert fmt(1234, thousands=True) == r"1\,234"
    assert fmt(1.05e-9, scientific=True, precision=2) == r"$1.05 \times 10^{-9}$"


def test_interval_is_bracketed():
    assert interval(0.1, 0.3, precision=2) == r"$[0.1,\, 0.3]$"


def test_escape_covers_the_characters_that_break_latex():
    assert escape("L1:SUS-OM3_M1_DAMP_L_IN1_DQ") == r"L1:SUS-OM3\_M1\_DAMP\_L\_IN1\_DQ"
    assert escape("50%") == r"50\%"
    assert escape("a&b") == r"a\&b"


def test_code_wraps_and_escapes():
    assert code("frac_energy_below_40hz") == r"\code{frac\_energy\_below\_40hz}"


# ---------------------------------------------------------------------------
# File emission
# ---------------------------------------------------------------------------


def test_write_macros_round_trip(tmp_path):
    path = write_macros(tmp_path / "numbers.tex", {"quartile_ratio": "3.4"})
    text = path.read_text(encoding="utf-8")
    assert r"\renewcommand{\ResultQuartileRatio}{3.4}" in text
    assert "do not edit by hand" in text


def _spec(tmp_path, name, colspec, header):
    """Create the placeholder .spec that write_table_body reads."""
    d = tmp_path / "placeholders"
    d.mkdir(exist_ok=True)
    (d / f"{name}.spec").write_text(f"{colspec}\n@@@\n{header}\n", encoding="utf-8")
    (tmp_path / "generated").mkdir(exist_ok=True)
    return tmp_path / "generated" / f"{name}.tex"


def test_write_table_body_emits_a_complete_tabular(tmp_path):
    df = pd.DataFrame({"detector": ["H1", "L1"], "rate": [1.234, 5.678]})
    out = _spec(tmp_path, "tab", "lr", r"Detector & Rate \\")
    path = write_table_body(
        out, df, ["detector", "rate"],
        {"detector": str, "rate": lambda v: fmt(v, precision=2)},
    )
    text = path.read_text(encoding="utf-8")

    # The whole environment, because \input cannot be used inside an alignment.
    for token in (r"\begin{tabular}{lr}", r"\toprule", r"\midrule",
                  r"\bottomrule", r"\end{tabular}"):
        assert token in text, token
    assert r"H1 & 1.2 \\" in text
    assert r"L1 & 5.7 \\" in text


def test_write_table_body_rejects_a_column_count_mismatch(tmp_path):
    """A spec and an emitter that disagree must fail here, not in LaTeX."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = _spec(tmp_path, "tab", "lrr", r"A & B & C \\")
    with pytest.raises(ValueError, match="columns"):
        write_table_body(out, df, ["a", "b"])


def test_write_table_body_rejects_missing_columns(tmp_path):
    df = pd.DataFrame({"a": [1]})
    out = _spec(tmp_path, "tab", "lr", r"A & B \\")
    with pytest.raises(KeyError, match="missing columns"):
        write_table_body(out, df, ["a", "b"])


def test_write_table_body_respects_max_rows(tmp_path):
    df = pd.DataFrame({"a": range(20)})
    out = _spec(tmp_path, "tab", "r", r"A \\")
    path = write_table_body(out, df, ["a"], max_rows=5)
    rows = [l for l in path.read_text(encoding="utf-8").splitlines()
            if l.endswith(r"\\") and "rule" not in l and "tabular" not in l]
    assert len(rows) == 6           # five data rows plus the header


@pytest.mark.parametrize(
    "value,precision,expected",
    [
        (1.0, 2, "100"),            # "%.2g" would give "1e+02"
        (1.0, 3, "100"),
        (0.5, 2, "50"),
        (0.0083, 2, "0.83"),
        (0.0, 2, "0"),
        (0.12345, 3, "12.3"),
    ],
)
def test_percent_never_uses_exponent_notation(value, precision, expected):
    """Per cent values must read as per cent, not as "1e+02 per cent"."""
    assert fmt(value, precision=precision, percent=True) == (
        r"\SI{" + expected + r"}{\percent}"
    )


def test_read_spec_round_trip(tmp_path):
    from gwglitch.latex import read_spec

    p = tmp_path / "t.spec"
    p.write_text("llrr\n@@@\nA & B & C & D \\\\\n", encoding="utf-8")
    colspec, header = read_spec(p)
    assert colspec == "llrr"
    assert header.startswith("A & B")


def test_every_generated_table_has_a_placeholder_and_spec(cfg):
    """Each \\generated{name} needs both fallback files to exist.

    The placeholder is what the document falls back to before the pipeline has
    run; the spec is what keeps the generated table the same shape as it.
    """
    import re as _re

    sections = (cfg.root / "paper" / "sections").glob("*.tex")
    names = set()
    for path in sections:
        names |= set(_re.findall(r"\\generated\{([a-z_]+)\}",
                                 path.read_text(encoding="utf-8")))
    assert names, "no generated tables found at all"

    ph = cfg.root / "paper" / "placeholders"
    for name in sorted(names):
        assert (ph / f"{name}.tex").exists(), f"missing placeholder for {name}"
        assert (ph / f"{name}.spec").exists(), f"missing spec for {name}"


def test_placeholder_and_spec_agree_on_column_count(cfg):
    from gwglitch.latex import _count_columns, read_spec

    ph = cfg.root / "paper" / "placeholders"
    for spec_path in sorted(ph.glob("*.spec")):
        colspec, header = read_spec(spec_path)
        n_spec = _count_columns(colspec)
        # The header row has one fewer "&" than it has columns.
        n_header = header.count("&") + 1
        assert n_spec == n_header, (
            f"{spec_path.name}: spec has {n_spec} columns, header has {n_header}"
        )


# ---------------------------------------------------------------------------
# The section files must stay compilable
# ---------------------------------------------------------------------------


def test_generated_table_fallbacks_are_balanced(cfg):
    """Each \\generated{name}{fallback} must have matching braces.

    An unbalanced fallback block breaks the document only when the generated
    file is *absent*, which is exactly the state a reader compiling a fresh
    checkout is in.
    """
    for path in (cfg.root / "paper" / "sections").glob("*.tex"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\generated\{([a-z_]+)\}\{", text):
            depth = 1
            i = match.end()
            while i < len(text) and depth:
                char = text[i]
                if char == "\\":
                    # A backslash escapes whatever follows, including the row
                    # terminator "\\" -- whose second backslash must not be
                    # mistaken for one escaping a closing brace.
                    i += 2
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                i += 1
            assert depth == 0, (
                f"{path.name}: unbalanced fallback for \\generated{{{match.group(1)}}}"
            )


def test_no_control_characters_in_the_latex_sources(cfg):
    """A stray control byte breaks the build with an opaque Unicode error.

    Shell here-documents interpret backslash escapes, so text written that way
    can turn ``\approx`` into a bell character and ``\begin`` into a
    backspace. The compile then fails with "Unicode character ^^G", which does
    not point anywhere useful.
    """
    offenders = {}
    for path in sorted((cfg.root / "paper").rglob("*.tex")):
        text = path.read_text(encoding="utf-8")
        bad = sorted({hex(ord(c)) for c in text if ord(c) < 32 and c not in "\n\t"})
        if bad:
            offenders[path.name] = bad
    assert not offenders, f"control characters in LaTeX sources: {offenders}"


def test_no_control_characters_in_the_python_sources(cfg):
    offenders = {}
    for folder in ("src", "scripts", "tests"):
        for path in sorted((cfg.root / folder).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            bad = sorted({hex(ord(c)) for c in text
                          if ord(c) < 32 and c not in "\n\t"})
            if bad:
                offenders[str(path.relative_to(cfg.root))] = bad
    assert not offenders, f"control characters in Python sources: {offenders}"


# ---------------------------------------------------------------------------
# Draft flagging
# ---------------------------------------------------------------------------
#
# The results section carries a status notice, but it sits twenty pages in. A
# reader who opens the PDF and skims the title page would see nothing --- which
# is how a draft ends up circulated as though it were finished. Four markers
# (title banner, every-page footer, PDF title, PDF subject) hang off the same
# condition as the status notice, so they cannot drift apart from it.


def test_draft_condition_is_defined_once_and_used_everywhere(cfg):
    paper = cfg.root / "paper"
    main = (paper / "main.tex").read_text(encoding="utf-8")
    title = (paper / "sections" / "00_title.tex").read_text(encoding="utf-8")
    results = (paper / "sections" / "05_results.tex").read_text(encoding="utf-8")

    # One switch, set from the presence of the pipeline's output.
    assert r"\newif\ifresultspending" in main
    assert r"\IfFileExists{numbers.tex}{\resultspendingfalse}{\resultspendingtrue}" in main

    # The title page banner and the results status notice must both hang off
    # the presence of numbers.tex, not off independent hand-set flags.
    assert r"\ifresultspending" in title, "title page has no draft banner"
    assert r"\IfFileExists{numbers.tex}" in results, (
        "results status notice is not conditional on the pipeline output"
    )


def test_draft_markers_cover_the_obvious_reading_paths(cfg):
    """Title page, every page, and the file-properties dialog."""
    main = (cfg.root / "paper" / "main.tex").read_text(encoding="utf-8")
    title = (cfg.root / "paper" / "sections" / "00_title.tex").read_text(
        encoding="utf-8"
    )

    assert "DRAFT" in title, "nothing on the title page"
    assert r"\fancyfoot" in main and "DRAFT" in main, "no per-page footer"
    assert "pdftitle" in main and "DRAFT" in main, "PDF metadata not marked"
    assert "NOT FOR CIRCULATION" in title


def test_draft_banner_states_the_trial_caveat(cfg):
    """The 47-hour auxiliary numbers appear outside the results section.

    They are quoted in the validation section and the README, so the banner has
    to say they are a trial --- otherwise removing the results placeholders
    would still leave an unqualified correlation coefficient in circulation.
    """
    title = (cfg.root / "paper" / "sections" / "00_title.tex").read_text(
        encoding="utf-8"
    )
    assert "47" in title and "trial" in title.lower()
