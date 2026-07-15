"""Gates for the Nucleotide Transformer log-likelihood-ratio (LLR) connector.

WHY THIS FILE EXISTS -- 2026-07-15 (roadmap 6.27)
================================================
`genomiclm_llr` was **identically 0.0 for all 4,420,180 cohort rows, on every real
dataset, from the day the connector was written.** Nothing failed. Nothing printed.

The mechanism, proven on the owner's box 2026-07-15:

    is_fast: False | class: EsmTokenizer
    OFFSET RAISES -> NotImplementedError: return_offset_mapping is not available
    when using Python tokenizers.

`genomic_lm._masked_centre_logratio` located the variant's centre token with

    off = tok(win, return_offsets_mapping=True).get("offset_mapping")

Nucleotide Transformer's tokeniser is `EsmTokenizer`, a SLOW (pure-Python) tokeniser.
HuggingFace raises `NotImplementedError` for that argument on every slow tokeniser --
only `PreTrainedTokenizerFast` supports it. So it raised on EVERY window. A bare
`except Exception` swallowed it into `logger.debug(...)`, which is BELOW the default
level and printed nothing, and was additionally marked `# pragma: no cover` so the
coverage tool was told not to look either.

WHY IT SURVIVED SO LONG -- three independent blind spots, all now closed:

  1. `genomiclm_delta_norm` (the sibling feature) never touches offset mapping, so it
     stayed ALIVE and the connector looked healthy from the outside.
  2. `build_reference_slice` FEEDS `genomiclm_llr` a synthetic `rng.uniform(-12, 4)`.
     `engineer_features` reads it via a plain `df.get` passthrough, so the stage-5
     zero-audit graded the FIXTURE and never invoked the connector. Roadmap 7c: a gate
     that checks a PROXY instead of the thing it protects is not a gate.
  3. The harness comment asserted all six new columns were "live connectors -- Run-17
     real-data smoke shows them populated". The smoke audit it cited says
     `genomiclm_llr` = DEAD IN ALL SPLITS. The comment contradicted its own evidence.

THE LESSON, and what these tests bind:
`_assert_no_dead_features` would eventually have caught this -- by hard-failing Run 17
on the rented graphics-processing-unit box, after full data preparation, on paid
compute. That is a true gate firing in the most expensive possible place. These tests
move the detection to `pytest`, where it costs nothing.

These tests DO NOT require the network. They stub the tokeniser, and they read the
module's own source with the `ast` module. The real forward pass is exercised
separately on the training box (huggingface.co is unreachable from continuous
integration) -- which is exactly why the source-level tripwires below exist.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from genomic_variant_classifier.data import genomic_lm
from genomic_variant_classifier.data.genomic_lm import centre_token_index

WINDOW = 101


# ---------------------------------------------------------------------------
# A stand-in for Nucleotide Transformer's EsmTokenizer.
#
# Only what centre_token_index actually touches is modelled: convert_ids_to_tokens
# and all_special_tokens. Deliberately NOT a Mock -- a Mock would answer every
# question and prove nothing.
# ---------------------------------------------------------------------------
class _StubTok:
    all_special_tokens = ["<cls>", "<mask>", "<pad>", "<unk>"]
    is_fast = False

    def __init__(self, toks: list[str]) -> None:
        self._toks = toks

    def convert_ids_to_tokens(self, ids):  # noqa: ARG002 - ids unused by the stub
        return self._toks


def _nt_style_tokens(win: str, k: int = 6) -> list[str]:
    """<cls> + non-overlapping k-mers + leftover bases as single-character tokens.

    This mirrors Nucleotide Transformer's documented scheme. centre_token_index does
    NOT depend on it being right -- that is the whole point of reconstructing offsets
    from the emitted token strings -- but the tests need a realistic shape to assert
    against.
    """
    n_full = len(win) // k
    toks = ["<cls>"] + [win[i * k:(i + 1) * k] for i in range(n_full)]
    toks.extend(win[n_full * k:])
    return toks


def _ids(toks: list[str]) -> list[int]:
    return list(range(len(toks)))


# ---------------------------------------------------------------------------
# centre_token_index -- the replacement for the offset-mapping call
# ---------------------------------------------------------------------------
def test_centre_token_index_finds_the_token_spanning_the_variant_base():
    """The centre base of a 101 bp window is index 50 (WINDOW // 2), which is where
    delta_window_builder places the variant's first base."""
    win = "ACGT" * 25 + "A"
    assert len(win) == WINDOW

    toks = _nt_style_tokens(win)
    idx = centre_token_index(_StubTok(toks), win, _ids(toks))

    # 101 = 16 six-mers (96 bp) + 5 leftover bases; base 50 lands in six-mer 50 // 6 = 8;
    # one <cls> is prepended -> token index 9.
    assert idx == 9
    # Assert the RELATIONSHIP, not just the number: the token must really cover base 50.
    assert toks[idx] == win[48:54]
    assert 48 <= 50 < 54


def test_centre_token_index_is_independent_of_kmer_size():
    """The function reads the tokens it is given. It must not assume k == 6."""
    win = "ACGT" * 25 + "A"
    for k in (1, 2, 3, 4, 5, 6, 7, 8):
        toks = _nt_style_tokens(win, k=k)
        idx = centre_token_index(_StubTok(toks), win, _ids(toks))
        assert idx is not None, f"no centre token found for k={k}"
        # Reconstruct this token's span and assert it contains base 50.
        start = sum(len(t) for t in toks[1:idx])
        assert start <= (WINDOW // 2) < start + len(toks[idx]), f"k={k}"


def test_centre_token_index_skips_special_tokens_without_consuming_characters():
    """Special tokens must not advance the character cursor, or every index shifts."""
    win = "ACGT" * 25 + "A"
    base = _nt_style_tokens(win)
    idx_plain = centre_token_index(_StubTok(base), win, _ids(base))

    padded = ["<pad>", "<pad>"] + base + ["<pad>"]
    idx_padded = centre_token_index(_StubTok(padded), win, _ids(padded))

    assert idx_padded == idx_plain + 2
    assert padded[idx_padded] == base[idx_plain]


def test_centre_token_index_raises_when_tokens_do_not_reconstruct_the_window():
    """The load-bearing assertion.

    If the tokens do not span the window exactly, the centre index is wrong and every
    LLR scores the WRONG BASE -- a plausible, wrong number for every variant in the
    cohort. That is worse than a column of zeros, because zeros are visibly dead and a
    wrong base is not. It must raise.
    """
    win = "ACGT" * 25 + "A"
    truncated = ["<cls>"] + [win[i * 6:(i + 1) * 6] for i in range(16)]  # 5 bases dropped

    with pytest.raises(RuntimeError, match="round-trip mismatch"):
        centre_token_index(_StubTok(truncated), win, _ids(truncated))


def test_centre_token_index_raises_when_tokens_overrun_the_window():
    """Mismatch in the other direction must also raise -- negative-tested, per
    CLAUDE.md 6.4: 'A guard is not real until you have watched it FAIL.'"""
    win = "ACGT" * 25 + "A"
    overrun = _nt_style_tokens(win) + ["ACGTAC"]

    with pytest.raises(RuntimeError, match="round-trip mismatch"):
        centre_token_index(_StubTok(overrun), win, _ids(overrun))


def test_centre_token_index_returns_none_only_for_an_empty_window():
    """The single reachable None: no bases, so no token covers the centre. The caller
    counts these and raises if EVERY pair is unlocatable."""
    assert centre_token_index(_StubTok(["<cls>"]), "", [0]) is None


# ---------------------------------------------------------------------------
# SOURCE-LEVEL TRIPWIRES
#
# These are the tests that would actually have caught the original bug, and they are
# the ones that stop it returning. They read the module's own source, because the real
# forward pass cannot run in continuous integration (no network) -- which is precisely
# the gap the bug lived in for its entire life.
# ---------------------------------------------------------------------------
def _module_ast() -> ast.Module:
    return ast.parse(Path(genomic_lm.__file__).read_text(encoding="utf-8"))


def test_offset_mapping_is_never_requested_from_the_tokeniser():
    """`return_offsets_mapping=True` raises NotImplementedError on the slow
    EsmTokenizer. It must never be passed to a call in this module again.

    Uses the `ast` module, not a substring search: the docstrings in genomic_lm.py
    legitimately DISCUSS `return_offsets_mapping` while explaining this incident, and a
    text search would fire on the explanation. Only a real CALL is a defect.
    """
    offenders = [
        node.lineno
        for node in ast.walk(_module_ast())
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "return_offsets_mapping"
    ]
    assert not offenders, (
        f"genomic_lm.py passes return_offsets_mapping to a call at line(s) {offenders}. "
        f"Nucleotide Transformer's tokeniser is EsmTokenizer, is_fast=False; HuggingFace "
        f"raises NotImplementedError for that argument on slow tokenisers. This is the "
        f"exact call that kept genomiclm_llr at 0.0 for every row in the cohort. "
        f"Use centre_token_index() instead."
    )


def test_the_llr_scoring_loop_swallows_nothing():
    """No exception handler anywhere in the LLR path.

    CLAUDE.md 4: 'A bare `except Exception` that logs and continues is a defect, not
    robustness -- it is exactly what erased a base model from the ensemble.' Here it
    erased a feature instead, and did it at logger.debug so not one line was printed.
    """
    tree = _module_ast()
    guarded = {"_masked_centre_logratio", "centre_token_index"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in guarded:
            continue
        handlers = [h for h in ast.walk(node) if isinstance(h, ast.ExceptHandler)]
        assert not handlers, (
            f"{node.name}() contains {len(handlers)} exception handler(s) at line(s) "
            f"{[h.lineno for h in handlers]}. This function must fail loudly. A handler "
            f"here is how genomiclm_llr returned a column of zeros for 4,420,180 rows "
            f"without printing a single line."
        )


def test_no_pragma_no_cover_hides_the_llr_path():
    """`# pragma: no cover` on the original handler told the coverage tool to stop
    looking at the very line that was eating the error. Invisibility as policy.

    THIS TEST'S FIRST VERSION WAS ITSELF THE BUG IT WAS WRITTEN TO PREVENT (2026-07-15).
    ------------------------------------------------------------------------------------
    It banned the substring `pragma: no cover` from every line of the guarded functions.
    It then failed -- on the explanatory comment in `_masked_centre_logratio` that QUOTES
    the deleted handler in order to record why it was deleted::

        #     except Exception as exc:  # pragma: no cover

    That is the third time in this project that a blanket string ban has fired on the
    prose written to explain the very thing being banned (the others: a ban on "1,926"
    matching its own correction note; a feature-count sweep matching "36 of its 78
    features"). The sibling test in this file, test_offset_mapping_is_never_requested_
    from_the_tokeniser, gets it right by parsing with `ast` -- both were written in the
    same sitting.

    `ast` cannot rescue this one: a coverage pragma is ALWAYS a comment, so it never
    appears in the tree. The discriminator is structural instead --

        a REAL pragma follows CODE on its line:   except Exception:  # pragma: no cover
        a QUOTATION is a whole-line comment:      #     except ...   # pragma: no cover

    -- so lines whose first non-whitespace character is `#` are prose and are skipped.
    A ban that cannot distinguish a rule from a description of the rule is not a gate;
    it is a search-and-replace with opinions.
    """
    src = Path(genomic_lm.__file__).read_text(encoding="utf-8").splitlines()
    tree = _module_ast()

    spans: list[tuple[int, int, str]] = [
        (n.lineno, n.end_lineno or n.lineno, n.name)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_masked_centre_logratio", "centre_token_index"}
    ]
    for start, end, name in spans:
        for i in range(start - 1, end):
            line = src[i]
            if line.lstrip().startswith("#"):
                continue  # prose describing the old code, not a live pragma
            assert "pragma: no cover" not in line, (
                f"{name}() line {i + 1} carries 'pragma: no cover' on a line of live "
                f"code. The LLR path is already unreachable in continuous integration "
                f"(no network); telling the coverage tool to ignore it too is how this "
                f"stayed invisible for the connector's entire life.\n"
                f"    {line.strip()}"
            )


def test_this_files_pragma_guard_can_actually_fail():
    """Negative test for the guard above -- CLAUDE.md 6.4: 'A guard is not real until
    you have watched it FAIL.'

    The guard was just weakened to ignore comment lines. That weakening must not have
    blinded it to a real pragma on real code. Proven here on synthetic source rather
    than trusted.
    """
    real_pragma = "    x = 1  # pragma: no cover"
    quoted = "    #     except Exception as exc:  # pragma: no cover"

    def _guard(line: str) -> bool:
        """True == the guard would flag this line. Mirrors the logic above exactly."""
        if line.lstrip().startswith("#"):
            return False
        return "pragma: no cover" in line

    assert _guard(real_pragma), "the guard no longer catches a pragma on live code"
    assert not _guard(quoted), "the guard still fires on prose quoting the old code"


def test_an_all_zero_llr_column_is_unreachable_in_silence():
    """`_masked_centre_logratio` must RAISE rather than return all zeros.

    Binds the invariant by source, since the real path needs the model. The function
    allocates `out = np.zeros(...)` and fills it; the only way that array is returned
    untouched is if every pair was unlocatable, and that must raise.
    """
    tree = _module_ast()
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_masked_centre_logratio"
    )
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert raises, (
        "_masked_centre_logratio() contains no `raise`. It allocates a zero array and "
        "fills it in a loop; with no raise, a systematically-failing tokeniser returns "
        "a silent column of zeros -- which is what happened. It must refuse."
    )


def test_the_module_still_exposes_the_two_features_it_promises():
    """Cheap contract check: the connector's advertised outputs must remain the two
    columns TABULAR_FEATURES expects, so this file fails if they are renamed."""
    from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES

    for col in ("genomiclm_delta_norm", "genomiclm_llr"):
        assert col in TABULAR_FEATURES, f"{col} vanished from TABULAR_FEATURES"
    assert genomic_lm.NucleotideTransformerConnector._SCORE_CACHE_COLS[-2:] == [
        "genomiclm_delta_norm",
        "genomiclm_llr",
    ]
