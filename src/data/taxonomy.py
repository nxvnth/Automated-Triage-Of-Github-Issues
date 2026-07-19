"""Raw repo labels -> canonical `type` in {bug, feature, question, docs}.

Every repo labels differently (`bug`, `c-bug`, `Issue-Bug`, `type: bug` ...).
We lowercase, split the comma-joined label string into tokens, and match one
regex per class PER TOKEN so word boundaries behave (`\\bbug\\b` matches
`bug-report` but not vscode's `debug` label).

Issues matching no class get type None — we don't guess; they're excluded
from type training but still usable for priority/severity.

~3% of issues match 2+ classes; PRECEDENCE resolves them. Bug comes before
docs: an audit of the 430 bug+docs conflicts showed ~2/3 are genuinely bug
reports (repos attach `documentation` to a bug as a "docs need updating too"
flag, and labels like `addon: docs` name software components, not
documentation work). Bug-first also parks the residual ambiguity in the
large bug class instead of polluting the rare docs class. Full audit trail
in the EDA notebook.
"""
import re

CLASS_PATTERNS = {
    'docs':     r'\bdocs?\b|documentation|typo',
    'bug':      r'\bbug\b|defect|\bcrash\b|regression|broken',
    'feature':  r'\bfeature\b|enhancement|suggestion|proposal|\brequest\b|improvement|\bidea\b',
    'question': r'question|\bfaq\b|q&a|discussion',
}
PRECEDENCE = ['bug', 'docs', 'feature', 'question']
_compiled = {c: re.compile(p) for c, p in CLASS_PATTERNS.items()}


def label_classes(raw_labels: str) -> set:
    """Map a raw comma-joined label string to the set of matching classes."""
    if not isinstance(raw_labels, str):
        return set()
    tokens = [t.strip() for t in raw_labels.lower().split(',')]
    return {c for c, rx in _compiled.items() for t in tokens if rx.search(t)}


def resolve(classes: set):
    """Pick a single type from a set of matched classes (None if empty)."""
    for c in PRECEDENCE:
        if c in classes:
            return c
    return None


def to_type(raw_labels: str):
    """Raw label string -> canonical type or None."""
    return resolve(label_classes(raw_labels))
