"""Text cleaning + PII redaction for GitHub issues.

This module is the single source of truth for turning a raw issue into
model-ready text. It is imported by BOTH the data pipeline (src/make_dataset.py)
and the serving layer, so training data and live inputs always go through the
same transformations — any divergence between the two is train/serve skew.

Cleaning: fenced code blocks -> [CODE], stack traces (fenced or pasted bare)
-> [TRACEBACK], markdown normalised, links/images/URLs replaced by sentinels.
The final exception-message line of a trace is kept — it's the most
informative line. Counts of code blocks / tracebacks are returned as features.

Redaction: emails, @-mentions, home-dir paths, IPs, and leaked credentials
become sentinel tokens. Biased toward over-redaction: a lost token costs a
sliver of accuracy, a leaked username is a governance failure.
"""
import re

# --- markdown / structure patterns ---
RE_HTML_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)
RE_FENCE = re.compile(r'```.*?(?:```|$)|~~~.*?(?:~~~|$)', re.DOTALL)  # tolerate unclosed fences
RE_INLINE_CODE = re.compile(r'`([^`\n]+)`')
RE_IMAGE = re.compile(r'!\[([^\]]*)\]\([^)]*\)')
RE_LINK = re.compile(r'\[([^\]]+)\]\([^)]*\)')
RE_URL = re.compile(r'https?://[^\s\'")\]>]+')
RE_MD_MARKUP = re.compile(r'^#{1,6}\s+|^\s*>\s?|[*_]{2,}', re.MULTILINE)
RE_WS = re.compile(r'[ \t]+')
RE_BLANKS = re.compile(r'\n{3,}')

# Is this fenced block a stack trace? (Python, Java/JS, native, Go, Rust)
RE_TB_HINT = re.compile(
    r'Traceback \(most recent call last\)'
    r'|^\s*File "[^"]+", line \d+'
    r'|^\s*at [\w$.<>/@ -]+ ?\([^)\n]*\)\s*$'
    r'|Exception in thread'
    r'|^\s*#\d+\s+0x[0-9a-f]+'
    r'|^panic: |^goroutine \d+'
    r"|thread '[^']*' panicked",
    re.MULTILINE)

# Single-line frame detector for UNFENCED tracebacks pasted as plain text
RE_TB_LINE = re.compile(
    r'^\s*File "[^"]+", line \d+'
    r'|^\s*at [\w$.<>/@ -]+ ?\([^)\n]*\)\s*$'
    r'|^\s*#\d+\s+0x[0-9a-f]+'
    r'|Traceback \(most recent call last\)')


def clean_body(body):
    """Return (clean_text, n_code_blocks, n_tracebacks) for one issue body."""
    if not isinstance(body, str):
        return '', 0, 0
    n_code = n_tb = 0
    text = RE_HTML_COMMENT.sub(' ', body)          # template boilerplate first

    def fence_repl(m):
        nonlocal n_code, n_tb
        if RE_TB_HINT.search(m.group(0)):
            n_tb += 1
            return ' [TRACEBACK] '
        n_code += 1
        return ' [CODE] '

    text = RE_FENCE.sub(fence_repl, text)

    # Collapse runs of unfenced traceback lines into one token.
    # While inside a trace, indented lines are frame source lines -> swallow;
    # the first NON-indented non-frame line (the exception message) is kept.
    out, in_tb = [], False
    for ln in text.split('\n'):
        if RE_TB_LINE.search(ln):
            if not in_tb:
                out.append('[TRACEBACK]')
                n_tb += 1
                in_tb = True
        elif in_tb and (ln[:1] in (' ', '\t') or not ln.strip()):
            continue
        else:
            in_tb = False
            out.append(ln)
    text = '\n'.join(out)

    text = RE_INLINE_CODE.sub(r'\1', text)         # keep inline-code content
    text = RE_IMAGE.sub(r'\1 [IMAGE]', text)
    text = RE_LINK.sub(r'\1 [URL]', text)          # keep anchor text
    text = RE_URL.sub('[URL]', text)
    text = RE_MD_MARKUP.sub(' ', text)
    text = RE_WS.sub(' ', text)
    text = RE_BLANKS.sub('\n\n', text).strip()
    return text, n_code, n_tb


# --- PII redaction ---
# List order = application order: tokens first (an unredacted ghp_... is the
# worst outcome), emails before mentions (else the @domain half of an email
# would be half-eaten).
PII_PATTERNS = [
    ('[TOKEN]', re.compile(
        r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}'
        r'|github_pat_[A-Za-z0-9_]{20,}'
        r'|\bAKIA[0-9A-Z]{16}\b'
        r'|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}'   # JWT
        r'|\bxox[baprs]-[A-Za-z0-9-]{10,}')),                                # Slack
    ('[EMAIL]', re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b')),  # alphabetic TLD: pkg@3.1.0 survives
    ('[USER_PATH]', re.compile(r'(?:/(?:home|Users)/|C:\\+Users\\+)[\w.-]+', re.IGNORECASE)),
    ('[USER]', re.compile(r'(?<![\w.+-])@[A-Za-z\d][A-Za-z\d-]{0,38}\b(?!/)')),  # (?!/) skips @types/node
    ('[IP]', re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
]


def redact(text: str) -> str:
    """Replace PII with sentinel tokens.

    Substitutes to a FIXPOINT, not a single pass: lookbehinds are evaluated
    against the pre-replacement string, so in '@a@b' the second mention only
    becomes matchable after the first is replaced. One pass left 8 leaks.
    """
    for token, rx in PII_PATTERNS:
        prev = None
        while prev != text:
            prev = text
            text = rx.sub(token, text)
    return text


def prepare_text(title, body):
    """Raw title + body -> (model_text, n_code_blocks, n_tracebacks).

    The full raw-issue -> model-input transformation. The serving layer must
    call exactly this on incoming issues before prediction.
    Title goes first: it's the densest signal, and transformers truncate
    from the right.
    """
    body_clean, n_code, n_tb = clean_body(body)
    title_clean = redact(title if isinstance(title, str) else '')
    body_clean = redact(body_clean)
    text = (title_clean + '\n' + body_clean).strip()
    return text, n_code, n_tb
