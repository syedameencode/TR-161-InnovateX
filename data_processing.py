"""
data_processing.py
==================
Handles all data ingestion and feature engineering for the Log Anomaly Detection System.

Responsibilities:
  1. Parse structured logs (CSV / JSON) and unstructured raw text logs.
  2. Extract numerical + categorical features (timestamp deltas, HTTP status codes,
     error severity levels, request sizes, response times).
  3. Vectorise free-text log messages with TF-IDF so the ML models receive a fully
     numerical feature matrix.

Author: AI/ML Engineer  |  Hackathon 2026
"""

import re
import json
import io
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, issparse

# ---------------------------------------------------------------------------
# 1. CONSTANTS & REGEXES
# ---------------------------------------------------------------------------

# Common Apache / Nginx combined-log-format regex
APACHE_PATTERN = re.compile(
    r'(?P<ip>\S+)\s+'           # client IP
    r'\S+\s+\S+\s+'             # ident, auth
    r'\[(?P<ts>[^\]]+)\]\s+'    # timestamp
    r'"(?P<method>\S+)?\s*'     # HTTP method
    r'(?P<path>\S+)?\s*'        # request path
    r'\S+"\s+'                  # protocol
    r'(?P<status>\d{3})\s+'     # status code
    r'(?P<size>\S+)'            # response size
    r'(?:\s+"[^"]*"\s+"[^"]*")?' # referrer + UA (optional)
)

# Syslog-style: 2024-01-15 12:34:56 [ERROR] message
SYSLOG_PATTERN = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)'
    r'(?:\s+\[(?P<level>[A-Z]+)\])?'
    r'(?:\s+(?P<component>\S+):)?'
    r'\s*(?P<message>.*)'
)

# Severity / error-level keywords → numeric weight
LEVEL_MAP = {
    'DEBUG': 0, 'TRACE': 0,
    'INFO': 1,
    'NOTICE': 2,
    'WARNING': 3, 'WARN': 3,
    'ERROR': 4,
    'CRITICAL': 5, 'FATAL': 5, 'EMERG': 5, 'ALERT': 5,
}

# HTTP status → severity proxy (used when parsing web-server logs)
def _http_status_severity(code: int) -> int:
    """Maps HTTP status codes to a 0-5 severity scale."""
    if code < 300:
        return 0
    if code < 400:
        return 1
    if code < 500:
        return 2
    return 4  # 5xx server errors = high severity


# ---------------------------------------------------------------------------
# 2. PARSERS
# ---------------------------------------------------------------------------

def _parse_csv(content: str) -> pd.DataFrame:
    """
    Parse a CSV log file. Automatically detects common column aliases for
    timestamp, message, and severity level fields.
    """
    df = pd.read_csv(io.StringIO(content))

    # Normalise column names (lower-case, strip spaces)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Map common alias names → canonical names
    ts_aliases      = ['timestamp', 'time', 'date', 'datetime', '@timestamp']
    msg_aliases     = ['message', 'msg', 'log', 'log_message', 'text', 'body']
    level_aliases   = ['level', 'severity', 'log_level', 'loglevel']
    status_aliases  = ['status', 'status_code', 'http_status', 'response_code']
    size_aliases    = ['size', 'bytes', 'response_size', 'body_bytes_sent']
    rt_aliases      = ['response_time', 'duration', 'elapsed', 'latency_ms']

    def _pick(df, aliases, fallback=None):
        for a in aliases:
            if a in df.columns:
                return a
        return fallback

    ts_col     = _pick(df, ts_aliases)
    msg_col    = _pick(df, msg_aliases)
    level_col  = _pick(df, level_aliases)
    status_col = _pick(df, status_aliases)
    size_col   = _pick(df, size_aliases)
    rt_col     = _pick(df, rt_aliases)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            'raw_line'      : ' '.join(row.astype(str).tolist()),
            'timestamp'     : row[ts_col]     if ts_col     else None,
            'message'       : str(row[msg_col]) if msg_col   else '',
            'level'         : str(row[level_col]).upper() if level_col else 'INFO',
            'http_status'   : int(row[status_col]) if status_col and str(row[status_col]).isdigit() else None,
            'response_size' : _safe_float(row[size_col])  if size_col  else None,
            'response_time' : _safe_float(row[rt_col])    if rt_col    else None,
        })
    return pd.DataFrame(rows)


def _parse_json_lines(content: str) -> pd.DataFrame:
    """
    Parse newline-delimited JSON (NDJSON) log files.
    Each line is expected to be a valid JSON object.
    """
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # Skip malformed lines gracefully

        rows.append({
            'raw_line'      : line,
            'timestamp'     : _first_val(obj, ['timestamp', 'time', '@timestamp', 'date']),
            'message'       : _first_val(obj, ['message', 'msg', 'log', 'body'], default=''),
            'level'         : str(_first_val(obj, ['level', 'severity', 'log_level'], default='INFO')).upper(),
            'http_status'   : _safe_int(_first_val(obj, ['status', 'status_code', 'http_status'])),
            'response_size' : _safe_float(_first_val(obj, ['size', 'bytes', 'response_size'])),
            'response_time' : _safe_float(_first_val(obj, ['duration', 'response_time', 'elapsed', 'latency_ms'])),
        })
    return pd.DataFrame(rows)


def _parse_raw_text(content: str) -> pd.DataFrame:
    """
    Parse unstructured / free-text log files.
    Tries Apache/Nginx combined format first, then syslog format.
    Falls back to treating the entire line as a raw message.
    """
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        m_apache = APACHE_PATTERN.match(line)
        m_syslog = SYSLOG_PATTERN.match(line)

        if m_apache:
            g = m_apache.groupdict()
            status = _safe_int(g.get('status'))
            rows.append({
                'raw_line'      : line,
                'timestamp'     : g.get('ts'),
                'message'       : f"{g.get('method','')} {g.get('path','')}",
                'level'         : 'ERROR' if (status and status >= 500) else 'WARNING' if (status and status >= 400) else 'INFO',
                'http_status'   : status,
                'response_size' : _safe_float(g.get('size')),
                'response_time' : None,
            })
        elif m_syslog:
            g = m_syslog.groupdict()
            rows.append({
                'raw_line'      : line,
                'timestamp'     : g.get('ts'),
                'message'       : g.get('message', line),
                'level'         : (g.get('level') or 'INFO').upper(),
                'http_status'   : None,
                'response_size' : None,
                'response_time' : None,
            })
        else:
            # Fallback: no recognised pattern
            rows.append({
                'raw_line'      : line,
                'timestamp'     : None,
                'message'       : line,
                'level'         : _extract_level_from_text(line),
                'http_status'   : _extract_status_from_text(line),
                'response_size' : None,
                'response_time' : _extract_response_time_from_text(line),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. UNIFIED INGESTION ENTRY POINT
# ---------------------------------------------------------------------------

def ingest_logs(content: str, filename: str = '') -> pd.DataFrame:
    """
    Auto-detects log format (JSON-lines, CSV, or raw text) and returns a
    clean, unified DataFrame ready for feature extraction.

    Parameters
    ----------
    content  : str  — raw file text
    filename : str  — original filename (used for extension-based hints)

    Returns
    -------
    pd.DataFrame with columns:
        raw_line, timestamp, message, level, http_status,
        response_size, response_time
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    # --- Format detection ---
    if ext in ('json', 'jsonl', 'ndjson') or _looks_like_json_lines(content):
        df = _parse_json_lines(content)
    elif ext == 'csv' or _looks_like_csv(content):
        df = _parse_csv(content)
    else:
        df = _parse_raw_text(content)

    if df.empty:
        raise ValueError("No parseable log lines found in the uploaded file.")

    # --- Reset index & add line numbers for traceability ---
    df = df.reset_index(drop=True)
    df.insert(0, 'line_no', df.index + 1)

    logging.info(f"[Ingestion] Parsed {len(df)} log lines from '{filename}' (format={ext or 'auto'})")
    return df


# ---------------------------------------------------------------------------
# 4. FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Converts the unified log DataFrame into a numerical feature matrix
    suitable for the anomaly detection models.

    Features used:
      • severity_score  — numeric mapping of log level (0-5)
      • http_status_sev — HTTP status severity proxy (0-4)
      • response_size   — log-scaled response size in bytes
      • response_time   — log-scaled response time in ms
      • hour_of_day     — hour extracted from timestamp (0-23)
      • is_error_kw     — 1 if message contains error keywords
      • tfidf_*         — top TF-IDF components from raw message text

    Returns
    -------
    X       : np.ndarray of shape (n_samples, n_features)  — feature matrix
    feat_df : pd.DataFrame                                  — interpretable features
    """
    n = len(df)

    # 4a. Severity level score
    severity = df['level'].apply(lambda lv: LEVEL_MAP.get(str(lv).upper(), 1)).values.astype(float)

    # 4b. HTTP status severity
    http_sev = df['http_status'].apply(
        lambda s: _http_status_severity(int(s)) if pd.notna(s) else 1.0
    ).values.astype(float)

    # 4c. Response size — log scale (avoids domination by huge values)
    size_raw = pd.to_numeric(df['response_size'], errors='coerce').fillna(0).values
    size_feat = np.log1p(size_raw)

    # 4d. Response time — log scale
    rt_raw = pd.to_numeric(df['response_time'], errors='coerce').fillna(0).values
    rt_feat = np.log1p(rt_raw)

    # 4e. Hour of day extracted from timestamp string
    hours = df['timestamp'].apply(_extract_hour).values.astype(float)

    # 4f. Binary error-keyword flag
    error_kw_re = re.compile(
        r'\b(error|exception|fail|fatal|critical|traceback|segfault|oom|kill|crash|timeout|refused|denied)\b',
        re.IGNORECASE
    )
    error_flag = df['message'].apply(lambda m: 1.0 if error_kw_re.search(str(m)) else 0.0).values

    # 4g. HTTP status code as raw numeric feature (0 if not present)
    http_code = pd.to_numeric(df['http_status'], errors='coerce').fillna(200).values.astype(float)

    # 4h. TF-IDF on raw messages (max 50 features to keep matrix manageable)
    messages = df['message'].fillna('').astype(str).tolist()
    tfidf = TfidfVectorizer(
        max_features=50,
        stop_words='english',
        ngram_range=(1, 2),  # unigrams + bigrams capture multi-word patterns
        sublinear_tf=True,   # apply log(1+tf) to compress term frequency
    )
    try:
        tfidf_matrix = tfidf.fit_transform(messages).toarray()
    except ValueError:
        # Edge case: all messages empty
        tfidf_matrix = np.zeros((n, 1))

    # ---  Assemble structured numeric features into a DataFrame ---
    feat_df = pd.DataFrame({
        'severity_score'  : severity,
        'http_status_sev' : http_sev,
        'response_size'   : size_feat,
        'response_time'   : rt_feat,
        'hour_of_day'     : hours,
        'error_keyword'   : error_flag,
        'http_code'       : http_code,
    })

    # Combine structured features with TF-IDF
    structured = feat_df.values
    X = np.hstack([structured, tfidf_matrix])

    logging.info(f"[Features] Matrix shape: {X.shape}")
    return X, feat_df


# ---------------------------------------------------------------------------
# 5. HELPER UTILITIES
# ---------------------------------------------------------------------------

def _looks_like_json_lines(content: str) -> bool:
    """Heuristic: if ≥50% of non-empty lines start with '{', treat as NDJSON."""
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return False
    json_like = sum(1 for l in lines[:20] if l.startswith('{'))
    return json_like / min(len(lines), 20) >= 0.5


def _looks_like_csv(content: str) -> bool:
    """Heuristic: first line has commas and looks like a header."""
    first = content.splitlines()[0] if content else ''
    return first.count(',') >= 2


def _first_val(obj: dict, keys: list, default=''):
    """Return the value of the first matching key in a dict."""
    for k in keys:
        if k in obj:
            return obj[k]
    return default


def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_level_from_text(text: str) -> str:
    """Scan free-text for recognisable log-level keywords."""
    for lvl in ['FATAL', 'CRITICAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG', 'TRACE']:
        if lvl in text.upper():
            return lvl
    return 'INFO'


def _extract_status_from_text(text: str) -> int | None:
    """Try to pull an HTTP status code from a free-text log line."""
    m = re.search(r'\b([1-5]\d{2})\b', text)
    return int(m.group(1)) if m else None


def _extract_response_time_from_text(text: str) -> float | None:
    """Look for patterns like '123ms', '0.45s', 'took 200ms'."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*ms', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+(?:\.\d+)?)\s*s(?:ec)?\b', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000  # convert to ms
    return None


def _extract_hour(ts_val) -> float:
    """Parse a timestamp value and return the hour (0-23). Returns 0.0 on failure."""
    if pd.isna(ts_val) or ts_val is None:
        return 0.0
    ts_str = str(ts_val).strip()
    # Try common formats
    for fmt in (
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
        '%d/%b/%Y:%H:%M:%S %z', '%d/%b/%Y:%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f',
    ):
        try:
            return float(datetime.strptime(ts_str[:len(fmt)], fmt).hour)
        except (ValueError, TypeError):
            continue
    # Fallback: try pandas
    try:
        return float(pd.to_datetime(ts_str).hour)
    except Exception:
        return 0.0
