# feature engineering

import json

import pandas as pd


EVAL_CODES = {'WRONG': 0, 'PARTIAL': 1, 'CORRECT': 2}
N_EVAL_STATES = len(EVAL_CODES)

_ALWAYS_PARTIAL_TYPES = {
    'DND_PAIRS', 'DND_GROUP', 'DND_ORDER', 'DND_IN_TEXT',
    'HIGHLIGHT', 'MULTI_COLOR_HIGHLIGHT', 'FIX_TEXT',
}
_NEVER_PARTIAL_TYPES = {'OPEN', 'THEORY'}


def _partial_rule(qtype, content):
    """Structural rule from the document content; returns True/False/None."""
    if qtype in _NEVER_PARTIAL_TYPES:
        return False
    if qtype in _ALWAYS_PARTIAL_TYPES:
        return True
    if isinstance(content, str):
        try:
            c = json.loads(content)
        except Exception:
            return None
    elif isinstance(content, dict):
        c = content
    else:
        return None

    if qtype == 'MULTIPLE_CHOICE':
        return bool(c.get('multipleResponses'))
    if qtype in ('CLOZE_MATH', 'CLOZE_TEXT_INPUT', 'CLOZE_TEXT_DROPDOWN'):
        return len(c.get('clozeElements') or []) >= 2
    if qtype == 'SEPARATE_TEXT':
        return sum(1 for w in (c.get('words') or [])
                   if w.get('separatorRight', 'NONE') != 'NONE') >= 2
    if qtype == 'SOLUTION_FIELD':
        return len(c.get('solution') or []) >= 2
    if qtype == 'MATH_STEP_BY_STEP':
        return len(c.get('solutionSteps') or []) >= 2
    g = c.get('graph') or {}
    if qtype == 'GRAPH_PLOTTER_POINTS':
        return len(g.get('solutionPoints') or []) >= 2
    if qtype == 'GRAPH_PLOTTER_FIELD':
        return len(g.get('solutionFieldFunctions') or []) >= 2
    if qtype == 'GRAPH_PLOTTER_SELECT':
        return sum(len(v) for v in (g.get('selectableShapes') or {}).values()
                   if isinstance(v, list)) >= 2
    return None


def can_have_partial_answer(document_id, documents, transactions=None):
    """Return True/False/None for whether a document admits partial credit.

    Applies the structural rule on the document's `type` and parsed `content`.
    If `transactions` is given, documents whose rule says False but for which
    PARTIAL was actually observed in the transactions table are flipped to True.
    """
    row = documents.loc[documents['document_id'] == document_id]
    if row.empty:
        return None
    qtype = row['type'].iloc[0]
    content = row['content'].iloc[0]
    rule = _partial_rule(qtype, content)
    if rule is False and transactions is not None:
        observed = transactions.loc[
            (transactions['document_id'] == document_id)
            & (transactions['evaluation'] == 'PARTIAL')
        ]
        if not observed.empty:
            return True
    return rule


def add_partial_answer_flag(documents, transactions=None, column='can_partial'):
    """Vectorized version of `can_have_partial_answer` over the documents table.

    Returns a copy of `documents` with a new boolean (or None) column.
    """
    out = documents.copy()
    out[column] = [
        _partial_rule(t, c) for t, c in zip(out['type'], out['content'])
    ]
    if transactions is not None:
        observed_partial = set(
            transactions.loc[transactions['evaluation'] == 'PARTIAL', 'document_id'].unique()
        )
        flip = (out[column] == False) & (out['document_id'].isin(observed_partial))
        out.loc[flip, column] = True
    return out


def build_feature_matrix(transactions, documents):
    """Build the modeling DataFrame X from the (evaluated) transactions table.

    Output columns: user_id, skill_name, correct, start_time, skill_attempts,
    total_attempts, can_partial, time_since_first_attempt.
    """
    df = transactions.copy()

    # 1. Attach document-level estimatedDifficulty if not already present.
    if 'estimatedDifficulty' not in df.columns:
        diff_map = documents.set_index('document_id')['estimatedDifficulty']
        df['estimatedDifficulty'] = df['document_id'].map(diff_map)

    # 2. Attach can_partial flag from the document content.
    docs_flagged = add_partial_answer_flag(documents, transactions=transactions)
    partial_map = docs_flagged.set_index('document_id')['can_partial']
    df['can_partial'] = df['document_id'].map(partial_map)

    # 3. Time since the user's very first answered question (per user).
    start_dt = pd.to_datetime(df['start_time'], errors='coerce')
    first_attempt = start_dt.groupby(df['user_id']).transform('min')
    df['time_since_first_attempt'] = (start_dt - first_attempt).dt.total_seconds()

    # 4. Keep only the columns needed for modeling.
    X = df[['user_id', 'topic_id', 'evaluation', 'estimatedDifficulty',
            'start_time', 'document_id', 'can_partial',
            'time_since_first_attempt']].copy()

    # 5. Replace missing values with the sentinel -2.
    X = X.fillna(-2)

    # 6. Build a `skill_name` key combining topic and difficulty.
    X['skill_name'] = X['topic_id'].astype(int).astype(str) + '_' + X['estimatedDifficulty'].astype(int).astype(str)

    # 7. Map evaluation strings to ordinal codes.
    X['correct'] = X['evaluation'].map(EVAL_CODES)

    # 8. Trim to modeling columns.
    X = X[['user_id', 'skill_name', 'correct', 'start_time',
           'can_partial', 'time_since_first_attempt']].copy()

    # 9. Causal counters (only past attempts) — practice signal per skill and overall.
    X['skill_attempts'] = X.groupby(['user_id', 'skill_name']).cumcount()
    X['total_attempts'] = X.groupby('user_id').cumcount()

    return X


def prepare_seq(df):
    '''
    Extract user_id sequence in preparation for DKT. The output of this function
    feeds into the prepare_data() function.

    Each per-user tuple contains, aligned to the *next* step:
        past skill_with_answer, next skill, next label,
        next can_partial flag, next time_since_first_attempt (days).
    '''
    # Enumerate skill id as a categorical variable
    # (i.e. [32, 12, 32, 45] -> [0, 1, 0, 2])
    df['skill'], _   = pd.factorize(df['skill_name'], sort=True)
    df['skill_with_answer'] = df['skill'].astype('int32') * 3 + df['correct'].astype('int32')
    df['skill']    = df['skill'].astype('int32')
    df['correct'] = df['correct'].astype('int32')

    # Auxiliary per-step features.
    df['can_partial_f'] = (df['can_partial'] == True).astype('float32')
    # Convert seconds → days so the magnitude is comparable to the one-hot scale.
    df['time_since_first_days'] = (
        df['time_since_first_attempt'].clip(lower=0).astype('float32') / 86400.0
    )

    # Convert to a sequence per user_id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values[:-1],
            r['skill'].values[1:],
            r['correct'].values[1:],
            r['can_partial_f'].values[1:],
            r['time_since_first_days'].values[1:],
        ),
        include_groups=False,
    )

    # Get max skill depth and max feature depth
    skill_depth = df['skill'].max()
    features_depth = df['skill_with_answer'].max() + 1

    return seq, features_depth, skill_depth
