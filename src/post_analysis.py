"""Post-training analysis helpers used by Post_Analysis.ipynb."""

import collections

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

from src.features import N_EVAL_STATES


# ---------------------------------------------------------------------------
# Sequence -> per-user arrays
# ---------------------------------------------------------------------------

def unpack_seq(seq):
    """Pull per-user arrays from the prepare_seq() output.

    Returns a dict with keys:
        uids   : list of user ids that have at least one prediction step.
        feats  : list[np.int32]   — past skill_with_answer.
        skills : list[np.int32]   — next skill.
        cps    : list[np.float32] — next can_partial flag.
        tsfs   : list[np.float32] — next time_since_first_attempt (days).
        lens   : list[int]        — sequence lengths per user.
    """
    uids   = [uid for uid, tup in seq.items() if len(tup[0]) > 0]
    feats  = [seq[uid][0].astype(np.int32)   for uid in uids]
    skills = [seq[uid][1].astype(np.int32)   for uid in uids]
    cps    = [seq[uid][3].astype(np.float32) for uid in uids]
    tsfs   = [seq[uid][4].astype(np.float32) for uid in uids]
    lens   = [len(f) for f in feats]
    return {
        'uids': uids, 'feats': feats, 'skills': skills,
        'cps': cps, 'tsfs': tsfs, 'lens': lens,
    }


def _pad_batch(fb, sb, cb, tb):
    """Right-pad an aligned (feat, skill, can_partial, time_since) batch."""
    Lmax = max(len(f) for f in fb)
    fp = np.zeros((len(fb), Lmax), dtype=np.int32)
    sp = np.zeros((len(sb), Lmax), dtype=np.int32) if sb is not None else None
    cp = np.zeros((len(cb), Lmax), dtype=np.float32)
    tp = np.zeros((len(tb), Lmax), dtype=np.float32)
    for i in range(len(fb)):
        fp[i, :len(fb[i])] = fb[i]
        if sp is not None:
            sp[i, :len(sb[i])] = sb[i]
        cp[i, :len(cb[i])] = cb[i]
        tp[i, :len(tb[i])] = tb[i]
    return fp, sp, cp, tp


def _build_inputs(fp, cp, tp, features_depth):
    """One-hot the past features and concat the two scalar aux features."""
    feat_oh = tf.one_hot(fp, depth=features_depth)
    aux     = tf.stack([cp, tp], axis=-1)
    return tf.concat([feat_oh, aux], axis=-1)


# ---------------------------------------------------------------------------
# Per-row predictions: P(WRONG/PARTIAL/CORRECT) on the skill the user answered.
# ---------------------------------------------------------------------------

def predict_per_row_probs(seq, dkt_model, X, features_depth, batch_size):
    """Run the trained DKT and align next-step probabilities back to X rows.

    Returns a (len(X), N_EVAL_STATES) float32 array. Row 0 per user has no
    predecessor so its row is left as NaN.
    """
    bundle = unpack_seq(seq)
    uids, feats, skills = bundle['uids'], bundle['feats'], bundle['skills']
    cps, tsfs, lens = bundle['cps'], bundle['tsfs'], bundle['lens']

    probs_per_row = np.full((len(X), N_EVAL_STATES), np.nan, dtype=np.float32)
    user_row_idx  = X.groupby('user_id').indices

    for start in range(0, len(uids), batch_size):
        end = start + batch_size
        fb, sb = feats[start:end], skills[start:end]
        cb, tb = cps[start:end],   tsfs[start:end]
        fp, sp, cp, tp = _pad_batch(fb, sb, cb, tb)
        feat_full = _build_inputs(fp, cp, tp, features_depth)
        logits = dkt_model({'features': feat_full, 'next_skill': sp},
                           training=False).numpy()
        probs  = tf.nn.softmax(logits, axis=-1).numpy()
        for i, uid in enumerate(uids[start:end]):
            n    = lens[start + i]
            rows = user_row_idx[uid]
            probs_per_row[rows[1:1 + n]] = probs[i, :n]
    return probs_per_row, user_row_idx


# ---------------------------------------------------------------------------
# Per-skill predictions: probabilities for *every* skill at every step.
# ---------------------------------------------------------------------------

def _all_skills_submodel(dkt_model):
    """Tap the layer just before GatherSkill so we get the full (B,T,S,3) cube."""
    reshape_layer = next(
        l for l in dkt_model.layers if isinstance(l, tf.keras.layers.Reshape)
    )
    return tf.keras.Model(
        inputs  = dkt_model.get_layer('features').input,
        outputs = reshape_layer.output,
    )


def predict_per_skill_probs(seq, dkt_model, user_row_idx, features_depth,
                            batch_size, progress=False):
    """Long-format DataFrame: one row per (user_id, row_idx_in_X, skill).

    Columns: user_id, row_idx, skill, p_wrong, p_partial, p_correct.
    """
    bundle = unpack_seq(seq)
    uids, feats = bundle['uids'], bundle['feats']
    cps, tsfs, lens = bundle['cps'], bundle['tsfs'], bundle['lens']

    sub = _all_skills_submodel(dkt_model)
    n_skills = int(sub.output_shape[2])

    iterator = range(0, len(uids), batch_size)
    if progress:
        from tqdm.auto import tqdm
        n_batches = (len(uids) + batch_size - 1) // batch_size
        iterator = tqdm(iterator, total=n_batches, desc='user-batches')

    chunks = []
    for start in iterator:
        end = start + batch_size
        fb = feats[start:end]
        cb, tb = cps[start:end], tsfs[start:end]
        fp, _, cp, tp = _pad_batch(fb, None, cb, tb)
        feat_full = _build_inputs(fp, cp, tp, features_depth)
        logits = sub(feat_full, training=False).numpy()
        per_skill_probs = tf.nn.softmax(logits, axis=-1).numpy()
        for i, uid in enumerate(uids[start:end]):
            n = lens[start + i]
            if n == 0:
                continue
            rows  = user_row_idx[uid][1:1 + n]
            block = per_skill_probs[i, :n].reshape(n * n_skills, N_EVAL_STATES)
            chunk = pd.DataFrame(block, columns=['p_wrong', 'p_partial', 'p_correct'])
            chunk['user_id'] = uid
            chunk['row_idx'] = np.repeat(rows, n_skills)
            chunk['skill']   = np.tile(np.arange(n_skills), n)
            chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    return out[['user_id', 'row_idx', 'skill', 'p_wrong', 'p_partial', 'p_correct']]


# ---------------------------------------------------------------------------
# Plotting: predicted-probability traces for one or two users.
# ---------------------------------------------------------------------------

_EVAL_COLOR = {0: 'tab:red', 1: 'tab:orange', 2: 'tab:green'}
_EVAL_LABEL = {0: 'WRONG', 1: 'PARTIAL', 2: 'CORRECT'}
_LINESTYLES = ['-', '--', ':', '-.']
_MARKERS    = ['o', 's', '^', 'D']


def plot_user_skill_traces(ax, user_id, skill_ids, X, skill_probs, user_row_idx):
    """Plot P(WRONG/PARTIAL/CORRECT) over time for one user × several skills."""
    user_rows = user_row_idx[user_id]
    step_of   = pd.Series(np.arange(len(user_rows)), index=user_rows)
    attempts_user = X.iloc[user_rows].copy()
    attempts_user['step'] = np.arange(len(attempts_user))

    for k, skill_id in enumerate(skill_ids):
        ls = _LINESTYLES[k % len(_LINESTYLES)]
        trace = (
            skill_probs[(skill_probs['user_id'] == user_id)
                        & (skill_probs['skill'] == skill_id)]
            .sort_values('row_idx').reset_index(drop=True)
        )
        trace['step'] = trace['row_idx'].map(step_of)
        ax.plot(trace['step'], trace['p_wrong'],   lw=1.4, ls=ls,
                color='tab:red',    label=f'P(WRONG) – skill {skill_id}')
        ax.plot(trace['step'], trace['p_partial'], lw=1.4, ls=ls,
                color='tab:orange', label=f'P(PARTIAL) – skill {skill_id}')
        ax.plot(trace['step'], trace['p_correct'], lw=1.4, ls=ls,
                color='tab:green',  label=f'P(CORRECT) – skill {skill_id}')

        sub = attempts_user[attempts_user['skill'] == skill_id]
        y0  = 0.02 + 0.10 * k
        for code, sub_c in sub.groupby('correct'):
            ax.scatter(sub_c['step'], np.full(len(sub_c), y0 + 0.025 * code),
                       color=_EVAL_COLOR[code], s=40,
                       edgecolor='black', linewidth=0.5,
                       marker=_MARKERS[k % len(_MARKERS)],
                       label=f'skill {skill_id} – {_EVAL_LABEL[code]}', zorder=3)

    ax.set_xlabel('interaction step (per user)')
    ax.set_ylim(0, 1)
    ax.set_title(f'user {user_id} — skills {skill_ids}')
    ax.legend(loc='best', fontsize=8, ncol=2)


# ---------------------------------------------------------------------------
# Skill metadata + topic-tree helpers
# ---------------------------------------------------------------------------

def build_skill_info(X, id_to_name):
    """One row per skill with topic_id / topic_name / difficulty / n_attempts."""
    counts = X.groupby(['skill', 'skill_name']).size().rename('n_attempts').reset_index()
    split = counts['skill_name'].str.rsplit('_', n=1, expand=True)
    counts['topic_id']   = split[0].astype(int)
    counts['difficulty'] = split[1].astype(int)
    counts['topic_name'] = counts['topic_id'].map(id_to_name)
    return (
        counts[['skill', 'skill_name', 'topic_id', 'topic_name',
                'difficulty', 'n_attempts']]
        .sort_values('n_attempts', ascending=False)
        .reset_index(drop=True)
    )


def top_topics_with_siblings(skill_info, child_to_parent, top_n=15):
    """Top-N most-represented topics, then re-add their topic-tree siblings."""
    topic_attempts = (
        skill_info.groupby('topic_id', as_index=False)
                  .agg(topic_name=('topic_name', 'first'),
                       n_attempts=('n_attempts', 'sum'),
                       n_difficulty_bands=('skill', 'nunique'))
                  .sort_values('n_attempts', ascending=False)
    )
    top_topics = topic_attempts.head(top_n)['topic_id'].tolist()

    parent_to_children = collections.defaultdict(list)
    for child, parent in child_to_parent.items():
        parent_to_children[parent].append(child)

    expanded = set()
    for tid in top_topics:
        expanded.add(tid)
        parent = child_to_parent.get(tid)
        if parent is not None:
            expanded.update(parent_to_children.get(parent, []))

    return (
        topic_attempts[topic_attempts['topic_id'].isin(expanded)]
        .assign(in_top_n=lambda d: d['topic_id'].isin(top_topics))
        .sort_values(['n_attempts', 'topic_id'], ascending=[False, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Topic-impact analysis: |Δ P(correct on B)| around an answer to A.
# ---------------------------------------------------------------------------

def build_pairwise_deltas(X, skill_probs, user_row_idx):
    """Pre-compute the structures shared by single-source and all-pairs analyses.

    Returns:
        wide_pc        : DataFrame (row_idx × skill) of P(correct).
        next_pos_arr   : ndarray[len(X)] mapping each X row to the next row of
                         the same user (-1 if last).
        delta_all      : (n_events, n_skills) |Δ P(correct)| across all events
                         with both ends predicted.
        source_skill   : ndarray[n_events] of the skill answered at each event.
        all_pos        : ndarray[n_events] X positions of the source row.
        nxt_all        : ndarray[n_events] X positions of the next row.
        source_correct : ndarray[n_events] evaluation code at each event
                         (0=WRONG, 1=PARTIAL, 2=CORRECT).
        before         : (n_events, n_skills) P(correct) at the source row,
                         needed to compute headroom-normalized deltas.
    """
    wide_pc = (skill_probs
               .pivot_table(index='row_idx', columns='skill', values='p_correct')
               .sort_index())

    next_pos_arr = np.full(len(X), -1, dtype=np.int64)
    for uid, positions in user_row_idx.items():
        if len(positions) > 1:
            next_pos_arr[positions[:-1]] = positions[1:]

    all_pos = np.arange(len(X))
    nxt_all = next_pos_arr
    keep = nxt_all != -1
    all_pos, nxt_all = all_pos[keep], nxt_all[keep]
    valid = wide_pc.index
    keep = np.isin(all_pos, valid) & np.isin(nxt_all, valid)
    all_pos, nxt_all = all_pos[keep], nxt_all[keep]

    source_skill   = X['skill'].values[all_pos]
    source_correct = X['correct'].values[all_pos]   # 0=WRONG, 1=PARTIAL, 2=CORRECT
    before = wide_pc.loc[all_pos].to_numpy(dtype=np.float32)
    after  = wide_pc.loc[nxt_all].to_numpy(dtype=np.float32)
    delta_all = np.abs(after - before)
    return (wide_pc, next_pos_arr, delta_all, source_skill,
            all_pos, nxt_all, source_correct, before)


def topic_impact_for_source(source_skill_id, wide_pc, delta_all, source_skill_arr,
                            skill_info):
    """Mean / std / 95% CI of |Δ P(correct on B)| for every target B,
    restricted to events where the user just answered `source_skill_id`."""
    mask = source_skill_arr == source_skill_id
    delta = delta_all[mask]
    n_obs = delta.shape[0]
    mean_d = delta.mean(axis=0)
    std_d  = delta.std(axis=0, ddof=1)
    sem    = std_d / np.sqrt(n_obs)
    ci_half = stats.t.ppf(0.975, df=n_obs - 1) * sem

    impact = pd.DataFrame({
        'target_skill':   wide_pc.columns,
        'mean_abs_delta': mean_d,
        'std_abs_delta':  std_d,
        'ci_low':         mean_d - ci_half,
        'ci_high':        mean_d + ci_half,
        'n_obs':          n_obs,
    })
    impact = (impact
              .merge(skill_info[['skill', 'topic_id', 'topic_name', 'difficulty']],
                     left_on='target_skill', right_on='skill', how='left')
              .drop(columns='skill')
              .sort_values('mean_abs_delta', ascending=False)
              .reset_index(drop=True))
    return impact, n_obs


def all_skill_interactions(wide_pc, delta_all, source_skill_arr, skill_info):
    """Long-format (source, target) → mean |Δ P(correct)|, 95% CI, n_obs.
    Diagonal (source == target) is dropped."""
    deltas = pd.DataFrame(delta_all, columns=wide_pc.columns)
    deltas['source'] = source_skill_arr
    g = deltas.groupby('source')

    mean_long = g.mean().stack(dropna=False).rename('mean_abs_delta')
    std_long  = g.std().stack(dropna=False).rename('std_abs_delta')
    long = pd.concat([mean_long, std_long], axis=1).reset_index()
    long.columns = ['source', 'target', 'mean_abs_delta', 'std_abs_delta']
    long['n_obs'] = long['source'].map(g.size()).astype(int)
    long = long[long['source'] != long['target']].copy()
    long = long.dropna(subset=['mean_abs_delta'])

    sem    = long['std_abs_delta'] / np.sqrt(long['n_obs'])
    t_crit = stats.t.ppf(0.975, df=long['n_obs'] - 1)
    long['ci_low']  = long['mean_abs_delta'] - t_crit * sem
    long['ci_high'] = long['mean_abs_delta'] + t_crit * sem

    name_of = skill_info.set_index('skill')[['topic_name', 'difficulty']]
    long['source_name'] = long['source'].map(name_of['topic_name'])
    long['source_diff'] = long['source'].map(name_of['difficulty'])
    long['target_name'] = long['target'].map(name_of['topic_name'])
    long['target_diff'] = long['target'].map(name_of['difficulty'])
    return long


# ---------------------------------------------------------------------------
# Topic-tree relationship labels
# ---------------------------------------------------------------------------

def _ancestors(node, child_to_parent):
    """Strict ancestors of `node` (excluding node itself), walked to the root."""
    out, cur = set(), child_to_parent.get(node)
    while cur is not None and cur not in out:
        out.add(cur)
        cur = child_to_parent.get(cur)
    return out


def topic_relation(a, b, child_to_parent):
    """Transitive tree relationship of topic A relative to topic B.

    Buckets:
      - 'ancestor→descendant' : A is an ancestor of B (any depth)
      - 'descendant→ancestor' : A is a descendant of B (any depth)
      - 'sibling'             : A and B share the same immediate parent
      - 'other'               : everything else (including cousins, unrelated branches)
    """
    if a == b: return 'other'
    anc_a = _ancestors(a, child_to_parent)
    anc_b = _ancestors(b, child_to_parent)
    if a in anc_b: return 'ancestor→descendant'
    if b in anc_a: return 'descendant→ancestor'
    pa, pb = child_to_parent.get(a), child_to_parent.get(b)
    if pa is not None and pa == pb: return 'sibling'
    return 'other'


def skill_relation(ta, tb, child_to_parent):
    """Same as topic_relation but with a 'same-topic' bucket for skills sharing topic."""
    if ta == tb: return 'same-topic'
    return topic_relation(ta, tb, child_to_parent)


def topic_topic_interactions(wide_pc, delta_all, source_skill_arr, skill_info,
                              focus_topic_ids, child_to_parent):
    """Aggregate skill-level deltas to topic-level among `focus_topic_ids`."""
    skill_to_topic = skill_info.set_index('skill')['topic_id']
    focus_ids = np.asarray(list(focus_topic_ids))

    src_topic = pd.Series(source_skill_arr).map(skill_to_topic).to_numpy()
    src_mask  = np.isin(src_topic, focus_ids)

    skill_cols = np.asarray(wide_pc.columns)
    col_topic  = pd.Series(skill_cols).map(skill_to_topic).to_numpy()
    col_mask   = np.isin(col_topic, focus_ids)

    delta_sub      = delta_all[src_mask][:, col_mask]
    sub_col_topics = col_topic[col_mask]
    src_topic_sub  = src_topic[src_mask]

    delta_topic = np.zeros((delta_sub.shape[0], len(focus_ids)), dtype=np.float32)
    for j, tid in enumerate(focus_ids):
        cols = sub_col_topics == tid
        if cols.any():
            delta_topic[:, j] = delta_sub[:, cols].mean(axis=1)

    df_t = pd.DataFrame(delta_topic, columns=focus_ids)
    df_t['source_topic'] = src_topic_sub
    gt = df_t.groupby('source_topic')

    mean_long_t = gt.mean().stack(dropna=False).rename('mean_abs_delta')
    std_long_t  = gt.std().stack(dropna=False).rename('std_abs_delta')
    long_t = pd.concat([mean_long_t, std_long_t], axis=1).reset_index()
    long_t.columns = ['source_topic', 'target_topic', 'mean_abs_delta', 'std_abs_delta']
    long_t['n_obs'] = long_t['source_topic'].map(gt.size()).astype(int)
    long_t = long_t[long_t['source_topic'] != long_t['target_topic']].copy()
    long_t = long_t.dropna(subset=['mean_abs_delta'])

    sem    = long_t['std_abs_delta'] / np.sqrt(long_t['n_obs'])
    t_crit = stats.t.ppf(0.975, df=long_t['n_obs'] - 1)
    long_t['ci_low']  = long_t['mean_abs_delta'] - t_crit * sem
    long_t['ci_high'] = long_t['mean_abs_delta'] + t_crit * sem

    long_t['relation'] = [
        topic_relation(a, b, child_to_parent)
        for a, b in zip(long_t['source_topic'], long_t['target_topic'])
    ]

    topic_name = (skill_info.groupby('topic_id')['topic_name'].first())
    long_t['source_name'] = long_t['source_topic'].map(topic_name)
    long_t['target_name'] = long_t['target_topic'].map(topic_name)
    return long_t


def topic_topic_interactions_by_outcome(
    wide_pc, delta_all, source_skill_arr, source_correct,
    skill_info, focus_topic_ids, child_to_parent,
    before_all=None, mode='absolute', eps=0.01,
    balance_by_outcome_ratio=False,
):
    """Topic-level |Δ P(correct)| split by whether the source answer was correct or wrong.

    PARTIAL answers (code 1) are excluded from both groups so the signal stays clean.

    `mode` controls how the per-event delta is scaled before aggregation:
      - 'absolute' : raw |Δ P(correct)|.
      - 'relative' : headroom-normalized:
            correct events → |Δ| / max(1 - p_current, eps)
            wrong   events → |Δ| / max(p_current,     eps)
        This rewards small absolute changes near the saturation end (high p when
        correct, low p when wrong), where mastery has little headroom left.

    `before_all` is the (n_events, n_skills) P(correct) snapshot at source rows,
    returned by `build_pairwise_deltas`. Required when mode='relative'.

    `balance_by_outcome_ratio`: if True, for each source topic A scale the two
    outcome means by how often the topic was answered each way:
        r_correct = n_correct / (n_correct + n_wrong)
        r_wrong   = n_wrong   / (n_correct + n_wrong)
        mean_abs_delta_correct *= r_correct
        mean_abs_delta_wrong   *= r_wrong
    This down-weights the rare outcome (e.g. wrong answers on an easy topic),
    so a strong directional signal needs both magnitude and frequency.

    Returns a DataFrame with columns:
        source_topic, target_topic,
        mean_abs_delta_correct, n_obs_correct,
        mean_abs_delta_wrong,   n_obs_wrong,
        strength        — (delta_correct + delta_wrong) / 2
        directionality  — (delta_correct - delta_wrong) / strength  ∈ [-1, +1]
                          +1 → B is a prerequisite of A (correct answer on A boosts P(B))
                          -1 → A is a prerequisite of B (wrong answer on A drops P(B))
                           0 → complementary / symmetric
        relation, source_name, target_name
    """
    if mode == 'relative':
        if before_all is None:
            raise ValueError("mode='relative' requires before_all from build_pairwise_deltas")
        scale = np.ones_like(delta_all, dtype=np.float32)
        cm = source_correct == 2
        wm = source_correct == 0
        scale[cm] = np.maximum(1.0 - before_all[cm], eps)
        scale[wm] = np.maximum(before_all[wm],       eps)
        delta_all = delta_all / scale
    elif mode != 'absolute':
        raise ValueError(f"unknown mode: {mode!r}")

    skill_to_topic = skill_info.set_index('skill')['topic_id']
    focus_ids = np.asarray(list(focus_topic_ids))

    src_topic = pd.Series(source_skill_arr).map(skill_to_topic).to_numpy()
    src_mask  = np.isin(src_topic, focus_ids)

    skill_cols = np.asarray(wide_pc.columns)
    col_topic  = pd.Series(skill_cols).map(skill_to_topic).to_numpy()
    col_mask   = np.isin(col_topic, focus_ids)

    delta_sub       = delta_all[src_mask][:, col_mask]
    sub_col_topics  = col_topic[col_mask]
    src_topic_sub   = src_topic[src_mask]
    src_correct_sub = source_correct[src_mask]

    # Aggregate skill columns → topic columns.
    delta_topic = np.zeros((delta_sub.shape[0], len(focus_ids)), dtype=np.float32)
    for j, tid in enumerate(focus_ids):
        cols = sub_col_topics == tid
        if cols.any():
            delta_topic[:, j] = delta_sub[:, cols].mean(axis=1)

    df_t = pd.DataFrame(delta_topic, columns=focus_ids)
    df_t['source_topic']   = src_topic_sub
    df_t['source_correct'] = src_correct_sub

    def _agg_outcome(outcome_val, suffix):
        sub_o = df_t[df_t['source_correct'] == outcome_val].drop(columns='source_correct')
        gt    = sub_o.groupby('source_topic')
        mean_l = gt.mean().stack(dropna=False).rename(f'mean_abs_delta_{suffix}')
        n_obs  = gt.size().rename(f'n_obs_{suffix}')
        result = mean_l.reset_index()
        result.columns = ['source_topic', 'target_topic', f'mean_abs_delta_{suffix}']
        result[f'n_obs_{suffix}'] = result['source_topic'].map(n_obs)
        result = result[result['source_topic'] != result['target_topic']]
        return result.dropna(subset=[f'mean_abs_delta_{suffix}'])

    correct_df = _agg_outcome(2, 'correct')
    wrong_df   = _agg_outcome(0, 'wrong')

    long_t = correct_df.merge(wrong_df, on=['source_topic', 'target_topic'], how='outer')

    if balance_by_outcome_ratio:
        total = long_t['n_obs_correct'].fillna(0) + long_t['n_obs_wrong'].fillna(0)
        r_correct = long_t['n_obs_correct'].fillna(0) / total
        r_wrong   = long_t['n_obs_wrong'].fillna(0)   / total
        long_t['mean_abs_delta_correct'] = long_t['mean_abs_delta_correct'] * r_correct
        long_t['mean_abs_delta_wrong']   = long_t['mean_abs_delta_wrong']   * r_wrong

    long_t['strength'] = (
        long_t['mean_abs_delta_correct'] + long_t['mean_abs_delta_wrong']
    ) / 2
    long_t['directionality'] = (
        (long_t['mean_abs_delta_correct'] - long_t['mean_abs_delta_wrong'])
        / long_t['strength']
    )

    long_t['relation'] = [
        topic_relation(a, b, child_to_parent)
        for a, b in zip(long_t['source_topic'], long_t['target_topic'])
    ]
    topic_name = skill_info.groupby('topic_id')['topic_name'].first()
    long_t['source_name'] = long_t['source_topic'].map(topic_name)
    long_t['target_name'] = long_t['target_topic'].map(topic_name)
    return long_t.reset_index(drop=True)


def add_reverse_directionality(outcome_df):
    """Self-merge to add reverse_directionality and asymmetry columns.

    For each (A→B) row, looks up directionality(B→A) and computes:
        reverse_directionality — directionality of the B→A pair
        asymmetry              — directionality(A→B) - directionality(B→A)

    High positive asymmetry → strongly directional prerequisite.
    Near-zero asymmetry     → complementary / symmetric relationship.
    """
    reverse = (
        outcome_df[['source_topic', 'target_topic', 'directionality']]
        .rename(columns={
            'source_topic': 'target_topic',
            'target_topic': 'source_topic',
            'directionality': 'reverse_directionality',
        })
    )
    merged = outcome_df.merge(reverse, on=['source_topic', 'target_topic'], how='left')
    merged['asymmetry'] = merged['directionality'] - merged['reverse_directionality']
    return merged


def prerequisite_score(outcome_df):
    """Antisymmetric prerequisite score f(A, B) ∈ [-1, 1].

    f(A,B) = (directionality(B→A) − directionality(A→B)) / 2
           = −asymmetry / 2

    Properties
    ----------
    f(A,B) = +1  →  A is a strong prerequisite of B
    f(A,B) = −1  →  B is a strong prerequisite of A
    f(A,B) =  0  →  complementary / symmetric
    f(A,B) = −f(B,A)          (antisymmetric by construction)
    f(A,A) =  0

    Expects the output of add_reverse_directionality() as input
    (columns 'directionality' and 'reverse_directionality' must exist).
    Adds a 'prerequisite_score' column in-place on a copy.
    """
    if 'reverse_directionality' not in outcome_df.columns:
        outcome_df = add_reverse_directionality(outcome_df)
    df = outcome_df.copy()
    df['prerequisite_score'] = (df['reverse_directionality'] - df['directionality']) / 2
    return df


def build_curriculum_graph(prereq_df, strength_percentile=50, f_percentile=50):
    """Build a curriculum DAG from prerequisite scores.

    Algorithm :
    1. Keep only edges where prerequisite_score > 0  (direction: A is prerequisite of B)
    2. Strength floor: drop edges below `strength_percentile` of remaining strength values
    3. f ranking: keep top (100 - f_percentile)% by prerequisite_score among survivors
    4. Sort by prerequisite_score descending; add each edge unless it creates a cycle

    Parameters
    ----------
    prereq_df          : output of prerequisite_score() — must have columns
                         source_topic, target_topic, prerequisite_score, strength,
                         source_name, target_name
    strength_percentile: percentile used as the strength reliability floor (default 50)
    f_percentile       : percentile used as the f quality cutoff among survivors (default 50)

    Returns
    -------
    edges_df : DataFrame of accepted edges (columns: source_topic, target_topic,
               source_name, target_name, prerequisite_score, strength, relation)
    G        : networkx.DiGraph — nodes are topic_ids, edge attributes: f, strength
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    df = prereq_df.dropna(subset=['prerequisite_score', 'strength']).copy()

    # Step 1 — keep only directed edges (f > 0 means source is prerequisite of target)
    df = df[df['prerequisite_score'] > 0]

    # Step 2 — strength floor
    strength_floor = df['strength'].quantile(strength_percentile / 100)
    df = df[df['strength'] >= strength_floor]

    # Step 3 — f quality cutoff among survivors
    f_cutoff = df['prerequisite_score'].quantile(f_percentile / 100)
    df = df[df['prerequisite_score'] >= f_cutoff]

    # Step 4 — greedy acyclic addition
    df = df.sort_values('prerequisite_score', ascending=False).reset_index(drop=True)

    all_topics = set(df['source_topic']).union(df['target_topic'])
    G = nx.DiGraph()
    G.add_nodes_from(all_topics)

    # Attach topic names as node attributes for visualisation
    name_map = (
        dict(zip(df['source_topic'], df['source_name'])) |
        dict(zip(df['target_topic'], df['target_name']))
    )
    nx.set_node_attributes(G, name_map, 'name')

    accepted = []
    for _, row in df.iterrows():
        u, v = row['source_topic'], row['target_topic']
        # Adding u→v is safe iff v cannot already reach u
        if not nx.has_path(G, v, u):
            G.add_edge(u, v, f=row['prerequisite_score'], strength=row['strength'])
            accepted.append(row)

    keep_cols = [c for c in
                 ['source_topic', 'target_topic', 'source_name', 'target_name',
                  'prerequisite_score', 'strength', 'relation']
                 if c in df.columns]
    edges_df = (pd.DataFrame(accepted)[keep_cols].reset_index(drop=True)
                if accepted else pd.DataFrame(columns=keep_cols))
    return edges_df, G


def plot_curriculum_graph(G, subject, ax=None, figsize=(14, 7)):
    """Draw the curriculum DAG with a left-to-right topological layout.

    Nodes are coloured by their topological generation (depth in the prerequisite
    chain). Edges have uniform width; colour encodes |f| (magnitude of the
    prerequisite directionality) so stronger prerequisite edges stand out.
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
    except ImportError:
        raise ImportError("networkx and matplotlib are required")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if len(G.nodes) == 0:
        ax.set_title(f'{subject} — Curriculum Graph (empty)')
        return

    # Topological generations → x-coordinate (learning order left→right)
    generations = list(nx.topological_generations(G))
    pos = {}
    for gen_idx, gen in enumerate(generations):
        gen = sorted(gen)
        for node_idx, node in enumerate(gen):
            y = node_idx - (len(gen) - 1) / 2
            pos[node] = (gen_idx, y)

    # Node colour by generation depth (light → dark blue as topics get later)
    gen_of = {node: i for i, gen in enumerate(generations) for node in gen}
    n_gens = max(gen_of.values()) + 1 if gen_of else 1
    node_colours = [cm.Blues(0.35 + 0.55 * gen_of[n] / max(n_gens - 1, 1))
                    for n in G.nodes()]

    labels = {n: G.nodes[n].get('name', str(n)) for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colours, node_size=2200, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_weight='bold')

    if G.edges():
        # Colour by |f| (magnitude of prerequisite directionality) — uniform width
        # so all edges are equally visible; darker = stronger directional signal.
        f_vals = [abs(G[u][v]['f']) for u, v in G.edges()]
        f_norm = mcolors.Normalize(vmin=min(f_vals), vmax=max(f_vals))
        edge_colours = [cm.Oranges(0.35 + 0.65 * f_norm(f)) for f in f_vals]
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color=edge_colours, width=2.0,
                               arrows=True, arrowsize=18,
                               connectionstyle='arc3,rad=0.08',
                               min_source_margin=30, min_target_margin=30)
        sm = cm.ScalarMappable(cmap=cm.Oranges, norm=f_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='|prerequisite score f|', shrink=0.6)

    isolated = list(nx.isolates(G))
    if isolated:
        iso_names = [G.nodes[n].get('name', str(n)) for n in isolated]
        ax.annotate(f"Isolated (no strong signal): {', '.join(iso_names)}",
                    xy=(0.01, 0.01), xycoords='axes fraction',
                    fontsize=7, color='gray', style='italic')

    ax.set_title(f'{subject} — Curriculum Graph\n'
                 f'({len(G.edges())} edges, {len(G.nodes())} topics, '
                 f'{len(isolated)} isolated)',
                 fontsize=11, fontweight='bold')
    ax.axis('off')


def annotate_skill_interactions(long, skill_info, child_to_parent, focus_topic_ids=None):
    """Attach topic metadata + tree-relation label to a skill-level `long` table.
    If `focus_topic_ids` is given, restrict to skills whose topic_id is in it."""
    skill_meta = skill_info.set_index('skill')[['topic_id', 'topic_name', 'difficulty']]
    if focus_topic_ids is not None:
        focus_skills = set(skill_info.loc[skill_info['topic_id'].isin(focus_topic_ids), 'skill'])
        long = long[long['source'].isin(focus_skills) & long['target'].isin(focus_skills)].copy()
    else:
        long = long.copy()
    long['source_topic'] = long['source'].map(skill_meta['topic_id'])
    long['source_diff']  = long['source'].map(skill_meta['difficulty'])
    long['source_name']  = long['source'].map(skill_meta['topic_name'])
    long['target_topic'] = long['target'].map(skill_meta['topic_id'])
    long['target_diff']  = long['target'].map(skill_meta['difficulty'])
    long['target_name']  = long['target'].map(skill_meta['topic_name'])
    long['relation'] = [
        skill_relation(a, b, child_to_parent)
        for a, b in zip(long['source_topic'], long['target_topic'])
    ]
    return long
