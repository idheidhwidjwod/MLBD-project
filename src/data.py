# loading and cleaning data

import json

import pandas as pd
from IPython.display import display


def load_all_data():
    """Load every raw CSV from ./data into a dict of DataFrames."""
    # 1. Read each file (gzipped CSVs are decoded transparently).
    documents = pd.read_csv('data/documents.csv.gz')
    # events = pd.read_csv('data/events.csv.gz')
    # feedback = pd.read_csv('data/feedback.csv.gz')
    topic_trees = pd.read_csv('data/topic_trees.csv.gz')
    topics_translated = pd.read_csv('data/topics_translated.csv')
    transactions = pd.read_csv('data/transactions.csv.gz')
    users = pd.read_csv('data/users.csv.gz')

    # 2. Collect under human-readable keys so callers can pick by name.
    dfs = {
        'documents': documents,
        # 'events': events,
        # 'feedback': feedback,
        'topic_trees': topic_trees,
        'topics_translated': topics_translated,
        'transactions': transactions,
        'users': users
    }
    return dfs


def get_latest_documents(documents):
    """Reduce documents to one row per document with parsed metadata columns."""
    # 1. Sort by version, keep the last row per document_id (= latest version).
    latest = documents.sort_values('version').drop_duplicates('document_id', keep='last')
    # 2. Drop bookkeeping columns we don't need downstream.
    latest = latest.drop(columns=['author_id', 'status', 'version_comment'])
    # 3. Parse the JSON `content` string for each row.
    parsed = latest['content'].apply(lambda c: json.loads(c) if isinstance(c, str) else (c or {}))
    # 4. Pull the two fields we care about out of the nested `metaData` object.
    metadata = parsed.apply(lambda d: d.get('metaData') or {})
    latest['estimatedDuration'] = metadata.apply(lambda m: m.get('estimatedDuration'))
    latest['estimatedDifficulty'] = metadata.apply(lambda m: m.get('estimatedDifficulty'))
    return latest


def summarize_documents(latest):
    """Print coverage stats for the latest-documents table and return the dropouts."""
    # 1. Count distinct documents.
    total = latest['document_id'].nunique()
    # 2. Flag rows where both metadata fields are present.
    has_meta = latest['estimatedDuration'].notna() & latest['estimatedDifficulty'].notna()
    kept = int(has_meta.sum())
    filtered_out = latest.loc[~has_meta, 'document_id'].tolist()

    # 3. Compute percentages (guarded against empty input).
    kept_pct = kept / total * 100 if total else 0
    out_pct = len(filtered_out) / total * 100 if total else 0

    # 4. Print a short report and return the list of dropouts.
    n_topics = latest['topic_id'].nunique()
    print(f"Total distinct documents: {total}")
    print(f"Documents with estimatedDuration and estimatedDifficulty: {kept} ({kept_pct:.1f}%)")
    print(f"Filtered out: {len(filtered_out)} ({out_pct:.1f}%)")
    print(f"Distinct topics: {n_topics}")
    return filtered_out


def build_topic_lookups(topics, topic_trees):
    """Build convenient dict look-ups for the topic hierarchy."""
    # 1. Direct id → attribute maps from the topics table.
    id_to_name = dict(zip(topics['id'], topics['name']))
    id_to_math = dict(zip(topics['id'], topics['math']))

    # 2. Drop roots (rows with NaN parent_id) before building tree maps.
    tree = topic_trees.dropna(subset=['parent_id'])
    # 3. Walk-up map: each child points to its single parent.
    child_to_parent = (
        tree.set_index('child_id')['parent_id'].astype(int).to_dict()
    )
    # 4. Walk-down map: each parent points to the list of its children.
    parent_to_children = (
        tree.groupby('parent_id')['child_id'].apply(list).to_dict()
    )

    return {
        'id_to_name': id_to_name,
        'id_to_math': id_to_math,
        'child_to_parent': child_to_parent,
        'parent_to_children': parent_to_children,
    }


def get_depth(node_id, child_to_parent, max_iter=20):
    """Return the number of edges from `node_id` up to the root."""
    depth, current = 0, node_id
    # Walk parent pointers; max_iter caps the loop in case of accidental cycles.
    for _ in range(max_iter):
        parent = child_to_parent.get(current)
        if parent is None:
            break
        depth += 1
        current = parent
    return depth


def add_topic_depth(topics, child_to_parent, show=True):
    """Return a copy of `topics` with a new `depth` column."""
    # 1. Copy so the caller's DataFrame is left untouched.
    topics = topics.copy()
    # 2. Compute depth for each topic id.
    topics['depth'] = topics['id'].apply(lambda x: get_depth(x, child_to_parent))
    # 3. Optionally print the distribution of topics per depth.
    if show:
        print('Depth distribution across all topics:')
        print(topics['depth'].value_counts().sort_index().to_frame('n_topics'))
    return topics


def docs_per_topic(documents):
    """Count unique documents per topic, sorted descending."""
    return (
        documents.groupby('topic_id')['document_id']
        .nunique()
        .sort_values(ascending=False)
        .to_frame('n_documents')
    )


def docs_per_depth(documents, topics):
    """Count unique documents grouped by the depth of their topic."""
    # 1. Attach each document's topic depth via a left join.
    merged = documents.merge(
        topics[['id', 'depth']], left_on='topic_id', right_on='id', how='left'
    )
    # 2. Count distinct document_ids per depth level.
    return (
        merged.groupby('depth')['document_id']
        .nunique()
        .sort_index()
        .to_frame('n_documents')
    )


def reparent_topics(topic_trees, child_ids, new_parent_id):
    """Move a set of topics under a new parent in the topic-tree table."""
    # 1. Drop existing rows for those children (detach the subtree).
    trees = topic_trees[~topic_trees['child_id'].isin(child_ids)].copy()
    # 2. Build fresh rows linking each child to the new parent.
    #    Synthetic topic_ids beyond the current max keep primary keys unique.
    max_topic_id = int(topic_trees['topic_id'].max())
    new_rows = pd.DataFrame([
        {
            'topic_id': max_topic_id + i + 1,
            'parent_id': float(new_parent_id),
            'child_id': cid,
            'sibling_rank': 0,
            'displayed_on_dashboard': 0,
        }
        for i, cid in enumerate(child_ids)
    ])
    # 3. Concatenate and return the rebuilt table.
    return pd.concat([trees, new_rows], ignore_index=True)


def _non_empty_topic_ids(topic_trees, documents):
    """Set of topic ids whose subtree contains at least one document."""
    # 1. Build child → parent map so we can walk upwards from each active topic.
    tree = topic_trees.dropna(subset=['parent_id'])
    child_to_parent = tree.set_index('child_id')['parent_id'].astype(int).to_dict()

    # 2. Start with the set of topics that actually host at least one document.
    active = set(documents['topic_id'].dropna().unique())
    keep = set(active)

    # 3. Add every ancestor of an active topic (they host active descendants).
    for node in active:
        current = node
        while current in child_to_parent:
            current = child_to_parent[current]
            if current in keep:
                break
            keep.add(current)
    return keep


def prune_empty_topics(topic_trees, documents):
    """Remove topics that have no document and whose whole subtree has none either."""
    keep = _non_empty_topic_ids(topic_trees, documents)
    # Drop rows whose child_id is not in the keep set (those subtrees are empty).
    return topic_trees[topic_trees['child_id'].isin(keep)].reset_index(drop=True)


def _subtree_ids(topic_trees, root_ids):
    """Set of topic ids in the subtree rooted at each id in `root_ids` (roots included)."""
    # 1. Build parent → [children] map from the tree edges.
    tree = topic_trees.dropna(subset=['parent_id'])
    parent_to_children = tree.groupby('parent_id')['child_id'].apply(list).to_dict()

    # 2. BFS downward from each root, collecting every visited id.
    collected = set()
    stack = list(root_ids)
    while stack:
        node = stack.pop()
        if node in collected:
            continue
        collected.add(node)
        stack.extend(parent_to_children.get(node, []))
    return collected


def drop_topic_subtrees(topic_trees, topics, root_ids):
    """Remove `root_ids` and all their descendants from both tables."""
    # 1. Compute the full set of topics to remove.
    to_drop = _subtree_ids(topic_trees, root_ids)
    # 2. Drop rows from each table.
    pruned_trees = topic_trees[~topic_trees['child_id'].isin(to_drop)].reset_index(drop=True)
    pruned_topics = topics[~topics['id'].isin(to_drop)].reset_index(drop=True)
    return pruned_trees, pruned_topics


def prune_empty_topics_translated(topics, topic_trees, documents):
    """Remove rows from `topics_translated` whose subtree contains no documents."""
    keep = _non_empty_topic_ids(topic_trees, documents)
    return topics[topics['id'].isin(keep)].reset_index(drop=True)


def split_by_subject(evaluated, topics):
    """Split evaluated transactions into (math_df, german_df) using `topics.math`."""
    # 1. Build the id sets for each subject from the topics table.
    math_ids = set(topics.loc[topics['math'] == 1, 'id'])
    german_ids = set(topics.loc[topics['math'] == 0, 'id'])
    # 2. Filter the evaluated rows by their topic's subject.
    math_df = evaluated[evaluated['topic_id'].isin(math_ids)].reset_index(drop=True)
    german_df = evaluated[evaluated['topic_id'].isin(german_ids)].reset_index(drop=True)
    return math_df, german_df


def summarize_transactions(transactions, topics, documents):
    """Report on the transactions table and return evaluated rows with context stats."""
    # 1. Total number of transactions.
    total = len(transactions)
    print(f"Total transactions: {total}")

    # 2. Keep only rows where evaluation is set.
    evaluated = transactions[transactions['evaluation'].notna()].copy()
    ev_pct = len(evaluated) / total * 100 if total else 0
    print(f"Evaluated transactions: {len(evaluated)} ({ev_pct:.1f}%)")

    # 3a. Flag rows whose topic_id is not in the (pruned) topics table.
    known_topics = set(topics['id'])
    missing_topic = ~evaluated['topic_id'].isin(known_topics)
    n_missing_topic = int(missing_topic.sum())
    mt_pct = n_missing_topic / len(evaluated) * 100 if len(evaluated) else 0
    print(f"Evaluated with unknown topic_id: {n_missing_topic} ({mt_pct:.1f}%)")

    # 3b. Flag rows whose document has no estimatedDifficulty.
    diff_map = documents.set_index('document_id')['estimatedDifficulty']
    evaluated['estimatedDifficulty'] = evaluated['document_id'].map(diff_map)
    missing_diff = evaluated['estimatedDifficulty'].isna()
    n_missing_diff = int(missing_diff.sum())
    md_pct = n_missing_diff / len(evaluated) * 100 if len(evaluated) else 0
    print(f"Evaluated with no estimatedDifficulty: {n_missing_diff} ({md_pct:.1f}%)")

    return evaluated


def topics_with_docs_per_depth(documents, topics):
    """Count, per depth level, how many topics have at least one document."""
    # 1. Collect the set of topic ids that actually appear in documents.
    active = documents['topic_id'].unique()
    # 2. Filter topics down to those active ids.
    active_topics = topics[topics['id'].isin(active)]
    # 3. Count distinct topic ids per depth.
    return (
        active_topics.groupby('depth')['id']
        .nunique()
        .sort_index()
        .to_frame('n_topics_with_docs')
    )



if __name__ == "__main__":
    dfs = load_all_data()
    display(dfs)
