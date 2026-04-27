# feature engineering

import pandas as pd


EVAL_CODES = {'WRONG': 0, 'PARTIAL': 1, 'CORRECT': 2}
N_EVAL_STATES = len(EVAL_CODES)


def build_feature_matrix(transactions, documents):
    """Build the modeling DataFrame X from the (evaluated) transactions table.

    Output columns: user_id, skill_name, correct, start_time, skill_attempts, total_attempts.
    """
    df = transactions.copy()

    # 1. Attach document-level estimatedDifficulty if not already present.
    if 'estimatedDifficulty' not in df.columns:
        diff_map = documents.set_index('document_id')['estimatedDifficulty']
        df['estimatedDifficulty'] = df['document_id'].map(diff_map)

    # 2. Keep only the columns needed for modeling.
    X = df[['user_id', 'topic_id', 'evaluation', 'estimatedDifficulty', 'start_time']].copy()

    # 3. Replace missing values with the sentinel -2.
    X = X.fillna(-2)

    # 4. Build a `skill_name` key combining topic and difficulty.
    X['skill_name'] = X['topic_id'].astype(int).astype(str) + '_' + X['estimatedDifficulty'].astype(int).astype(str)

    # 5. Map evaluation strings to ordinal codes.
    X['correct'] = X['evaluation'].map(EVAL_CODES)

    # 6. Trim to modeling columns.
    X = X[['user_id', 'skill_name', 'correct', 'start_time']].copy()

    # 7. Causal counters (only past attempts) — practice signal per skill and overall.
    X['skill_attempts'] = X.groupby(['user_id', 'skill_name']).cumcount()
    X['total_attempts'] = X.groupby('user_id').cumcount()

    return X


def prepare_seq(df):
    '''
    Extract user_id sequence in preparation for DKT. The output of this function 
    feeds into the prepare_data() function. 
    '''
    # Enumerate skill id as a categorical variable 
    # (i.e. [32, 12, 32, 45] -> [0, 1, 0, 2])
    df['skill'], _   = pd.factorize(df['skill_name'], sort=True)
    df['skill_with_answer'] = df['skill'].astype('int32') * 3 + df['correct'].astype('int32')
    df['skill']    = df['skill'].astype('int32')
    df['correct'] = df['correct'].astype('int32')

    # Convert to a sequence per user_id and shift features 1 timestep
    seq = df.groupby('user_id').apply(lambda r: (r['skill_with_answer'].values[:-1], r['skill'].values[1:], r['correct'].values[1:],), include_groups=False)
    
    # Get max skill depth and max feature depth
    skill_depth = df['skill'].max() 
    features_depth = df['skill_with_answer'].max() + 1

    return seq, features_depth, skill_depth
