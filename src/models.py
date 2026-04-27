# model fitting helpers

import numpy as np
import tensorflow as tf
from sklearn import model_selection

from src.features import N_EVAL_STATES


def create_iterator(data):
    '''
    Create an iterator to split interactions in data into train and test, with the same student not appearing in two diverse folds.
    :param data:        Dataframe with student's interactions.
    :return:            An iterator.
    '''    
    # Both passing a matrix with the raw data or just an array of indexes works
    X = np.arange(len(data.index))
    # Groups of interactions are identified by the user id (we do not want the same user appearing in two folds)
    groups = data['user_id'].values 
    return model_selection.GroupShuffleSplit(n_splits=1, train_size=.8, test_size=0.2, random_state=0).split(X, groups=groups)


def prepare_data(seq, params, features_depth, skill_depth):
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_signature=(
            tf.TensorSpec(shape=None, dtype=tf.int32),   # past skill_with_answer
            tf.TensorSpec(shape=None, dtype=tf.int32),   # next skill
            tf.TensorSpec(shape=None, dtype=tf.int32),   # next eval (label)
        ),
    )

    def to_xy(feat, skill, label):
        feat_oh = tf.one_hot(feat, depth=features_depth)
        weight  = tf.ones_like(label, dtype=tf.float32)
        return {'features': feat_oh, 'next_skill': skill}, label, weight

    dataset = dataset.map(to_xy)

    dataset = dataset.padded_batch(
    batch_size=params['batch_size'],
    padded_shapes=({'features': [None, features_depth],
                    'next_skill': [None]},
                    [None], [None]),
    padding_values=({'features': 0.0, 'next_skill': 0},
                    0, 0.0),
    drop_remainder=True,
)
    return dataset.repeat(), len(seq)



def train_dkt(model, tf_train, tf_val, params):
    """Fit a DKT model on padded tf.data streams, saving the best weights.

    Expected params keys: `best_model_weights`, `epochs`, `train_size`, `val_size`, `verbose`.
    """
    # 1. Checkpoint callback — keep only the best validation weights.
    ckp_callback = tf.keras.callbacks.ModelCheckpoint(
        params['best_model_weights'],
        save_best_only=True,
        save_weights_only=True,
    )

    # 2. Fit on the training stream, validating on the held-out one.
    #    `train_size` / `val_size` are step counts (batches), so use them directly.
    history = model.fit(
        tf_train,
        epochs=params['epochs'],
        steps_per_epoch=params['train_size'],
        validation_data=tf_val,
        validation_steps=params['val_size'],
        callbacks=[ckp_callback],
        verbose=params['verbose'],
    )
    return history


class GatherSkill(tf.keras.layers.Layer):
    """Pick the next skill's 3-vec from the (B, T, n_skills, 3) cube,
    via dot product with a one-hot skill mask. Gradient is just matmul."""
    def __init__(self, nb_skills, **kwargs):
        super().__init__(**kwargs)
        self.nb_skills = int(nb_skills)
    def call(self, inputs):
        logits, skill = inputs                      # (B,T,S,3), (B,T)
        oh = tf.one_hot(skill, depth=self.nb_skills, dtype=logits.dtype)  # (B,T,S)
        return tf.einsum('btsk,bts->btk', logits, oh)                     # (B,T,3)
    def compute_mask(self, inputs, mask=None):
        return None


def create_model_lstm(nb_features, nb_skills, params):
    feat_in  = tf.keras.Input(shape=(None, nb_features), name='features')
    skill_in = tf.keras.Input(shape=(None,), dtype=tf.int32, name='next_skill')

    # x = tf.keras.layers.Masking(mask_value=0.0)(feat_in)
    x = tf.keras.layers.LSTM(
        params['recurrent_units'], return_sequences=True,
        dropout=params['dropout_rate'],
    )(feat_in)
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(int(nb_skills) * N_EVAL_STATES)
    )(x)
    logits = tf.keras.layers.Reshape((-1, int(nb_skills), N_EVAL_STATES))(x)
    out = GatherSkill(nb_skills=int(nb_skills), name='outputs')([logits, skill_in])

    model = tf.keras.models.Model(inputs=[feat_in, skill_in], outputs=out, name='DKT')
    model.compile(loss=loss_fn,
                  optimizer=params['optimizer'],
                  weighted_metrics=[
                    AUC(), 
                    RMSE(), 
                    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
])
    return model


def _flatten(y_true, y_pred, sample_weight):
    """Reshape (B, T, ...) -> (B*T, ...). sample_weight handles masking."""
    label  = tf.reshape(y_true, [-1])
    logits = tf.reshape(y_pred, [-1, N_EVAL_STATES])
    sw     = None if sample_weight is None else tf.reshape(sample_weight, [-1])
    return label, logits, sw


class AUC(tf.keras.metrics.AUC):
    """Macro one-vs-rest AUC across the 3 states."""
    def __init__(self):
        super().__init__(name='auc',
                        multi_label=True,
                        num_labels=N_EVAL_STATES)
    def update_state(self, y_true, y_pred, sample_weight=None):
        label, logits, sw = _flatten(y_true, y_pred, sample_weight)
        super().update_state(
            tf.one_hot(tf.cast(label, tf.int32), N_EVAL_STATES),
            tf.nn.softmax(logits),
            sample_weight=sw,
        )

class RMSE(tf.keras.metrics.RootMeanSquaredError):
    """Ordinal RMSE between true class index and softmax-expected value."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        probs    = tf.nn.softmax(y_pred, axis=-1)
        states   = tf.cast(tf.range(N_EVAL_STATES), probs.dtype)
        expected = tf.reduce_sum(probs * states, axis=-1)
        super().update_state(
            tf.cast(y_true, expected.dtype),
            expected,
            sample_weight=sample_weight,   # element-wise, no flatten needed
        )

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
