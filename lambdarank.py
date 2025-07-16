import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def dcg(scores):
    """Compute DCG for a list of gains (assume ideal gains sorted descending)."""
    discounts = tf.math.log(tf.range(2, tf.shape(scores)[-1] + 2, dtype=tf.float32))
    return tf.reduce_sum((tf.cast(scores, tf.float32) / discounts), axis=-1)

def compute_delta_ndcg(y_true, y_pred):
    # y_true: (batch, 2). y_pred: (batch, 2)
    # We only have pairs, so delta NDCG is nonzero if swapping the pair changes ranking (i.e., their relevansi berbeda)
    # If relevansi sama, delta NDCG = 0
    gain = lambda y: tf.pow(2.0, tf.cast(y, tf.float32)) - 1.0
    # Hitung dcg sebelum dan sesudah swap
    g1 = gain(y_true[:, 0])
    g2 = gain(y_true[:, 1])

    # Ideal DCG: urutkan terbesar dulu
    ideal_gains = tf.sort(tf.stack([g1, g2], axis=1), axis=-1, direction="DESCENDING")
    idcg = dcg(ideal_gains)

    # DCG prediksi: urutan sesuai prediksi (skor lebih besar di depan)
    bigger_first = tf.cast(y_pred[:, 0] >= y_pred[:, 1], tf.float32)
    gains_pred = tf.stack([
        bigger_first * g1 + (1 - bigger_first) * g2,
        bigger_first * g2 + (1 - bigger_first) * g1
    ], axis=1)
    dcg_pred = dcg(gains_pred)

    # DCG jika swap urutan
    gains_swapped = tf.stack([
        (1 - bigger_first) * g1 + bigger_first * g2,
        (1 - bigger_first) * g2 + bigger_first * g1
    ], axis=1)
    dcg_swapped = dcg(gains_swapped)

    # delta NDCG: (|dcg_pred - dcg_swapped|) / idcg
    delta_ndcg = tf.abs(dcg_pred - dcg_swapped) / (idcg + 1e-8)
    return delta_ndcg

class LambdaRank(Model):
    def __init__(self, input_dim=4, hidden_layers=[128, 64], dropout_rate=0.1, l2_reg=0.001):
        super().__init__()
        self.scoring = tf.keras.Sequential()
        for units in hidden_layers:
            self.scoring.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
            self.scoring.add(layers.Dropout(dropout_rate))
        self.scoring.add(layers.Dense(1, activation=None))

    def call(self, inputs):
        x_i = inputs[:, 0, :]
        x_j = inputs[:, 1, :]
        s_i = self.scoring(x_i)
        s_j = self.scoring(x_j)
        s_i = tf.squeeze(s_i, axis=-1)
        s_j = tf.squeeze(s_j, axis=-1)
        return tf.stack([s_i, s_j], axis=1)

    def lambda_loss(self, y_true, y_pred):
        # y_true: (batch, 2) atau (batch,) (label ranking: 1 jika i lebih bagus dari j, 0 sebaliknya)
        # y_pred: (batch, 2)
        s_i = y_pred[:, 0]
        s_j = y_pred[:, 1]
        diff = s_i - s_j
        P_ij = tf.sigmoid(diff)
        epsilon = 1e-7
        P_ij = tf.clip_by_value(P_ij, epsilon, 1 - epsilon)
        # Untuk delta NDCG, kita butuh label relevansi asli (bukan label pairwise 0/1)
        y_true_pair = tf.reshape(y_true, [-1])  # label 1/0
        # Simulasikan label relevansi: 1 jika i lebih bagus dari j, 0 jika sama atau sebaliknya
        rel_i = tf.cast(y_true_pair, tf.float32)
        rel_j = 1 - rel_i
        # Buat tensor relevansi per pair (batch, 2)
        rels = tf.stack([rel_i, rel_j], axis=1)
        # Delta NDCG dihitung antar dua instance/pair
        delta_ndcg = compute_delta_ndcg(rels, y_pred)
        lambda_weight = delta_ndcg  # inilah kunci LambdaRank

        loss = lambda_weight * (- (y_true_pair * tf.math.log(P_ij) + (1 - y_true_pair) * tf.math.log(1 - P_ij)))
        tf.debugging.assert_all_finite(loss, "Loss NaN/Inf ditemukan dalam batch LambdaRank ini!")
        return tf.reduce_mean(loss)