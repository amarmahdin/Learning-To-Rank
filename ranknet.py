import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class RankNet(Model):
    def __init__(self, input_dim=4, hidden_layers=[128, 64, 32], dropout_rate=0.1, l2_reg=0.001):
        super().__init__()
        self.scoring = tf.keras.Sequential()
        # HAPUS batch normalization di awal!
        for units in hidden_layers:
            self.scoring.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
            self.scoring.add(layers.Dropout(dropout_rate))
        self.scoring.add(layers.Dense(1, activation=None))  # Output: skor (real number)

    def call(self, inputs):
        x_i = inputs[:, 0, :]
        x_j = inputs[:, 1, :]
        s_i = self.scoring(x_i)
        s_j = self.scoring(x_j)
        s_i = tf.squeeze(s_i, axis=-1)
        s_j = tf.squeeze(s_j, axis=-1)
        return tf.stack([s_i, s_j], axis=1)

    def pairwise_loss(self, y_true, y_pred):
        s_i = y_pred[:, 0]
        s_j = y_pred[:, 1]
        P_ij = tf.sigmoid(s_i - s_j)
        epsilon = 1e-7
        P_ij = tf.clip_by_value(P_ij, epsilon, 1 - epsilon)
        loss = - (y_true * tf.math.log(P_ij) + (1 - y_true) * tf.math.log(1 - P_ij))
        tf.debugging.assert_all_finite(loss, "Loss NaN/Inf ditemukan dalam batch ini!")
        return tf.reduce_mean(loss)