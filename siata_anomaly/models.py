"""Model builders for temperature anomaly detection."""

import tensorflow as tf


def weighted_binary_crossentropy(pos_weight):
    """Binary crossentropy that penalizes false negatives more.

    Args:
        pos_weight: Weight multiplier for the positive (anomaly) class.
                    Typically n_negative / n_positive.

    Returns:
        Keras loss function.
    """
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1.0 - y_true)
        return tf.reduce_mean(weight * bce)
    loss.__name__ = 'weighted_binary_crossentropy'
    return loss


def build_mlp(window_size, n_features):
    """MLP baseline model.

    Flattens the time window and passes it through dense layers.

    Args:
        window_size: Number of timesteps in each window.
        n_features: Number of input features.

    Returns:
        Uncompiled tf.keras.Model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, n_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ], name='mlp_baseline')
    return model


def build_cnn_backbone(window_size, n_features):
    """1D-CNN feature extractor with one residual block.

    Architecture:
        Conv1D(64) → BatchNorm → [ResidualBlock(128) with skip] → GAP

    Args:
        window_size: Number of timesteps in each window.
        n_features: Number of input features.

    Returns:
        Uncompiled tf.keras.Model that outputs a feature vector.
    """
    inputs = tf.keras.Input(shape=(window_size, n_features), name='input')

    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # Residual block
    shortcut = tf.keras.layers.Conv1D(128, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])

    features = tf.keras.layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inputs, features, name='cnn_backbone')


def attach_head(backbone, trainable=True):
    """Attach a classification head to a backbone.

    This mirrors the taller_3 pattern:
        backbone.trainable = False  →  only head trains
        backbone.trainable = True   →  full fine-tuning

    Args:
        backbone: Feature extractor model (from build_cnn_backbone).
        trainable: Whether to allow backbone weights to update.

    Returns:
        Uncompiled tf.keras.Model.
    """
    backbone.trainable = trainable
    inputs = tf.keras.Input(shape=backbone.input_shape[1:])
    x = backbone(inputs, training=trainable)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, output,
                          name='cnn_frozen' if not trainable else 'cnn_finetuned')
