import tensorflow as tf


def get_k_layer_logits(g, W, b, dropout_keep_prob=None, activations=None):
    assert(isinstance(W, list))
    assert(isinstance(b, list))
    assert(isinstance(activations, list))
    assert(len(W) == len(b) == len(activations) - 1)

    def __activate(tensor, activation):
        return activation(tensor) if activation is not None else tensor

    if activations is None:
        activations = [None] * (len(W) + 1)

    r = g

    for i in range(len(W)):
        r = __activate(r, activations[i])
        r = tf.matmul(r, W[i]) + b[i]

        if dropout_keep_prob is None:
            continue

        r = tf.nn.dropout(r, dropout_keep_prob)

    return __activate(r, activations[-1])
