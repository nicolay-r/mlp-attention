import tensorflow as tf
from attention import MultiLayerPerceptronAttention
from config import MultiLayerPerceptronAttentionConfig


def init_session():
    init_op = tf.compat.v1.global_variables_initializer()
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    sess.run(init_op)
    return sess


def feed(attention):
    assert(isinstance(attention, MultiLayerPerceptronAttention))
    return {
        attention.Input[attention.I_x]: feed_dict['x'],
        attention.Input[attention.I_pos]: feed_dict['pos'],
        attention.Input[attention.I_dist_obj]: feed_dict['dist_obj'],
        attention.Input[attention.I_dist_subj]: feed_dict['dist_subj'],
        attention.Input[attention.I_entities]: feed_dict['entities']
    }


if __name__ == "__main__":

    feed_dict = {
        "x": [
            [1, 0, 1, 4, 2],
            [1, 2, 0, 3, 0],
            [1, 2, 1, 3, 3]
        ],

        "dist_subj": [
            [1, 0, 1, 4, 2],
            [1, 2, 0, 3, 0],
            [1, 2, 1, 3, 3]
        ],

        "dist_obj": [
            [1, 0, 1, 4, 2],
            [1, 2, 0, 3, 0],
            [1, 2, 1, 3, 3]
        ],

        "pos": [
            [1, 0, 1, 4, 2],
            [1, 2, 0, 3, 0],
            [1, 2, 1, 3, 3]
        ],

        "entities": [
            [0, 4],
            [3, 1],
            [3, 1]
        ],
    }

    term_embedding = [
            [1.0, 5.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 2.0, 2.0, 1.0],
            [3.0, 3.0, 1.0, 3.0, 3.0],
            [4.0, 2.0, 4.0, 2.0, 4.0],
            [5.0, 6.0, 1.0, 5.0, 0.0]
        ]

    pos_embedding = term_embedding
    dist_embedding = term_embedding

    term_embedding_tensor = tf.constant(
        value=term_embedding,
        dtype=tf.float32,
        shape=[len(term_embedding), len(term_embedding[0])])

    pos_embedding_tensor = tf.constant(
        value=pos_embedding,
        dtype=tf.float32,
        shape=[len(pos_embedding), len(pos_embedding[0])])

    dist_embedding_tensor = tf.constant(
        value=dist_embedding,
        dtype=tf.float32,
        shape=[len(dist_embedding), len(dist_embedding[0])])

    attention = MultiLayerPerceptronAttention(
        cfg=MultiLayerPerceptronAttentionConfig(),
        batch_size=len(feed_dict['x']),
        terms_per_context=len(feed_dict['x'][0]),
        term_embedding_size=len(term_embedding),
        pos_embedding_size=len(pos_embedding),
        dist_embedding_size=len(dist_embedding))

    attention.init_input()
    attention.init_hidden()
    e_sum, weights = attention.init_body(
        term_embedding=term_embedding_tensor,
        pos_embedding=pos_embedding_tensor,
        dist_embedding=dist_embedding_tensor)

    with init_session() as sess:
        r_sum, r_weights = sess.run([e_sum, weights], feed_dict=feed(attention))

        print("att_sum:")
        print(r_sum[0].shape)
        print(r_sum[0])

        print("att_weights:")
        print(r_weights[0].shape)
        print(r_weights[0])
