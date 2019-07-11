import tensorflow as tf
from attention import AttentionYatianColing2016
from config import AttentionYatianColing2016Config


def initialize_session():
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init_op)
    return sess


def feed(attention):
    assert(isinstance(attention, AttentionYatianColing2016))
    return {
        attention.Input[attention.I_x]: feed_dict['x'],
        attention.Input[attention.I_entities]: feed_dict['entities']
    }


feed_dict = {
    "x": [
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

term_embedding_tensor = tf.constant(
    value=term_embedding,
    dtype=tf.float32,
    shape=[len(term_embedding), len(term_embedding[0])])

attention = AttentionYatianColing2016(cfg=AttentionYatianColing2016Config(),
                                      batch_size=len(feed_dict['x']),
                                      terms_per_context=len(feed_dict['x'][0]),
                                      term_embedding_size=len(term_embedding))

attention.init_input()
attention.init_hidden()
e_sum, weights = attention.init_body(term_embedding=term_embedding_tensor)

with initialize_session() as sess:
    r_sum, r_weights = sess.run([e_sum, weights], feed_dict=feed(attention))

    print "att_sum:"
    print r_sum[0].shape
    print r_sum[0]

    print "att_weights:"
    print r_weights[0].shape
    print r_weights[0]
