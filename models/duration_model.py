import tensorflow as tf


class DurationModel(object):

    def __init__(self,
                 X, X_seq_len, input_vocab_size,
                 Y, model_parameters,
                 train_parameters, reuse=None
                 ):

        self.global_step = tf.Variable(
            0, name="global_step", trainable=False
        )

        with tf.variable_scope("encoder", reuse=reuse):
            embedding = tf.get_variable(
                'embedding',
                shape=(input_vocab_size, model_parameters["embedding_size"]),
                dtype=tf.float32
            )
            dense_output = tf.nn.embedding_lookup(embedding, X)

            for i in xrange(model_parameters["num_dense_layers"]):
                with tf.variable_scope("dense_layer_" + str(i), reuse=reuse):
                    dense_output = tf.layers.dense(
                        dense_output, model_parameters["dense_layers_units"]
                    )

            cells_fw = [
                tf.nn.rnn_cell.GRUCell(
                    model_parameters["num_bidirectional_units"]
                ) for i in xrange(model_parameters["num_bidirectional_layers"])
            ]
            cells_bw = [
                tf.nn.rnn_cell.GRUCell(
                    model_parameters["num_bidirectional_units"]
                ) for i in xrange(model_parameters["num_bidirectional_layers"])
            ]

            bidirectional, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, dense_output,
                sequence_length=X_seq_len, dtype=tf.float32
            )

            dropout_output = tf.layers.dropout(
                bidirectional, model_parameters["dropout_prob"]
            )

            self.logits = tf.layers.dense(
                dense_output, model_parameters["buckets"]
            )

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, Y, X_seq_len
        )
        self.loss = tf.reduce_mean(-log_likelihood)
        tf.summary.scalar('loss', self.loss)

        self.train_op = self.__train_op(
            self.loss, self.global_step, train_parameters
        )

        self.summary = tf.summary.merge_all()

    def __train_op(self, loss, global_step, train_parameters):
        learning_rate = tf.train.exponential_decay(
            train_parameters["lr"], global_step,
            train_parameters["decay_steps"],
            train_parameters["decay_rate"]
        )
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients, variables = zip(*opt.compute_gradients(loss))
        train_op = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
        return train_op
