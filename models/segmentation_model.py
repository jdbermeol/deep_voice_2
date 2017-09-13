import tensorflow as tf


class SegmentationModel(object):

    def __init__(self, output_vocab_size, num_speakers, model_parameters):
        self.output_vocab_size = output_vocab_size
        self.model_parameters = model_parameters
        self.num_speakers = num_speakers

    def build_train_operations(self,
                 frequencies, frequencies_seq_len,
                 speaker_ids,
                 phonemes,
                 train_parameters, reuse=None
                 ):

        encoder_output = self.__build_encoder(
            frequencies, frequencies_seq_len,
            speaker_ids,
            train_parameters["dropout_prob"],
            reuse
        )

        loss = tf.reduce_mean(tf.nn.ctc_loss(
            phonemes,
            tf.transpose(encoder_output, perm=[1, 0, 2]),
            frequencies_seq_len
        ))
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(
            0, name="global_step", trainable=False
        )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = tf.train.exponential_decay(
                train_parameters["lr"], global_step,
                train_parameters["decay_steps"],
                train_parameters["decay_rate"]
            )
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*opt.compute_gradients(loss))
            train_op = opt.apply_gradients(zip(gradients, variables), global_step=global_step)
        summary = tf.summary.merge_all()

        return train_op, loss, global_step, summary

    def build_greedy_predictor(self,
                 frequencies, frequencies_seq_len,
                 speaker_ids,
                 reuse=None
                 ):

        encoder_output = self.__build_encoder(
            frequencies, frequencies_seq_len,
            speaker_ids,
            0.0, reuse
        )

        with tf.variable_scope("decoder", reuse=reuse):
            return tf.nn.ctc_greedy_decoder(
                tf.transpose(encoder_output, perm=[1, 0, 2]),
                frequencies_seq_len
            )

    def build_beam_search_predictor(self,
                 frequencies, frequencies_seq_len,
                 speaker_ids, num_beams,
                 reuse=None
                 ):

        encoder_output = self.__build_encoder(
            frequencies, frequencies_seq_len,
            speaker_ids,
            0.0, reuse
        )

        with tf.variable_scope("decoder", reuse=reuse):
            return tf.nn.ctc_beam_search_decoder(
                tf.transpose(encoder_output, perm=[1, 0, 2]),
                frequencies_seq_len,
                beam_width=num_beams,
            )

    def __build_encoder(self,
                        frequencies, frequencies_seq_len,
                        speaker_ids,
                        dropout_prob , reuse):
        is_training = dropout_prob > 0.0

        with tf.variable_scope("speaker_embedding", reuse=reuse):
            speaker_embedding = tf.get_variable(
                'speaker_embedding',
                shape=(self.num_speakers, self.model_parameters["speaker_embedding_size"]),
                dtype=tf.float32
            )
            speaker_embedding_output = tf.nn.embedding_lookup(speaker_embedding, speaker_ids)
        with tf.variable_scope("encoder", reuse=reuse):

            speaker_embedding_projection = tf.layers.dense(
                speaker_embedding_output, self.model_parameters["conv_num_filters"],
                tf.nn.softsign
            )

            speaker_embedding_projection = tf.expand_dims(tf.expand_dims(
                speaker_embedding_projection,
                1), 1)

            speaker_embedding_projection = tf.tile(
                speaker_embedding_projection, [1, tf.shape(frequencies)[1], tf.shape(frequencies)[2], 1]
            )

            with tf.variable_scope("conv_layers"):
                with tf.variable_scope("conv_layer_0"):
                    output = tf.layers.conv2d(
                        tf.expand_dims(frequencies, -1),
                        self.model_parameters["conv_num_filters"],
                        self.model_parameters["conv_kernel_size"],
                        padding="same"
                    )
                for i in xrange(1, self.model_parameters["num_conv_layers"]):
                    with tf.variable_scope("conv_layer_" + str(i)):
                        output = tf.nn.relu(
                            tf.contrib.layers.batch_norm(tf.layers.conv2d(
                                output,
                                self.model_parameters["conv_num_filters"],
                                self.model_parameters["conv_kernel_size"],
                                padding='same', use_bias=False
                            ), scope='bn', is_training=is_training) * speaker_embedding_projection + output
                        )

            dropout_output = tf.layers.dropout(
                output, dropout_prob
            )

            with tf.variable_scope("bidirectional_layers"):
                cells_fw = [
                    tf.nn.rnn_cell.GRUCell(self.model_parameters["num_bidirectional_units"])
                    for i in xrange(self.model_parameters["num_bidirectional_layers"])
                ]
                cells_bw = [
                    tf.nn.rnn_cell.GRUCell(self.model_parameters["num_bidirectional_units"])
                    for i in xrange(self.model_parameters["num_bidirectional_layers"])
                ]

                speaker_embedding_projection = tf.layers.dense(
                    speaker_embedding_output, self.model_parameters["num_bidirectional_units"],
                    tf.nn.softsign
                )

                bidirectional_output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw, cells_bw, tf.concat(tf.unstack(dropout_output, axis=3), 2),
                    sequence_length=frequencies_seq_len, dtype=tf.float32,
                    initial_states_fw=[speaker_embedding_projection] * self.model_parameters["num_bidirectional_layers"],
                    initial_states_bw=[speaker_embedding_projection] * self.model_parameters["num_bidirectional_layers"]
                )

            dropout_output = tf.layers.dropout(
                bidirectional_output, dropout_prob
            )

            with tf.variable_scope("output_projection"):
                return tf.layers.dense(
                    dropout_output, self.output_vocab_size
                )
