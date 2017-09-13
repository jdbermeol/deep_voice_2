import tensorflow as tf


class FrequencyModel(object):

    def __init__(self,
                 input_vocab_size,
                 num_speakers,
                 model_parameters
                 ):
        self.input_vocab_size = input_vocab_size
        self.num_speakers = num_speakers
        self.model_parameters = model_parameters

    def __build_encoder(self,
                        phonemes, phonemes_seq_len,
                        speaker_ids, reuse
                       ):

        with tf.variable_scope("speaker_embedding", reuse=reuse):
            speaker_embedding = tf.get_variable(
                'speaker_embedding',
                shape=(self.num_speakers, self.model_parameters["speaker_embedding_size"]),
                dtype=tf.float32
            )
            speaker_embedding_output = tf.nn.embedding_lookup(speaker_embedding, speaker_ids)
        with tf.variable_scope("bidirectional_layers", reuse=reuse):
            phonemes_embedding = tf.get_variable(
                "phonemes_embedding",
                shape=(self.input_vocab_size, self.model_parameters["phonemes_embedding_size"]),
                dtype=tf.float32
            )
            phonemes_output = tf.nn.embedding_lookup(phonemes_embedding, phonemes)

            cells_fw = [
                tf.nn.rnn_cell.GRUCell(
                    self.model_parameters["num_bidirectional_units"]
                ) for i in xrange(self.model_parameters["num_bidirectional_layers"])
            ]
            cells_bw = [
                tf.nn.rnn_cell.GRUCell(
                    self.model_parameters["num_bidirectional_units"]
                ) for i in xrange(self.model_parameters["num_bidirectional_layers"])
            ]

            speaker_embedding_projection = tf.layers.dense(
                speaker_embedding_output, self.model_parameters["num_bidirectional_units"]
            )

            bidirectional, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, phonemes_output,
                sequence_length=phonemes_seq_len, dtype=tf.float32,
                initial_states_fw=[speaker_embedding_projection] * self.model_parameters["num_bidirectional_layers"],
                initial_states_bw=[speaker_embedding_projection] * self.model_parameters["num_bidirectional_layers"]
            )
        with tf.variable_scope("voiced_probability_model", reuse=reuse):
            voiced = tf.squeeze(tf.layers.dense(bidirectional, 1, tf.sigmoid))
        with tf.variable_scope("w", reuse=reuse):
            w = tf.squeeze(tf.layers.dense(bidirectional, 1, tf.sigmoid))
        with tf.variable_scope("f_conv", reuse=reuse):
            output = bidirectional
            for i, width in enumerate(self.model_parameters["conv_widths"]):
                with tf.variable_scope("conv_layer_" + str(i)):
                    output = tf.layers.conv1d(output, 1, width, padding="same")
            f_conv = tf.squeeze(output)
        with tf.variable_scope("f_gru", reuse=reuse):
            f_gru, _, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(self.model_parameters["output_dimension"]),
                tf.nn.rnn_cell.GRUCell(self.model_parameters["output_dimension"]),
                bidirectional,
                sequence_length=phonemes_seq_len, dtype=tf.float32,
            )
            f_gru = tf.squeeze(tf.layers.dense(f_gru, 1))

        f = w * f_gru + (1 - w) + f_conv

        with tf.variable_scope("f_zero", reuse=reuse):
            left_side = tf.squeeze(tf.layers.dense(
                speaker_embedding_output, 1,
                tf.nn.softsign
            )) + 1
            right_side = tf.squeeze(tf.layers.dense(
                speaker_embedding_output, 1,
                tf.nn.softsign
            )) + 1
            mu = tf.get_variable("mu", shape=(1), dtype=tf.float32)
            sigma = tf.get_variable("sigma", shape=(1), dtype=tf.float32)
            f_zero = mu * left_side + sigma * right_side * f
        return voiced, f_zero
