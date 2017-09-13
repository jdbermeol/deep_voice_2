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
            voiced = tf.layers.dense(bidirectional, 2, tf.sigmoid)
        with tf.variable_scope("w", reuse=reuse):
            w = tf.layers.dense(bidirectional, 1, tf.sigmoid)
        with tf.variable_scope("f_conv", reuse=reuse):
            output = bidirectional
            for i, width in enumerate(self.model_parameters["conv_widths"]):
                with tf.variable_scope("conv_layer_" + str(i)):
                    output = tf.layers.conv1d(output, 1, width, padding="same")
            f_conv = output
        with tf.variable_scope("f_gru", reuse=reuse):
            f_gru, _ = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(self.model_parameters["output_dimension"]),
                tf.nn.rnn_cell.GRUCell(self.model_parameters["output_dimension"]),
                bidirectional,
                sequence_length=phonemes_seq_len, dtype=tf.float32,
            )
            f_gru = tf.layers.dense(tf.concat(f_gru, 2), 1)

        f = tf.squeeze(w * f_gru + (1 - w) + f_conv)

        with tf.variable_scope("f_zero", reuse=reuse):
            left_side = tf.layers.dense(
                speaker_embedding_output, 1,
                tf.nn.softsign
            ) + 1
            right_side = tf.layers.dense(
                speaker_embedding_output, 1,
                tf.nn.softsign
            ) + 1
            mu = tf.get_variable("mu", shape=(1), dtype=tf.float32)
            sigma = tf.get_variable("sigma", shape=(1), dtype=tf.float32)
            f_zero = mu * left_side + sigma * right_side * f
        return voiced, f_zero

    def __voiced_model_loss(self, logits, target, seq_len):
        max_len = tf.reduce_max(seq_len)
        masks = tf.sequence_mask(
            seq_len, max_len, dtype=tf.float32
        )
        targets = tf.slice(target, [0, 0], [-1, max_len])
        logits = tf.slice(logits, [0, 0, 0], [-1, max_len, -1])
        return tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=masks)

    def __frequency_model_loss(self, output, target, seq_len):
        max_len = tf.reduce_max(seq_len)
        weights = tf.sequence_mask(
            seq_len, max_len, dtype=tf.float32
        )
        targets = tf.slice(target, [0, 0], [-1, max_len])
        output = tf.slice(output, [0, 0], [-1, max_len])
        loss = tf.abs(targets - output) * weights
        loss = tf.reduce_sum(loss, 1)
        loss = loss / tf.cast(seq_len, tf.float32)
        return tf.reduce_mean(loss)

    def build_train_operations(self,
                        phonemes, phonemes_seq_len,
                        speaker_ids,
                        voiced_target,
                        frequency_target,
                        train_parameters,
                        reuse=None
                       ):
        voiced, f_zero = self.__build_encoder(
            phonemes, phonemes_seq_len, speaker_ids, reuse
        )

        loss = self.__voiced_model_loss(voiced, voiced_target, phonemes_seq_len)
        loss = loss + self.__frequency_model_loss(f_zero, frequency_target, phonemes_seq_len)
        tf.summary.scalar('loss', loss)

        global_step = tf.Variable(
            0, name="global_step", trainable=False
        )
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

    def build_prediction(self,
                        phonemes, phonemes_seq_len,
                        speaker_ids, reuse=None
                       ):
        return self.__build_encoder(phonemes, phonemes_seq_len, speaker_ids, reuse)
