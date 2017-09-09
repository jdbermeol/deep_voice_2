
import tensorflow as tf


class GraphemeToPhoneme(object):

    def __init__(self,
                 X, X_seq_len, input_vocab_size,
                 output_vocab_size, end_token, model_parameters,
                 Y=None, Y_seq_len=None,
                 train_parameters=None, reuse=None
                 ):
        prediction_model_parameters = model_parameters.copy()
        prediction_model_parameters["dropout_prob"] = 1.0

        target_input_vocab_size = output_vocab_size + 1

        self.global_step = tf.Variable(
            0, name='global_step', trainable=False
        )

        if Y is None:
            with tf.variable_scope("encoder", reuse=reuse):
                _, prediction_states, _ = self.__build_encoder(
                    X, X_seq_len, input_vocab_size, prediction_model_parameters
                )
            with tf.variable_scope("decoder", reuse=reuse):
                prediction_decoder_cells, prediction_decoder_embedding = self.__build_decoder(
                    target_input_vocab_size,
                    prediction_model_parameters
                )
        else:
            with tf.variable_scope("encoder", reuse=reuse):
                _, training_states, _ = self.__build_encoder(
                    X, X_seq_len, input_vocab_size, model_parameters
                )
            with tf.variable_scope("encoder", reuse=True):
                _, prediction_states, _ = self.__build_encoder(
                    X, X_seq_len, input_vocab_size, prediction_model_parameters
                )
            with tf.variable_scope("decoder", reuse=reuse):
                training_decoder_cells, training_decoder_embedding = self.__build_decoder(
                    target_input_vocab_size,
                    model_parameters
                )
            with tf.variable_scope("decoder", reuse=True):
                prediction_decoder_cells, prediction_decoder_embedding = self.__build_decoder(
                    target_input_vocab_size,
                    prediction_model_parameters
                )
            target_input = tf.concat(
                [tf.fill([tf.shape(Y)[0], 1], output_vocab_size), Y], 1
            )
            with tf.variable_scope("training_decoder", reuse=reuse):
                (training_decoder_rnn_output, self.training_output), _, _  = self.__buil_train_decoder(
                    training_decoder_cells, training_decoder_embedding, training_states,
                    target_input, Y_seq_len
                )
            self.loss = self.__loss(training_decoder_rnn_output, Y, Y_seq_len)
            tf.summary.scalar('loss', self.loss)
            self.train_op= self.__train_op(
                self.loss, self.global_step, train_parameters
            )
        with tf.variable_scope("prediction_decoder", reuse=reuse):
            self.prediction  = self.__build_prediction_decoder(
                prediction_decoder_cells, prediction_decoder_embedding,
                prediction_states,
                output_vocab_size, end_token,
                prediction_model_parameters
            )

        self.summary = tf.summary.merge_all()

    def __build_encoder(self, X, X_seq_len, input_vocab_size, model_parameters):
        embedding = tf.get_variable(
            'embedding',
            shape=(input_vocab_size, model_parameters["embedding_size"]),
            dtype=tf.float32
        )
        embedded_inputs = tf.nn.embedding_lookup(embedding, X)
        cells_fw = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(model_parameters["num_units"]),
                output_keep_prob = model_parameters["dropout_prob"]
            )
            for i in xrange(model_parameters["num_layers"])
        ]
        cells_bw = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(model_parameters["num_units"]),
                output_keep_prob = model_parameters["dropout_prob"]
            )
            for i in xrange(model_parameters["num_layers"])
        ]
        return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw, embedded_inputs,
            sequence_length=X_seq_len, dtype=tf.float32
        )

    def __build_decoder(self, target_input_vocab_size, model_parameters):
        cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(model_parameters["num_units"]),
                output_keep_prob = model_parameters["dropout_prob"]
            ) for i in xrange(model_parameters["num_layers"])
        ]
        cells_net = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.MultiRNNCell(cells), target_input_vocab_size
        )
        embedding = tf.get_variable(
            'embedding',
            shape=(
                target_input_vocab_size,
                model_parameters["embedding_size"]
            ), dtype=tf.float32
        )
        return cells_net, embedding

    def __buil_train_decoder(self, cells_net, embedding, states, Y, Y_seq_len):
        decoder_helper = tf.contrib.seq2seq.TrainingHelper(
            tf.nn.embedding_lookup(embedding, Y),
            Y_seq_len
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = cells_net,
            helper = decoder_helper,
            initial_state = states
        )
        return tf.contrib.seq2seq.dynamic_decode(training_decoder)

    def __loss(self, logits, Y, Y_seq_len):
        max_len = tf.reduce_max(Y_seq_len)
        masks = tf.sequence_mask(
            Y_seq_len, max_len, dtype=tf.float32
        )
        targets = tf.slice(Y, [0, 0], [-1, max_len])
        return tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=masks)

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

    def __build_prediction_decoder(self,
                                   cells_net, embedding, states,
                                   start_token, end_token,
                                   model_parameters):
        start_tokens = tf.tile(
            tf.constant([start_token], dtype=tf.int32),
            [tf.shape(states[0])[0]]
        )
        decoder_state = tuple([
            tf.contrib.seq2seq.tile_batch(
                state, multiplier=model_parameters["num_beams"]
            ) for state in states
        ])
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cells_net, embedding, start_tokens, end_token,
            decoder_state, model_parameters["num_beams"]
        )
        (predictions, _, lengths) = tf.contrib.seq2seq.dynamic_decode(decoder)
        best_predictions = tf.slice(predictions.predicted_ids, [0, 0, 0], [-1, -1, 1])
        best_predictions_lengths = tf.slice(lengths, [0, 0], [-1, 1])
        return best_predictions, best_predictions_lengths
