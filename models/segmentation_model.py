import tensorflow as tf


class SegmentationModel(object):

    def __init__(self,
                 X, X_seq_len,
                 output_vocab_size,
                 model_parameters,
                 Y=None,
                 train_parameters=None, reuse=None
                 ):
        prediction_model_parameters = model_parameters.copy()
        prediction_model_parameters["dropout_prob"] = 0.0

        self.global_step = tf.Variable(
            0, name="global_step", trainable=False
        )

        X_expanded = tf.expand_dims(X, -1)

        with tf.variable_scope("conv_layer_0", reuse=reuse):
            conv_layer_0_output = tf.layers.conv2d(
                X_expanded,
                model_parameters["conv_num_filters"],
                model_parameters["conv_kernel_size"],
                padding="same"
            )

        prediction_encoder_output = self.__build_encoder(
            conv_layer_0_output, X_seq_len,
            output_vocab_size,
            prediction_model_parameters,
            reuse, False
        )
        with tf.variable_scope("decoder", reuse=reuse):
            self.prediction = tf.nn.ctc_beam_search_decoder(
                tf.transpose(prediction_encoder_output, perm=[1, 0, 2]),
                X_seq_len,
                beam_width=prediction_model_parameters["num_beams"],
            )

        if Y is not None:
            training_encoder_output = self.__build_encoder(
                conv_layer_0_output, X_seq_len,
                output_vocab_size,
                model_parameters,
                True, True
            )
            with tf.variable_scope("decoder", reuse=True):
                self.training_output = tf.nn.ctc_greedy_decoder(
                    tf.transpose(training_encoder_output, perm=[1, 0, 2]),
                    X_seq_len
                )
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(
                Y,
                tf.transpose(training_encoder_output, perm=[1, 0, 2]),
                X_seq_len
            ))
            tf.summary.scalar('loss', self.loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op= self.__train_op(
                    self.loss, self.global_step, train_parameters
                )
        self.summary = tf.summary.merge_all()

    def __build_encoder(self,
                        inputs, X_seq_len,
                        output_vocab_size,
                        model_parameters,
                        reuse, train_phase):
        conv_output = self.__buil_conv_layers(
            inputs, model_parameters, reuse, train_phase
        )
        dropout_output = tf.layers.dropout(
            conv_output, model_parameters["dropout_prob"]
        )
        #Merge last 2 dimessions
        conv_final_output = tf.concat(tf.unstack(
            dropout_output, axis=3
        ), 2)
        bidirectional_output, _, _ = self.__buil_bidirectional_layer(
            conv_final_output, X_seq_len, model_parameters, reuse
        )
        with tf.variable_scope("output_projection", reuse=reuse):
            encoder_output = tf.layers.dense(
                bidirectional_output, output_vocab_size
            )
        return encoder_output

    def __buil_conv_layers(self, output, model_parameters, reuse, train_phase):
        for i in xrange(1, model_parameters["num_conv_layers"]):
            with tf.variable_scope("conv_layer_" + str(i), reuse=reuse):
                output = tf.nn.relu(
                    self.__batch_norm(tf.layers.conv2d(
                        output,
                        model_parameters["conv_num_filters"],
                        model_parameters["conv_kernel_size"],
                        padding='same', use_bias=False
                    ), train_phase) + output
                )
        return output

    def __batch_norm(self, x, train_phase):
        return tf.contrib.layers.batch_norm(
            x,
            scope='bn',
            is_training=train_phase,
        )

    def __buil_bidirectional_layer(self, X, X_seq_len, model_parameters, reuse):
        with tf.variable_scope("bidirectional_layers", reuse=reuse):
            cells_fw = [
                tf.nn.rnn_cell.GRUCell(model_parameters["num_bidirectional_units"])
                for i in xrange(model_parameters["num_bidirectional_layers"])
            ]
            cells_bw = [
                tf.nn.rnn_cell.GRUCell(model_parameters["num_bidirectional_units"])
                for i in xrange(model_parameters["num_bidirectional_layers"])
            ]
            return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, X,
                sequence_length=X_seq_len, dtype=tf.float32
            )

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
