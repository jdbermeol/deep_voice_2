import sys
sys.path.append("..")

import os
from utils.dataset_utils import build_dataset
from models.grapheme_to_phoneme import GraphemeToPhoneme
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm

train_parameters = {
    "lr": 0.001,
    "decay_steps": 1000,
    "decay_rate": 0.85,
    "batch_size": 64
}
model_parameters = {
  "embedding_size": 256,
  "num_units": 1024,
  "num_layers": 3,
  "dropout_prob": 0.95,
  "num_beams": 5
}
shuffle_buffer_size = 1000
num_steps = 3000
save_energy = 200
metadata_file = "../data/cmu.pkl"
train_file = "../data/cmu_data.npz"
save_path = "../weights/train_grapheme_to_phoneme_model_script/"
log_path = "../log/train_grapheme_to_phoneme_model_script/"
restore = True
checkpoint_path = save_path



with open(metadata_file, "r") as read_file:
    meta = pickle.load(read_file)
char2id = meta["char2id"]
id2char = meta["id2char"]
phoneme2id = meta["phoneme2id"]
id2phoneme = meta["id2phoneme"]

input_vocab_size = len(char2id)
output_vocab_size= len(phoneme2id)

end_token = phoneme2id["<eos>"]

data = np.load(train_file)

with tf.Session() as sess:
    dataset = build_dataset(
        sess,
        (data["X"], data["X_seq_len"],data["Y"], data["Y_seq_len"]),
        ("X", "X_seq_len", "Y", "Y_seq_len"),
        train_parameters["batch_size"],
        shuffle_buffer_size
    )

    model = GraphemeToPhoneme(
        dataset["X"], dataset["X_seq_len"], input_vocab_size,
        output_vocab_size, end_token, model_parameters,
        dataset["Y"], dataset["Y_seq_len"], train_parameters
    )

    train_writer = tf.summary.FileWriter(log_path, sess.graph)

    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=3)

    if restore:
        print("restoring weights")
        if os.path.isdir(checkpoint_path):
            latest_ckpt = tf.train.latest_checkpoint(
                checkpoint_path
            )
            saver.restore(sess, latest_ckpt)
        else:
            saver.restore(sess, checkpoint_path)


    for _ in tqdm(xrange(num_steps)):
        out = sess.run([
            model.train_op,
            model.global_step,
            model.loss,
            model.training_output,
            model.summary,
            dataset,
        ])
        _, global_step, loss, output, summary, inputs = out

        train_writer.add_summary(summary, global_step)

        # detect gradient explosion
        if loss > 1e8 and global_step > 500:
            print("loss exploded")
            break

        if global_step % save_energy == 0 and global_step != 0:

            print("saving weights")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saver.save(sess, save_path, global_step=global_step)

            text = "".join([id2char[inputs["X"][0, i]] for i in xrange(inputs["X_seq_len"][0])])
            ideal = " ".join([id2phoneme[inputs["Y"][0, i]] for i in xrange(inputs["Y_seq_len"][0])])
            sample = " ".join([id2phoneme[i] for i in output[0]])
            step = "_" + str(global_step)
            merged = sess.run(tf.summary.merge([
                tf.summary.text("text" + step, tf.convert_to_tensor([text, ideal, sample]))
            ]))
            train_writer.add_summary(merged, global_step)

    coord.request_stop()
    coord.join(threads)
