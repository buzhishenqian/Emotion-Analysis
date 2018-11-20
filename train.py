#! /usr/bin/env python

import tensorflow as tf
import time
import datetime
from data_helpers import *
from model.text_cnn import TextCNN
from model.word_cnn import WordCNN
from model.word_rnn import WordRNN
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# y_true = [1,2,3,4]
# y_pred = [1,1,1,4]
# target_name = ['class0','class1','class2','class3']
# f1=classification_report(y_true, y_pred, target_names=target_name)
# print(type(f1))


# exit()

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("model_dir", "./model", "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("model", "word_rnn", "word_cnn/word_rnn")
tf.flags.DEFINE_integer("num_classes", 4, "Data source for the negative data.")
tf.flags.DEFINE_string("data_dir", "./data/data/input", "Data source for the negative data.")
tf.flags.DEFINE_string("output_dir", "./output/wordcnn1", "Data source for the negative data.")
tf.flags.DEFINE_bool("do_train",False,'')
tf.flags.DEFINE_bool("do_dev",False,'')
tf.flags.DEFINE_bool("do_predict",True,'')
tf.flags.DEFINE_bool("restore",False,'')

# Model Hyperparameters
# tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")


tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info("****FLAGS****")
tf.logging.info(FLAGS.flag_values_dict())




qinggan_data = data_name(FLAGS.data_dir, 100)
vocab_size=qinggan_data.word_dict_zize


with tf.Session() as sess:
    if FLAGS.model == "word_cnn":
        model = WordCNN(FLAGS.model_dir,4,vocab_size)
    elif FLAGS.model == "word_rnn":
        model = WordRNN(4,vocab_size)
    # elif FLAGS.model == "vd_cnn":
    #     model = VDCNN(alphabet_size, CHAR_MAX_LEN, NUM_CLASS)
    # elif FLAGS.model == "word_rnn":
    #     model = WordRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    # elif FLAGS.model == "att_rnn":
    #     model = AttentionRNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    # elif FLAGS.model == "rcnn":
    #     model = RCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)
    else:
        raise NotImplementedError()

    tf.logging.info("****model parameter****")
    # tf.logging.info(model.parameter)

    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver(tf.global_variables())
    checkpoint_dir = os.path.abspath(os.path.join(FLAGS.output_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    if FLAGS.restore==True:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        tf.logging.info("****restore model****")
        tf.logging.info("restore path:%s",checkpoint_file)


    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }
        _, step, loss = sess.run([model.optim, model.global_step, model.loss], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}".format(time_str, step, loss))
        # train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: False
        }
        step, loss, accuracy = sess.run([model.global_step, model.loss, model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        predictions_out=[]
        predictions=sess.run([model.predictions], feed_dict)
        for prediction in predictions[0]:
            predictions_out.append(prediction)

        accuracy=accuracy_score(y_true=y_batch,y_pred=predictions_out)
        print(accuracy)
        target_names=['class0','class1','class2','class3']
        f1=classification_report(y_batch,predictions_out,target_names=target_names)
        print(f1)

        return predictions_out,accuracy,f1
    def test_step(x_batch):
        feed_dict = {
            model.x:x_batch,
            model.is_training:False
        }
        preditions_out=[]
        preditions=sess.run([model.predictions],feed_dict)
        for predition in preditions[0]:
            preditions_out.append(predition)
        print(preditions_out)
        return preditions_out

    if FLAGS.do_train:
        x_train, y_train = qinggan_data.get_train_example()
        x_dev, y_dev = qinggan_data.get_valid_example()
        tf.logging.info("****do_train****")
        for epoch in range(FLAGS.num_epochs):
            tf.logging.info("epoch step:%d",epoch)
            batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size)

            # train_batches = batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
            # # num_batches_per_epoch = (len(x_train) - 1) // FLAGS.batch_size + 1
            # max_accuracy = 0

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch,y_batch)
                # train_feed_dict = {
                #     model.x: x_batch,
                #     model.y: y_batch,
                #     model.is_training: True
                # }
                # _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)

                current_step = tf.train.global_step(sess, model.global_step)

                # print(current_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
    if FLAGS.do_dev:
        x_dev, y_dev = qinggan_data.get_valid_example()
        tf.logging.info("****do_dev****")
        predictions_out,accuracy,f1=dev_step(x_dev,y_dev)
        x_text,y_text = qinggan_data.get_valid_text()

        tf.logging.info("****write dev result****")
        write_csv()


    if FLAGS.do_predict:
        x_test, y_test = qinggan_data.get_valid_example()
        tf.logging.info("****do_predict****")
        predictions=test_step(x_test)



# def main(argv=None):
#     tf.logging.set_verbosity(tf.logging.INFO)
#     x_train, y_train, x_dev, y_dev, vocab_size = preprocess()
#     train(x_train, y_train, x_dev, y_dev, vocab_size)
#
# if __name__ == '__main__':
#     tf.app.run()