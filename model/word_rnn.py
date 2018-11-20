
import tensorflow as tf
from tensorflow.contrib import rnn

class WordRNN(object):
    def __init__(self,num_classes,vocabulary_size,document_max_len=100,embeddings=None):
        self.num_classes=num_classes
        self.document_max_len=document_max_len
        self.embedding_size=100
        self.num_hidden=256
        self.num_layers=2
        self.learning_rate=1e-4
        self.dropout_cell=0.8


        self.x=tf.placeholder(tf.int32,[None,self.document_max_len],name="x")
        self.x_len=tf.reduce_mean(tf.sign(self.x),1)
        self.y=tf.placeholder(tf.int32,[None],name="y")
        self.is_training=tf.placeholder(tf.bool,[],name="is_training")
        self.global_step=tf.Variable(0,trainable=False)


        with tf.name_scope("embedding"):
            init_embeddings=tf.random_uniform([vocabulary_size,self.embedding_size])
            self.embeddings=tf.get_variable("embedding",initializer=init_embeddings)
            self.x_emb=tf.nn.embedding_lookup(self.embeddings,self.x)
            # self.x_emb=tf.expand_dims(self.x_emb,-1)

        with tf.name_scope("bilstm"):
            fw_cells=[rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells=[rnn.BasicLSTMCell(self.num_hidden) for _ in range(self.num_layers)]

            fw_cells=[rnn.DropoutWrapper(cell,output_keep_prob=self.dropout_cell) for cell in fw_cells]
            bw_cells=[rnn.DropoutWrapper(cell,output_keep_prob=self.dropout_cell)for cell in bw_cells]

            rnn_outputs, _, _=rnn.stack_bidirectional_dynamic_rnn(fw_cells,bw_cells,self.x_emb,dtype=tf.float32)
            rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, document_max_len * self.num_hidden * 2])

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(rnn_outputs_flat, self.num_classes, activation=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.optim=optimizer.apply_gradients(grads_and_vars, name='optimizer',global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
