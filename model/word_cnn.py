import tensorflow as tf
import json
import os

class WordCNN(object):
    def __init__(self,parameter_path,num_classes,vocabulary_size,document_max_len=100,embeddings=None,l2_reg_lambda=0.0):
        self.parameter = json.load(open(os.path.join(parameter_path,'word_cnn.json'), encoding='utf-8'))
        print(self.parameter)
        self.embedding_size = self.parameter['embedding_size']
        self.learning_rate = self.parameter['learning_rate']
        self.filter_sizes = self.parameter['filter_size']
        self.num_filters = self.parameter['num_filters']

        # self.dropout = parameter['dropout']
        self.vocabulary_size = vocabulary_size
        self.document_max_len = document_max_len
        # self.num_classes=num_classes
        l2_loss=tf.constant(0.0)

        self.x = tf.placeholder(tf.int32, [None, self.document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.where(self.is_training, 0.5, 1.0)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)
            self.x_emb = tf.expand_dims(self.x_emb, -1)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = tf.layers.conv2d(
                self.x_emb,
                filters=self.num_filters,
                kernel_size=[filter_size, self.embedding_size],
                strides=(1, 1),
                padding="VALID",
                activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[document_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
            pooled_outputs.append(pool)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(h_drop, num_classes, activation=None)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                 global_step=self.global_step)
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self.optim=optimizer.apply_gradients(grads_and_vars, name='optimizer',global_step=self.global_step)


        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



        # with tf.name_scope('embeddding'):
        #     if embeddings==None:
        #         print('embeddings is none,and it will be init_random_uniform')
        #         init_embeddings = tf.random_uniform([self.vocabulary_size,self.embedding_size])
        #         self.embeddings = tf.get_variable('embeddings',initializer=init_embeddings)
        #         self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)
        #     else:
        #         self.x_emb = tf.nn.embedding_lookup(embeddings, self.x)
        #     self.x_emb = tf.expand_dims(self.x_emb,-1)
        #
        # pool_outputs = []
        # for i,filter_size in enumerate(self.filter_sizes):
        #     with tf.name_scope('conv-maxpool-%s'%filter_size):
        #         filter_shape=[filter_size,self.embedding_size,1,self.num_filters]
        #         W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W')
        #         b=tf.Variable(tf.constant(0.1,shape=[self.num_filters]),name='b')
        #         conv=tf.nn.conv2d(
        #             self.x_emb,
        #             W,
        #             strides=[1,1,1,1],
        #             padding='VALID',
        #             name='conv'
        #         )
        #         h=tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
        #         pooled=tf.nn.max_pool(
        #             h,
        #             ksize=[1,self.document_max_len-filter_size+1,1,1],
        #             strides=[1,1,1,1],
        #             padding='VALID'
        #         )
        #         pool_outputs.append(pooled)
        # num_filters_total=self.num_filters*len(self.filter_sizes)
        # self.h_pool=tf.concat(pool_outputs,3)
        # self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        #
        # with tf.name_scope('dropout'):
        #     self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout)
        #
        # with tf.name_scope('output'):
        #     W=tf.get_variable('W',shape=[num_filters_total,num_classes],initializer=tf.contrib.layers.xavier_initializer())
        #     b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name='b')
        #     l2_loss+=tf.nn.l2_loss(W)
        #     l2_loss+=tf.nn.l2_loss(b)
        #     self.scores=tf.nn.xw_plus_b(self.h_drop,W,b,name='scores')
        #     self.predictions=tf.argmax(self.scores,1,name='predictions')
        #
        # with tf.name_scope('loss'):
        #     losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.y)
        #     self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss
        #
        # with tf.name_scope('optimizer'):
        #     optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #     grads_and_vars = optimizer.compute_gradients(self.loss)
        #     self.optim=optimizer.apply_gradients(grads_and_vars, name='optimizer')
        #
        # with tf.name_scope('accuracy'):
        #     correct_predictions=tf.equal(self.y,self.predictions)
        #     self.accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'),name='accuracy')

