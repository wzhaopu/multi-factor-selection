import numpy as np
import tensorflow as tf
from tensorflow.layers import Dense, dropout, batch_normalization
from tensorflow.contrib.layers import l2_regularizer as l2_reg
from tensorflow.keras.backend import hard_sigmoid
class Model(object):
    def __init__(self, model_name, columns, nb_words, maxlen_dict={},
                 emb_size=128, lr=1e-3,
                 dropout_rate=0.1, 
                 embedding_dropout_type='none',
                 product_scaling=1,
                 reg_lambda=0.,
                 min_cut=-50., max_cut=8., cut_size=100,
                 clip_range=1.,
                 penalty=5e-4,
                 **kwargs):
        self.penalty = penalty
        """
        model_name: string, name of the base model, one of
            'LR', 'FM', 'widendeep', 'deepFM', 'IPNN', 'OPNN', 'PNNstar'
        columns: list, names of the columns to use
        nb_words: dict, nb of words in each of these columns
        emb_size: embedding size
        lr: learning rate
        dropout_rate: dropout rate for the embedding layers and the last hidden layer
        embedding_dropout_type: one of 'none', 'normal', 'field', 'embedding'
        """
        def get_embeddings():
            # outputs:
            #     inputs: a list of input tensors
            #     embeddings: 
            #         3D tensor with shape (batch_size, field_size, embedding_size)
            #     lr_weights:
            #         2D tensor with shape (batch_size, field_size)
            inputs = {}
            embs, lr_weights = [], []
            for col in columns:
                if col in nb_words and col not in maxlen_dict:
                    # categorical feature
                    dev = '/cpu:0' if nb_words[col]>400000 else "/device:GPU:0"
                    with tf.device(dev):
                        # inputs[col]: shape (batch_size x 1) -> (batch_size)
                        inputs[col] = tf.placeholder(tf.int32, [None])
                        # table: shape (number_of_words, embedding_size)
                        table = tf.get_variable(
                            "table_{}".format(col), [nb_words[col], emb_size],
                            initializer=tf.glorot_uniform_initializer())
                        # emb: shape (batch_size, embedding_size)
                        emb = tf.nn.embedding_lookup(table, inputs[col])
                        lr_table = tf.get_variable("LRtable_{}".format(col), [nb_words[col], 1])
                        lr_w = tf.nn.embedding_lookup(lr_table, inputs[col])
                elif col in maxlen_dict:
                    # sequential features
                    inputs[col] = tf.placeholder(tf.int32, [None, maxlen_dict[col]])
                    query_emb = tf.contrib.layers.embed_sequence(
                        inputs[col], nb_words[col], emb_size, 
                        scope="seq_emb", reuse=tf.AUTO_REUSE)
                    emb = tf.reduce_mean(query_emb, axis=1)
                    lr_w = tf.contrib.layers.embed_sequence(
                        inputs[col], nb_words[col], 1, 
                        scope="seq_lr_table", reuse=tf.AUTO_REUSE)
                    lr_w = tf.layers.flatten(lr_w)
                else:
                    # numerical feature
                    inputs[col] = tf.placeholder(tf.float32, [None])
                    lr_w = tf.layers.dense(tf.expand_dims(inputs[col],-1), 1)
                    emb = tf.layers.dense(tf.expand_dims(inputs[col],-1), emb_size)
                if col != 'Predicted_Logit' or len(columns)==1:
                    embs.append(emb)
                    lr_weights.append(lr_w)
            # tf.stack(embs, 1): shape (batch_size, field_size, embedding_size)
            return inputs, tf.stack(embs, 1), tf.concat(lr_weights, -1)
        """
        Base models:
            LR, FM, MLP, deepFM, IPNN, OPNN, PNNstar, wide&deep
        """
        def latent_LR(embeddings, lr_weights=None, **kwargs):
            return tf.reduce_sum(lr_weights, -1, keepdims=True)
        def latent_FM(embeddings, lr_weights=None, **kwargs):
            reduce = tf.reduce_mean if product_scaling else tf.reduce_sum
            sum_of_emb = tf.reduce_sum(embeddings, 1, keepdims=True)
            diff_of_emb = sum_of_emb - embeddings
            dots = tf.reduce_sum(embeddings*diff_of_emb, axis=-1)
            if 'scaling_factor' in kwargs:
                dots = dots * kwargs['scaling_factor']
            biases = tf.reduce_sum(embeddings, -1)
            fm_latent = tf.concat([dots, biases], 1)
            return fm_latent
        def latent_MLP_concat(embeddings, lr_weights=None, **kwargs):
            h = tf.layers.flatten(embeddings)
            if kwargs.get('n_hidden', 0)>0:
                for i in range(kwargs['n_hidden']):
                    h = tf.nn.relu(tf.layers.dense(
                        h, kwargs.get('dense_size', 200), 
                        kernel_initializer=tf.initializers.he_uniform(),
                        kernel_regularizer=l2_reg(reg_lambda),
                        name='MLP-fc{}'.format(i)))
            return h
        def latent_MLP_avg(embeddings, lr_weights=None, **kwargs):
            h = tf.reduce_mean(embeddings, -1)
            if kwargs.get('n_hidden', 0)>0:
                for i in range(kwargs['n_hidden']):
                    h = tf.nn.relu(tf.layers.dense(
                        h, kwargs.get('dense_size', 200), 
                        kernel_initializer=tf.initializers.he_uniform(),
                        kernel_regularizer=l2_reg(reg_lambda),
                        name='MLP-fc{}'.format(i)))
            return h
        def latent_deepFM(embeddings, lr_weights=None, **kwargs):
            if 'n_hidden' not in kwargs:
                kwargs['n_hidden'] = 3
            fm_h = latent_FM(embeddings, **kwargs)
            deep_h = latent_MLP_concat(embeddings, **kwargs)
            h = tf.concat([fm_h, deep_h], 1)
            return h
        def latent_PNN(embeddings, lr_weights=None, **kwargs):
            fm_h = latent_FM(embeddings, **kwargs)
            h = tf.nn.relu(tf.layers.dense(fm_h, emb_size, name='fc', 
                                           kernel_initializer=tf.initializers.he_uniform(),
                                           kernel_regularizer=l2_reg(reg_lambda),))
            return h
        def latent_OPNN(embeddings, lr_weights=None, **kwargs):
            sum_of_emb = tf.reduce_sum(embeddings, 1)
            a = tf.expand_dims(sum_of_emb, -1)
            b = tf.expand_dims(sum_of_emb, 1)
            outer = a*b
            h = tf.layers.flatten(outer)
            biases = tf.reduce_sum(embeddings, -1)
            h = tf.concat([h, biases], 1)
            h = tf.nn.relu(tf.layers.dense(h, emb_size, name='fc', 
                                           kernel_initializer=tf.initializers.he_uniform(),
                                           kernel_regularizer=l2_reg(reg_lambda),))
            return h
        def latent_PNNstar(embeddings, lr_weights=None, **kwargs):
            sum_of_emb = tf.reduce_sum(embeddings, 1)
            a = tf.expand_dims(sum_of_emb, -1)
            b = tf.expand_dims(sum_of_emb, 1)
            outer = a*b
            h = tf.layers.flatten(outer)
            biases = tf.reduce_sum(embeddings, -1)
            fm_h = latent_FM(embeddings, **kwargs)
            h = tf.concat([h, fm_h, biases], 1)
            h = tf.nn.relu(tf.layers.dense(h, emb_size, name='fc', 
                                           kernel_initializer=tf.initializers.he_uniform(),
                                           kernel_regularizer=l2_reg(reg_lambda),))
            return h
        def latent_widendeep(embeddings, lr_weights=None, **kwargs):
            if 'n_hidden' not in kwargs:
                kwargs['n_hidden'] = 2
            deep_h = latent_MLP_concat(embeddings, **kwargs)
            lr_out = tf.reduce_sum(lr_weights, -1, keepdims=True)
            return tf.concat([deep_h, lr_out], 1)
        '''
        *CHOOSE THE BASE MODEL HERE*
        '''
        models = {
            'LR': latent_LR, 
            'FM': latent_FM,
            'MLP_concat': latent_MLP_concat,
            'MLP_avg': latent_MLP_avg,
            'deepFM': latent_deepFM,
            'IPNN': latent_PNN,
            'OPNN': latent_OPNN,
            'PNNstar': latent_PNNstar,
            'widendeep': latent_widendeep,
        }
        get_latent = models[model_name]
        def build_model(embs, lr_w, dropout_on=tf.constant(False), 
                        min_cut=min_cut, max_cut=max_cut, cut_size=cut_size):
            if embedding_dropout_type == 'none' or dropout_rate<0.01:
                pass
            elif embedding_dropout_type=='standard':
                embs = tf.layaers.dropout(embs, rate=dropout_rate, training=dropout_on)
            elif embedding_dropout_type=='field':
                # random dropout some fields
                embs = tf.layers.dropout(
                    embs, rate=dropout_rate, training=dropout_on,
                    noise_shape=[embs.shape[1], tf.constant(1)])
                kwargs['scaling_factor'] = tf.cond(dropout_on, lambda: 1.-dropout_rate, lambda: 1.)
            elif embedding_dropout_type=='embedding':
                # random dropout some dimensions of embeddings
                embs = tf.layers.dropout(
                    embs, rate=dropout_rate, training=dropout_on,
                    noise_shape=[tf.constant(1), embs.shape[2]])
                kwargs['scaling_factor'] = tf.cond(dropout_on, lambda: 1.-dropout_rate, lambda: 1.)
            else:
                print("Unknown embedding_dropout_type! Embedding Dropout is bypassed.")
            self.embs = embs
            latent = get_latent(embs, lr_w, **kwargs)
            latent = dropout(latent, dropout_rate, training=dropout_on)
            logit = tf.layers.dense(latent, 1, name='last_fc', kernel_initializer=tf.initializers.he_uniform())
            return tf.nn.sigmoid(logit), logit
        with tf.variable_scope("model"):
            # build placeholders for inputs
            inputs, embs, lr_weights = get_embeddings()
            label = tf.placeholder(tf.float32, [None, 1])
            dropout_on = tf.placeholder(tf.bool)
            # build the optimizer and update op for the original model
            lr_now = tf.placeholder_with_default(lr, [], name='learning_rate')
            optimizer = tf.train.AdamOptimizer(lr_now)
            yhat, logit = build_model(embs, lr_weights, dropout_on)
            emb_reg_loss = tf.nn.l2_loss(embs) + tf.nn.l2_loss(lr_weights)
            pred_loss = tf.losses.log_loss(label, yhat)
            loss = pred_loss + reg_lambda*emb_reg_loss 
            gvs = optimizer.compute_gradients(loss)
            
            capped_gvs = [(tf.clip_by_value(grad, -clip_range, clip_range), var) for grad, var in gvs if grad is not None]
            update_op = optimizer.apply_gradients(capped_gvs)
        
        def predict(sess, X, drop=False):
            '''
            output: a list [yhat, logit]
            '''
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict[dropout_on] = drop
            return sess.run([yhat, logit], feed_dict)
        def train(sess, X, y, lr=lr):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict[label] = y.reshape((-1,1))
            feed_dict[dropout_on] = True
            feed_dict[lr_now] = lr
            return sess.run([loss, update_op], feed_dict)[0]
        self.predict = predict
        self.train = train
        self.inputs = inputs