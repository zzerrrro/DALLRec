import tensorflow as tf
from utility.helper import *
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class BPRMF(object):
    def __init__(self, data_config):
        self.model_type = 'bprmf'

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr
        # self.lr_decay = args.lr_decay

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # self.global_step = tf.Variable(0, trainable=False)

        self.weights = self._init_weights()

        # Original embedding.
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        # All ratings for all users.
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        self.mf_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.reg_loss

        # self.dy_lr = tf.train.exponential_decay(self.lr, self.global_step, 10000, self.lr_decay, staircase=True)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

        # self.updates = self.opt.minimize(self.loss, var_list=self.weights)

        self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()


        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')

        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)


def test1(sess, model):
    user_id = 0  # 假设我们关心的用户ID是0
    item_id = 43  # 假设我们关心的物品ID是10

    # 获取用户和物品的嵌入
    user_batch = np.array([user_id])  # 用户ID，作为批次的一部分
    item_batch = np.array([item_id])  # 物品ID，作为批次的一部分

    # 获取评分
    rating = sess.run(model.batch_ratings, {model.users: user_batch, model.pos_items: item_batch})
    print(f"User {user_id} rating for item {item_id} is {rating[0]}")
    predicted_probability = sess.run(tf.nn.sigmoid(model.batch_ratings),
                                     feed_dict={model.users: user_batch, model.pos_items: item_batch})
    print(predicted_probability)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    model = BPRMF(data_config=config)

    saver = tf.train.Saver()

    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                         '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # *********************************************************
    # reload the pretrained model parameters.
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/l%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.lr),
                                                         '-'.join([str(r) for r in eval(args.regs)]))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            users_to_test = list(data_generator.test_set.keys())
            # ret = test(sess, model, users_to_test, drop_flag=False)
            ret = test1(sess, model)

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')



