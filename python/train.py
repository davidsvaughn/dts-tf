from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#######################################
'''
https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
https://github.com/yvesx/tf-rnn-pub
( https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/ )
https://danijar.com/variable-sequence-lengths-in-tensorflow/
'''
###########################################################################################

# python -u train.py | tee log.txt
# sudo python -u train.py | tee log.txt
# sudo python -u train.py | tee log.txt | grep -Ev '(bfc_allocator)'

''' TensorBoard '''
# tensorboard --logdir=mod --port 6006 --debugger_port 6064

''' attn_vis '''
# python -m SimpleHTTPServer &
# http://localhost:8000/attn_vis.html

###########################################################################################
''' config '''
# config_file = 'config/flat.conf'
# config_file = 'config/han.conf'

# config_file = 'config/flat18.conf'

# config_file = 'config/han_pool.conf'
config_file = 'config/han.conf'

# config_file = 'config/flat_insuff.conf'
# config_file = 'config/han_insuff.conf'

# config_file = 'config/han_unc.conf'

#######################################

import os
import sys
import time
import argparse
import pprint
import pickle as pk
import json
import numpy as np
from backports import tempfile
from pathlib2 import Path
import subprocess

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import sonnet as snt
import checkmate

from config import options, config
import utils as U
import kappa as K

from vocab import Vocab
from text_reader import GlobReader, TextParser, FieldParser, EssayBatcher, ResponseBatcher, REGEX_NUM

import model.models as mod
import model.restore_variables as rv

############################################

def get_model(model_name):
    return getattr(mod, model_name)

def get_batcher(batcher_name):
    if batcher_name=='ResponseBatcher':
        return ResponseBatcher
    if batcher_name=='EssayBatcher':
        return EssayBatcher
    return None

#######################################

FLAGS = config.parse_config(config_file, options.get_parser())
train_ids = FLAGS.train_ids; FLAGS.train_ids = None
valid_ids = FLAGS.valid_ids; FLAGS.valid_ids = None
test_ids = FLAGS.test_ids; FLAGS.test_ids = None
test_y = FLAGS.test_y; FLAGS.test_y = None
test_yint = FLAGS.test_yint; FLAGS.test_yint = None

pid = FLAGS.item_id
trait = '' if FLAGS.trait is None else '_{}'.format(FLAGS.trait)
FLAGS.chkpt_dir = FLAGS.chkpt_dir.format(pid, trait)
if FLAGS.load_chkpt_dir: FLAGS.load_chkpt_dir = FLAGS.load_chkpt_dir.format(pid, trait)
pprint.pprint(FLAGS)

if test_ids is None:
    test_ids = []
if valid_ids is None:
    valid_ids = []

#######################################

''' setup checkpoint directory '''
if not os.path.exists(FLAGS.chkpt_dir):
    U.mkdirs(FLAGS.chkpt_dir)
    print('Created checkpoint directory', FLAGS.chkpt_dir)
else:
    U.purge_pattern(FLAGS.chkpt_dir, r'.dvaughn-linux$')
config.save_local_config(FLAGS)
config.save_log(FLAGS)

''' setup ATTN_VIS json dir '''
if FLAGS.attn_vis:
    attn_vis_path = os.path.join(FLAGS.chkpt_dir,'attn_vis')
    if not os.path.exists(attn_vis_path): U.mkdirs(attn_vis_path)
    else: U.clear_path(attn_vis_path)

essay_file = os.path.join(FLAGS.data_dir, FLAGS.text_pat).format(pid)

embed_matrix=None
if FLAGS.embed.word:
    ''' load Glove word embeddings, along with word vocab '''
    embed_matrix, word_vocab = Vocab.load_word_embeddings_ORIG(FLAGS.embed_path, 
                                                               FLAGS.embed_dim, 
                                                               essay_file, 
                                                               min_freq=FLAGS.min_word_count, 
                                                               spell_corr=FLAGS.spell_corr)
#     embed_matrix, word_vocab = Vocab.load_word_embeddings(FLAGS.embed_path, FLAGS.embed_dim, essay_file, min_freq=FLAGS.min_word_count)
    char_vocab, max_word_length = None, None
    print('Embedding matrix shape: {}'.format(embed_matrix.shape))
else:
    vocab_file = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
FLAGS.max_word_length = max_word_length
    
''' create essay reader, parser, & batcher '''
word_vocab.reset_counts()

## random bits
word_vocab.reset_counts()
tstats = False
FLAGS.is_test = False
score_col=1
if FLAGS.trait is not None: score_col = 1 + int(FLAGS.trait)

def create_batcher(essay_file, text_parser, FLAGS, shuf=True, batch_size=None, pkl=False, name=''):
    if shuf: pkl=False
    reader =  GlobReader(essay_file, chunk_size=10000, regex=REGEX_NUM, seed=FLAGS.rand_seed, shuf=shuf)
    fields = {0:'id', score_col:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader, seed=FLAGS.rand_seed)
    Batcher = get_batcher(FLAGS.batcher)
    batcher = Batcher(reader=field_parser, FLAGS=FLAGS, tstats=tstats, batch_size=batch_size, pkl=pkl, name=name)
    return batcher

text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length, FLAGS=FLAGS, tokenize=FLAGS.tokenize, keep_unk=FLAGS.keep_unk)

#########
## OLD SPLIT ##
if not FLAGS.new_split:
    batcher = create_batcher(essay_file, text_parser, FLAGS, shuf=not FLAGS.no_shuffle)

## NEW ##
else:
    if FLAGS.train_id_file is None:
        FLAGS.train_id_file = os.path.join(FLAGS.id_dir, 'tmp_train_ids.txt')
        cmd = 'grep -v -Fwf <(cat {} {} | cut -f1) <(cut -f1 {}) > {}'.format(FLAGS.test_id_file, FLAGS.valid_id_file, essay_file, FLAGS.train_id_file)
        print(cmd)
        subprocess.call(['bash', '-c', cmd])
    
    FLAGS.tmp_train_all = os.path.join(FLAGS.id_dir, 'tmp_train_all.txt')
    cmd = 'grep -Fwf {} {} > {}'.format(FLAGS.train_id_file, essay_file, FLAGS.tmp_train_all)
    print(cmd)
    subprocess.call(['bash', '-c', cmd])#os.system(cmd)
    
    if FLAGS.augment:
        FLAGS.tmp_train_base = os.path.join(FLAGS.id_dir, 'tmp_train_base.txt')
        cmd = 'cp {} {}'.format(FLAGS.tmp_train_all, FLAGS.tmp_train_base)
        print(cmd)
        os.system(cmd)
        
    batcher = create_batcher(FLAGS.tmp_train_all, text_parser, FLAGS, shuf=not FLAGS.no_shuffle)
#########

if tstats:
    u2t = word_vocab.unk2tot
    print('%unk:\t{0:0.1f}'.format(u2t*100.))
    sys.exit()
    
''' loop through full set, build subsets '''
all_ids = batcher.ystats.id
ymax = batcher.ystats.max
ymin = batcher.ystats.min
yw = ymax-ymin
print('YMIN={}\tYMAX={}'.format(ymin, ymax))

if FLAGS.new_split:
    FLAGS.tmp_test = os.path.join(FLAGS.id_dir, 'tmp_test.txt')
    cmd = 'grep -Fwf <(cut -f1 {}) {} > {}'.format(FLAGS.test_id_file, essay_file, FLAGS.tmp_test)
#     cmd = "grep -wf <(cut -f1 {} | sed -e 's/\(.*\)/^\\1/g') {} > {}".format(FLAGS.test_id_file, essay_file, FLAGS.tmp_test)
    print(cmd)
    subprocess.call(['bash', '-c', cmd])#os.system(cmd)
    
    FLAGS.tmp_valid = os.path.join(FLAGS.id_dir, 'tmp_valid.txt')
    cmd = 'grep -Fwf <(cut -f1 {}) {} > {}'.format(FLAGS.valid_id_file, essay_file, FLAGS.tmp_valid)
#     cmd = "grep -wf <(cut -f1 {} | sed -e 's/\(.*\)/^\\1/g') {} > {}".format(FLAGS.valid_id_file, essay_file, FLAGS.tmp_valid)
    print(cmd)
    subprocess.call(['bash', '-c', cmd])#os.system(cmd)
    
    valid_batcher = create_batcher(FLAGS.tmp_valid, text_parser, FLAGS, shuf=False, batch_size=100, pkl=FLAGS.pickle_data, name='valid')
    test_batcher = create_batcher(FLAGS.tmp_test, text_parser, FLAGS, shuf=False, batch_size=100, pkl=FLAGS.pickle_data, name='test')

if FLAGS.fast_sample and FLAGS.min_cut<1.0:
    FLAGS.tmp_train = os.path.join(FLAGS.id_dir, 'tmp_train.txt')
    n = float(batcher.ystats.c.sum())
    N = int(n * FLAGS.min_cut)
    FLAGS.new_train_set_cmd = 'shuf -n {} {} > {}'.format(N, FLAGS.tmp_train_all, FLAGS.tmp_train)
    print(FLAGS.new_train_set_cmd)
    os.system(FLAGS.new_train_set_cmd )
    train_batcher = create_batcher(FLAGS.tmp_train, text_parser, FLAGS, shuf=not FLAGS.no_shuffle)
    batcher = train_batcher

###############################################################################

ACC = FLAGS.loss=='mse' and len(batcher.ystats.c)==2
ROC = FLAGS.roc and len(batcher.ystats.c)==2
if ROC: FLAGS.roc_test_file = os.path.join(FLAGS.chkpt_dir,'roc_test.txt')

skip_test = None
if FLAGS.ets18 and len(test_ids)==0:
    z = [id for id in all_ids if not id.endswith('b')]
    zb = set([id+'b' for id in z])
    skip_test = z + list(zb.intersection(set(all_ids)))


if test_y is not None:
    all_y = batcher.ystats.y
    miny, maxy = float(all_y.min()), float(all_y.max())
    
    ## get test data truth
    test_id_list = np.array(list(test_ids))
    idx = np.in1d(all_ids, test_id_list)
    true_y = all_y[idx]
    true_ids = all_ids[idx]
    if len(true_ids)!=len(test_id_list):
        idx = np.in1d(test_id_list, true_ids)
        miss_ids = test_id_list[~idx]
        print('TEST SET ITEMS {} MISSING FROM TRAIN SET!'.format(miss_ids))
    #     sys.exit()
        if len(miss_ids)>10:
            sys.exit()
        test_id_list = test_ids = test_id_list[idx]
        test_y = test_y[idx]
        test_yint = test_yint[idx]
    
    ## sort truth data
    idx = np.argsort(true_ids)
    true_ids = true_ids[idx]
    true_y = true_y[idx]
    
    ## sort test data
    idx = np.argsort(test_id_list)
    test_ids = test_id_list = test_id_list[idx]
    test_y = test_y[idx]
    test_yint = test_yint[idx]
    
    # print (sanity check)
#     for i,id in enumerate(true_ids):
#         print('{}\t{}\t{}'.format(id, true_y[i], test_yint[i]))
    
    ## compute test kappas
    kappa_int = U.ikappa(true_y, test_yint, all_y)
    kappa_float = U.nkappa(true_y, test_y)
    
    print('\nTEST QWK (int):\t{0:0.4f}\nTEST QWK (flt):\t{1:0.4f}\n'.format(kappa_int, kappa_float))
    # sys.exit()

###############################
test_ids = set(test_ids)
valid_ids = set(valid_ids)
train_ids = set(train_ids)

''' DEFINE DEFAULT INITIALIZER '''
initializer=tf.glorot_uniform_initializer(seed=FLAGS.rand_seed, dtype=tf.float32)
# initializer=tf.contrib.layers.xavier_initializer(seed=FLAGS.rand_seed, dtype=tf.float32)

if FLAGS.model_std and FLAGS.model_std>0:
    initializer=tf.truncated_normal_initializer(stddev=FLAGS.model_std,
                                                seed=FLAGS.rand_seed, 
                                                dtype=tf.float32)

''' DEFINE REGRESSION MODEL '''
with tf.variable_scope("Model", initializer=initializer) as scope:
    
    ''' build graph '''
    Model = get_model(FLAGS.model)
    model = Model(FLAGS=FLAGS,
                  embed_word=FLAGS.embed.word,
                  embed_matrix=embed_matrix,
                  char_vocab=char_vocab
                  )
    
    ''' get placeholders '''
    inputs = model.get_input_placeholder()
    targets = model.get_output_placeholder()
    
    ''' connect graph '''
    preds = model(inputs)#if isinstance(preds, tuple): preds = preds[-1]
    
    #########################################################################################
    
    ''' loss function '''
    if FLAGS.loss == 'mse':
        loss_op = tf.losses.mean_squared_error(targets, preds)
    else:
        loss_op = U.kappa_loss(targets, preds)
    
    if model.penalty is not None:
        p_loss_op = tf.reduce_mean(FLAGS.attn_coef * model.penalty)
        loss_op += p_loss_op
    
    learning_rate = tf.get_variable(
        "learning_rate",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(FLAGS.learning_rate),
        trainable=False)
    
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
    
    max_norm = tf.get_variable(
        "max_norm",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(FLAGS.max_grad_norm),
        trainable=False)
    
    ''' TRAIN OPS '''
    kwargs = {}
    if FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer
    elif FLAGS.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer
        kwargs = { 'momentum':0.88 , 'decay':0.0 }
        learning_rate *= 3
    elif FLAGS.optimizer == 'adadelta':
        opt = tf.train.AdadeltaOptimizer
        #kwargs = { 'rho':0.88 }
    elif FLAGS.optimizer == 'adagrad':
        opt = tf.train.AdagradOptimizer
        #kwargs = { 'momentum':0.88 , 'decay':0.0 }
    elif FLAGS.optimizer == 'adagradda':
        opt = tf.train.AdagradDAOptimizer
        #kwargs = { 'momentum':0.88 , 'decay':0.0 }
    else:
        opt = tf.train.GradientDescentOptimizer
    optimizer = opt(learning_rate, **kwargs)
    
    def training_op(notrain=[], col=2):
        #[print(tv) for tv in tf.trainable_variables()]; print('')
        tvars = [v for v in tf.trainable_variables() if not U.any_in(notrain, v.name.split('/'))]
        #[print(tv) for tv in tvars]; print('')
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss_op, tvars), max_norm)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=global_step), global_norm
    
    train_ops = []
    if FLAGS.embed.char:
        train_ops.append(training_op(notrain=['char_embed_b', 'TDNN'])) # 0
        train_ops.append(training_op(notrain=['char_embed_b']))         # 1
    if FLAGS.embed.word:
        train_ops.append(training_op(notrain=['embeddings', 'word']))   # 0
        train_ops.append(training_op(notrain=['embeddings']))           # 1
    train_ops.append(training_op())                                     # 2
        
    #op = optimizer.minimize(loss_op)
    def train_op(epoch):
        #return op
        if FLAGS.embed.char:
            if epoch>=FLAGS.epoch_unfreeze_emb:
                #print('UNFREEZING CHAR EMBEDDINGS')
                return train_ops[2]
            if epoch>=FLAGS.epoch_unfreeze_filt:
                #print('UNFREEZING CHAR FILTERS')
                return train_ops[1]
        if FLAGS.embed.word:
            if epoch>=FLAGS.epoch_unfreeze_emb:
                #print('UNFREEZING WORD EMBEDDINGS')
                return train_ops[2]
            if epoch>=FLAGS.epoch_unfreeze_word:
                #print('UNFREEZING WORD LEVEL')
                return train_ops[1]
        return train_ops[0]

config.save_log(FLAGS)
############################################################################

saver = tf.train.Saver()
if FLAGS.save_model:
    best_ckpt_saver = checkmate.BestCheckpointSaver(
        save_dir=FLAGS.chkpt_dir,
        saver=saver,
    )
    
''' TRAINING SESSION '''
graph = tf.get_default_graph()
session = tf.Session()

''' setup TensorBoard '''
if FLAGS.tensorboard:
    session = tf_debug.TensorBoardDebugWrapperSession(session, "dvaughn-linux:6064")
    [tf.summary.histogram(v.name.replace(':', '_'), v) for v in tf.trainable_variables()]
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.chkpt_dir)#, sess.graph)
    writer.add_graph(graph)

''' detect GPU use '''#gpu
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
gpu_options.allow_growth = True
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def get_value_like(k, v):
    if isinstance(k, tuple):
        n = np.shape(k)[0]
        return v[:n]
    if isinstance(v, tuple):
        return v[0]
    return v
    #return np.squeeze(v)

def fetch(d,k):
    if isinstance(k, (list, tuple)):
        ret = map(lambda x: fetch(d,x), k)
        if isinstance(k, tuple):
            return tuple(ret)
        return ret
    return d[k] if k in d else None

''' START SESSION '''
with graph.as_default(), session as sess:
    tf.set_random_seed(FLAGS.rand_seed)
    tf.global_variables_initializer().run()
    
    ################################################################
    ''' RESTORE CHAR EMBEDDINGS FROM CHAR LANG MODEL '''
    if FLAGS.embed.char:
        variables_to_restore = [var for var in tf.global_variables()
                                if (not FLAGS.optimizer in var.name.lower())
                                and ('odel/char_embed_b/' in var.name
                                     or 'odel/TDNN/' in var.name)
#                                 and ('char_cnn_embedding/char_embed_b/' in var.name
#                                      or 'char_cnn_embedding/TDNN/' in var.name)
                                ]
        print('\nRESTORING CHAR WEIGHTS...')
        [print(var.name) for var in variables_to_restore]
        rv.restore_vars(sess, variables_to_restore, chkpt_dir=FLAGS.char_embed_chkpt)
        print('DONE.\n')

    ################################################################
    ''' RESTORE PRE-TRAINED WEIGHTS ? '''
    if FLAGS.load_model and FLAGS.load_chkpt_dir:
        #[print(var.name) for var in tf.global_variables()]
        #saver.restore(sess, checkmate.get_best_checkpoint(FLAGS.load_chkpt_dir))
        variables_to_restore = [var for var in tf.global_variables()
                                if (FLAGS.model in var.name)
                                and (not 'embeddings' in var.name)
                                and (not FLAGS.optimizer in var.name.lower())
                                ]
        print('\nRESTORING MODEL WEIGHTS...')
        [print(var.name) for var in variables_to_restore]
        rv.restore_vars(sess, variables_to_restore, chkpt_dir=FLAGS.load_chkpt_dir)
        print('DONE.\n')
    
    num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('NUM PARAMS = {}'.format(num_params))
    ################################################################

    ''' TensorBoard '''
    #if FLAGS.tensorboard: writer = tf.summary.FileWriter(FLAGS.chkpt_dir, sess.graph)
    
    test_batches, best_test_score = [],-10000
    valid_batches, best_valid_score = [],-10000
    
    batcher.word_count(reset=True)
    config.save_log(FLAGS)
    epochs, epoch, step = FLAGS.epochs, 0, 0
    dropout = FLAGS.dropout * FLAGS.drop_sign
    lr = FLAGS.learning_rate
    gnorms = []
    tensor_shapes = None
    w_alphas = None
    aux_epoch=6 
    test_valid_ids = set(test_ids).union(set(valid_ids))
    
    while epoch < epochs:
        U.tic()
        epoch+=1
        if FLAGS.learning_rates and epoch in FLAGS.learning_rates:
            lr = FLAGS.learning_rates[epoch]
            print('NEW LEARNING RATE: {0:0.3g}'.format(lr))
        
        FLAGS.is_test = (epoch==1 and len(test_ids)==0 and U.rng.rand()<FLAGS.test_cut)
        
        print('==================================\tEPOCH {}\t\t=================================='.format(epoch))
        losses, qwks, P, Y, batch, k = [],[],[],[],0,0
        
        ''' load auxiliary info '''
        td_train, td_test = {},{}
        if model.aux_inputs is not None:
            for key in ['vector','pred']:
                fn = os.path.join(FLAGS.aux_dir, '{}_train_{}.npy'.format(key, aux_epoch))
                td_train[key] = np.load(fn)[()]
                fn = os.path.join(FLAGS.aux_dir, '{}_test_{}.npy'.format(key, aux_epoch))
                td_test[key] = np.load(fn)[()]
                
        ''' augment data '''
        if FLAGS.augment:
            cmd = 'cp {} {}'.format(FLAGS.tmp_train_base, FLAGS.tmp_train_all)
            print(cmd)
            os.system(cmd)
            cmd = 'python augment.py {} 20 1 >> {}'.format(FLAGS.tmp_train_base, FLAGS.tmp_train_all)
            print(cmd)
            os.system(cmd)
            # sys.exit()
                
        ''' TRAINING LOOP '''
        if FLAGS.fast_sample: 
            print(FLAGS.new_train_set_cmd )        
            os.system(FLAGS.new_train_set_cmd )
        
        for b in batcher.batch_stream(stop=True,
                                      skip_ids=None if FLAGS.new_split else test_valid_ids,
                                      sample=not FLAGS.fast_sample, 
                                      FLAGS=FLAGS,
                                      SZ=FLAGS.tensor_vol,
                                      skip_test=skip_test,
                                      ):
            batch+=1
            step+=1
            k+=b.n # k+=FLAGS.batch_size

            FLAGS.is_test = (epoch==1 and len(test_ids)==0 and U.rng.rand()<FLAGS.test_cut)
            
            if epoch==1:
                if b.is_test:
                    test_batches.append(b)
                    continue
            
                if len(valid_ids)==0 and U.rng.rand()<FLAGS.valid_cut:# and b.n==FLAGS.batch_size:
                    valid_batches.append(b)
                    continue
            
            feed_dict = { inputs: b.x,
                         targets: b.y,
                         model.keep_prob: 1.0-abs(FLAGS.dropout),
                         learning_rate: lr,
                         max_norm: U.get_max_norm(gnorms, default=FLAGS.max_grad_norm),
                         model.seq_len: get_value_like(model.seq_len, b.s),
                         }
            if model.aux_inputs is not None:
                feed_dict[model.aux_inputs] = [td_train['vector'][id] for id in b.id]
                
            train_step, global_norm = train_op(epoch)
            
            #####################################

            fetches = {}
            fetches['train_step'] = train_step
            fetches['loss'] = loss_op
            fetches['p'] = preds
            fetches['gnorm'] = global_norm
            if batch+epoch==2: fetches['tensor_shapes'] = model.tensor_shapes
            #fetches['tensors'] = model.tensors 
            
            fetched = sess.run(fetches, feed_dict)
            
            loss, p, gnorm = fetch(fetched, ('loss','p','gnorm'))
            tensors, tensor_shapes = fetch(fetched, ('tensors','tensor_shapes'))
            
            #####################################
            
            losses.append(loss)
            gnorms.append(gnorm)
            P.extend(np.squeeze(p)); Y.extend(np.squeeze(b.y))
            
            word_count = batcher.word_count(reset=False)
            sec = U.toc(reset=False)
            wps = int(word_count/sec)
            
            if batch % FLAGS.print_every == 0:
                if ACC:
                    acc = U.nacc(Y[-k:],P[-k:])
                    sys.stdout.write('\tacc={0:0.3f}'.format(acc))
                else:
                    kappa = U.nkappa(Y[-k:],P[-k:])
                    sys.stdout.write('\tqwk={0:0.3f}'.format(kappa))
                sys.stdout.write('|loss={0:0.3f}'.format(loss))
                #sys.stdout.write('|ploss={0:0.3g}'.format(p_loss))
                sys.stdout.write('|wps={0}'.format(wps))
                sys.stdout.write('|bs={0}'.format(b.n))
                #sys.stdout.write('|gnm={0:0.2f}'.format(gnorm))
                sys.stdout.flush()
                k=0
               
            if tensor_shapes is not None:
                print('')
                for ts in tensor_shapes: print('\t{}\t{}'.format(ts[0], ts[1]))
                print('')
        
        ''' SET UP TEST & VALID SETS (only after epoch 1) '''
        if not FLAGS.new_split and epoch==1:
            ##################################
            ## valid ids
            if len(valid_ids)==0:
                valid_ids = set([id for b in valid_batches for id in b.id])
                if FLAGS.save_valid:
                    U.write_sequence(FLAGS.valid_id_file, valid_ids)
            else:# pre-loaded valid ids
                print('\nSTILL SAVING VALID BATCHES!!!!!')
                for b in batcher.batch_stream(stop=True,
                                              hit_ids=valid_ids,
                                              sample=False,
                                              partial=True,
                                              FLAGS=FLAGS,
                                              ):
                    valid_batches.append(b)
                    #seen_ids.update(b.id)
            valid_ys = [y for b in valid_batches for y in b.y]
            valid_ystats = U.compute_ystats(valid_ys)
            
            ##################################
            ## test ids
            if len(test_ids)==0:
                test_ids = set([id for b in test_batches for id in b.id])
                if FLAGS.save_test:
                    U.write_sequence(FLAGS.test_id_file, test_ids)
            else:# pre-loaded test_ids
                print('STILL SAVING TEST BATCHES!!!!!')
                for b in batcher.batch_stream(stop=True,
                                              hit_ids=test_ids,
                                              sample=False,
                                              partial=True,
                                              FLAGS=FLAGS,
                                              ):
                    test_batches.append(b)
                    #seen_ids.update(b.id)
            test_ys = [y for b in test_batches for y in b.y]
            test_ystats = U.compute_ystats(test_ys)
            
            print('\n\nVALID SET :\t{} {}\t({} batches)'.format(len(valid_ids), list(valid_ystats.c), len(valid_batches)))
            print('TEST SET :\t{} {}\t({} batches)'.format(len(test_ids), list(test_ystats.c), len(test_batches)))
            
            ## finish
            train_ids = None
            test_valid_ids = set(test_ids).union(set(valid_ids))
        
        
        word_count = batcher.word_count(reset=True)
        sec = U.toc(reset=True)
        wps = int(word_count/sec)
        if ACC:
            ACC_train = U.nacc(Y,P)
            train_msg = 'Epoch {0} \tTRAIN Loss : {1:0.4}\tTRAIN Accuracy : {2:0.4}\t{3:0.2g}min|{4}wps'.format(epoch, np.mean(losses), ACC_train, float(sec)/60.0, wps)
        else:
            QWK_train = U.nkappa(Y,P)
            train_msg = 'Epoch {0} \tTRAIN Loss : {1:0.4}\tTRAIN Kappa : {2:0.4}\t{3:0.2g}min|{4}wps'.format(epoch, np.mean(losses), QWK_train, float(sec)/60.0, wps)
        print('\n[CURRENT]\t' + train_msg)
        
        
        ##########################################################################
        ''' VALIDATION '''
        
        ''' loop '''
        losses, qwks, P, Y, I, batch, k = [],[],[],[],[],0,0
        #for b in valid_batches 
        for b in (valid_batcher.batch_stream(stop=True,
                                            sample=False,
                                            partial=True,
                                            FLAGS=FLAGS,
                                            )
                  if FLAGS.new_split
                  else valid_batches):
            batch+=1
            k+=b.n
            feed_dict = { inputs: b.x,
                         targets: b.y,
                         model.keep_prob: 1.0,
                         model.seq_len: get_value_like(model.seq_len, b.s),
                         }
                
            fetches = {}
            fetches['loss'] = loss_op
            fetches['p'] = preds
            fetched = sess.run( fetches, feed_dict)
            
            loss, p = fetch(fetched, ('loss','p'))
            losses.append(loss)
            p = np.squeeze(p) if b.n>1 else p; P.extend(p[0:b.n])
            y = np.squeeze(b.y) if b.n>1 else b.y; Y.extend(y[0:b.n])
            id = np.squeeze(b.id) if b.n>1 else b.id; I.extend(id[0:b.n])
            
            if batch % FLAGS.print_every == 0:
                sys.stdout.write('\tloss={0:0.4}'.format(loss))
                if ACC:
                    acc = U.nacc(Y[-k:],P[-k:])
                    sys.stdout.write('|acc={0:0.4}\n'.format(acc))
                else:
                    kappa = U.nkappa(Y[-k:],P[-k:])
#                     sys.stdout.write('|qwk={0:0.4}\n'.format(kappa))
                    sys.stdout.write('|qwk={}\n'.format(kappa))
                sys.stdout.flush()
                k=0
        
        ## eliminate duplicate ids
        #U.write_sequence('/home/david/data/insuff/travis/bw/i1.txt', I)
        I,idx = np.unique(I, return_index=True)
        #U.write_sequence('/home/david/data/insuff/travis/bw/i2.txt', I)
        Y = np.array(Y, np.float64)[idx]
        P = np.array(P, np.float64)[idx]
        
        if epoch==1: valid_ystats = U.compute_ystats(Y)
        
        sec = U.toc(reset=False)
        rps = len(Y)/sec
        print('{} responses per second'.format(rps))
        
        if ACC:
            ACC_valid = U.nacc(Y,P)
            SCORE_valid = ACC_valid
        else:
            QWK_valid = U.nkappa(Y,P)
            QWK_valid_int = K.ikappa(Y,P,yw)
            SCORE_valid = QWK_valid
        
        ##########################################################################
        ''' TEST '''
        
        ''' attn_vis '''
        if FLAGS.attn_vis:
            json_path = os.path.join(attn_vis_path,'epoch_{}'.format(epoch))
            if not os.path.exists(json_path):
                U.mkdirs(json_path)
            json_files = []
        
        ''' loop '''
        losses, qwks, P, Y, S, I, batch, k = [],[],[],[],[],[],0,0
        #for b in test_batches:
        for b in (test_batcher.batch_stream(stop=True,
                                           sample=False,
                                           partial=True,
                                           FLAGS=FLAGS,
                                           )
                  if FLAGS.new_split
                  else test_batches):
            batch+=1
            k+=b.n
            feed_dict = { inputs: b.x,
                         targets: b.y,
                         model.keep_prob: 1.0,
                         model.seq_len: get_value_like(model.seq_len, b.s),
                         }
            if model.aux_inputs is not None:
                feed_dict[model.aux_inputs] = [td_test['vector'][id] for id in b.id]
                
            fetches = {}
            fetches['loss'] = loss_op
            fetches['p'] = preds
            if FLAGS.attn_vis:
                fetches['alpha_word'] = model.alpha_word
                fetches['alpha_sent'] = model.alpha_sent
            if FLAGS.tensorboard: fetches['summary'] = merged_summary
#             if ROC: fetches['score'] = model.score
            
            fetched = sess.run( fetches, feed_dict)
            
            loss, p = fetch(fetched, ('loss','p'))
            losses.append(loss)
            p = np.squeeze(p) if b.n>1 else p; P.extend(p[0:b.n])
            y = np.squeeze(b.y) if b.n>1 else b.y; Y.extend(y[0:b.n])
            id = np.squeeze(b.id) if b.n>1 else b.id; I.extend(id[0:b.n])
#             p = p[0]; P.extend(p[0:b.n])
#             y = b.y[0]; Y.extend(y[0:b.n])
            if FLAGS.tensorboard: writer.add_summary(fetch(fetched,'summary'), step)
            
            ################################################################
            
            ''' attn_vis files '''
            if FLAGS.attn_vis:
                alpha_word, alpha_sent = fetch(fetched, ('alpha_word','alpha_sent'))
                s_alphas = U.t2l(alpha_sent, b.s[0], b.p[0]) if len(alpha_sent)>0 else None
                w_alphas = U.tensor2list(alpha_word, b.s, b.p)
                words_idx = U.tensor2list(b.w, b.s, b.p)
                words = U.list2list(words_idx, word_vocab.token)
                  
                _json_files = U.generate_json_docs(path=json_path,
                                                   ids=b.id[0:b.n], 
                                                   targets=np.squeeze(b.y)[0:b.n], 
                                                   preds=np.squeeze(p)[0:b.n], 
                                                   words=words, 
                                                   w_alphas=w_alphas, 
                                                   s_alphas=s_alphas)
                json_files.extend(_json_files)
            
            ################################################################
            
            
            if batch % FLAGS.print_every == 0:
                sys.stdout.write('\tloss={0:0.4}'.format(loss))
                if ACC:
                    acc = U.nacc(Y[-k:],P[-k:])
                    sys.stdout.write('|acc={0:0.4}\n'.format(acc))
                else:
                    kappa = U.nkappa(Y[-k:],P[-k:])
#                     sys.stdout.write('|qwk={0:0.4}\n'.format(kappa))
                    sys.stdout.write('|qwk={}\n'.format(kappa))
                sys.stdout.flush()
                k=0
        
        ## eliminate duplicate ids
        I,idx = np.unique(I, return_index=True)
        Y = np.array(Y, np.float64)[idx]
        P = np.array(P, np.float64)[idx]
        
        ## trim item_id off response_id --> make integer
        I = [i.split('_')[0] for i in I]
#         I = np.array(I, np.int32)

        if epoch==1:
            test_ystats = U.compute_ystats(Y)
            print('\nVALID SET :\t{} {}'.format(len(valid_ids), list(valid_ystats.c)))
            print('TEST SET :\t{} {}'.format(len(test_ids), list(test_ystats.c)))
        
        if ACC:
            ACC_test = U.nacc(Y,P)
            SCORE_test = ACC_test
            valid_msg = 'Epoch {0} \tVALID Acc : {3}{2:0.4}{4}\tTEST Acc : {6}{5:0.4}{7}'.format(epoch, np.mean(losses), ACC_valid, U.BColors.BGREEN, U.BColors.ENDC, ACC_test, U.BColors.BGREEN, U.BColors.ENDC)

        else:              
            QWK_test = U.nkappa(Y,P)
            QWK_test_int = K.ikappa(Y,P,yw)
            SCORE_test = QWK_test
            valid_msg = 'Epoch {0} \tVALID Kappa : {3}{2:0.4}{4} [{8:0.4}]\tTEST Kappa : {6}{5:0.4}{7} [{9:0.4}]'.format(epoch, np.mean(losses), QWK_valid, U.BColors.BGREEN, U.BColors.ENDC, QWK_test, U.BColors.BGREEN, U.BColors.ENDC, QWK_valid_int, QWK_test_int)

        if SCORE_valid>best_valid_score:
            best_valid_score=SCORE_valid
            best_valid_msg=valid_msg
            if ROC: U.save_roc(FLAGS.roc_test_file, Y, P, I)
            
        if FLAGS.save_model: 
            if best_ckpt_saver.handle(SCORE_valid, sess, global_step):
                if ROC: U.save_roc(FLAGS.roc_test_file, Y, P, I)
        ''' 
        ## HowTo: load from checkpoint      
        saver.restore(sess, checkmate.get_best_checkpoint(FLAGS.load_chkpt_dir))
        '''
            
        print('[CURRENT]\t' + valid_msg)
        print('[BEST]\t\t' + best_valid_msg)
        
        if ACC:
            test_msg = 'Epoch {0} \tTEST Acc : {3}{2:0.4}{4}'.format(epoch, np.mean(losses), ACC_test, U.BColors.BGREEN, U.BColors.ENDC)
        else:
            test_msg = 'Epoch {0} \tTEST Kappa : {3}{2:0.4}{4} [{5:0.4}]'.format(epoch, np.mean(losses), QWK_test, U.BColors.BGREEN, U.BColors.ENDC, QWK_test_int)
            
        if SCORE_test>best_test_score:
            best_test_score=SCORE_test
            best_test_msg=test_msg
            
        #print('[CURRENT]\t' + test_msg)
        print('[BEST]\t\t' + best_test_msg + '\n')
        
        ###############################
        ''' attn_vis '''
        if FLAGS.attn_vis:
            data = {}
            json_files.sort()
            data["files"] = json_files
            json_data = json.dumps(data)
            with open(os.path.join(json_path, "attn_files.json"), 'w') as f:
                f.write(json_data)
        
        ###############################
        ''' ADAPT LEARNING RATE ?? '''
        if FLAGS.lr_decay!=None:
            lr*= FLAGS.lr_decay
            print('NEW LEARNING RATE: {0:0.4}'.format(lr))
        
        #### save log 
        config.save_log(FLAGS)
