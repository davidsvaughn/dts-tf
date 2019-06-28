import os.path
from pathlib2 import Path
import configargparse
import pprint
import json
import shutil

import utils as U
# from nlp.util import utils as U

## namespace->dict
def ns2dict(args):
    return vars(args)

## dict->namespace
def dict2ns(dict):
    return U.adict(dict)

def dump_flags_to_json(flags, file):
    with open(file, 'w') as f:
        json.dump(ns2dict(flags), f)#, ensure_ascii=False)

def restore_flags_from_json(file):
    with open(file, 'r') as f:
        str = f.read()#.encode('utf8')
    d = json.loads(str)
    return dict2ns(d)

def dump_config(flags, file):
    with open(file, 'w') as f:
        for k, v in sorted(flags.items()):
            if k != 'config':
                f.write('{} \t= {}\n'.format(k,v))
                
def copy_file(src, dst):
    shutil.copy2(src, dst)
    
def save_log(flags):
    try:
        copy_file(flags.log_file, flags.chkpt_dir)
    except IOError:
        pass    
    
def save_local_config(flags, verbose=True):
    loc_file = os.path.abspath(os.path.join(flags.chkpt_dir, os.path.basename(flags.config)))
    abs_config = os.path.abspath(flags.config)
    if os.path.realpath(loc_file) != os.path.realpath(abs_config):
        if not os.path.exists(flags.chkpt_dir):
            U.mkdirs(FLAGS.chkpt_dir)
            if verbose:
                print('Created checkpoint directory: {}'.format(os.path.abspath(flags.chkpt_dir)))
#         dump_config(flags, loc_file)
        copy_file(abs_config, loc_file)
        if verbose:
            print('Saving FLAGS to: {}'.format(loc_file))

def get_config(config_file=None, argv=[], parser=None):
    if config_file:
        # if passed an override config file -->
        # --> override chkpt_dir to point to same dir as config
        argv.append('--config'); argv.append(config_file)
        config_dir = os.path.dirname(config_file)
        if not config_dir.endswith('config'):
            argv.append('--chkpt_dir'); argv.append(config_dir)
    return parse_argv(argv=argv, parser=parser)

def parse_argv(argv=None, parser=None):
    if not parser:
        parser = get_demo_parser()
    args, unparsed = parser.parse_known_args(argv)
    if len(unparsed)>0:
        print('WARNING -- UNKOWN ARGUMENTS...')
        pprint.pprint(unparsed)
        print('\n')
    args = dict2ns(ns2dict(args)) ## convert namespace to attribute_dictionary
    #pprint.pprint(args)
    return args

def get_ids(fn, default=None):
    try:
        ids = set(U.read_col(fn, col=0, type='unicode'))
    except IOError:
        ids = default
    return ids

def parse_config(config_file, parser):
    #parser = options.get_parser()
    argv=[]# override config file here
    FLAGS = get_config(parser=parser, config_file=config_file, argv=argv)
    FLAGS.chkpt_dir = U.make_abs(FLAGS.chkpt_dir)
    
    if FLAGS.load_model:
        if FLAGS.load_chkpt_dir:
            FLAGS.load_chkpt_dir = U.make_abs(FLAGS.load_chkpt_dir)
        else:
            FLAGS.load_chkpt_dir = FLAGS.chkpt_dir
    else:
        if FLAGS.model=='HANModel':
            FLAGS.epoch_unfreeze_word = 0
    
    FLAGS.cwd = os.getcwd()
    FLAGS.log_file = os.path.abspath(os.path.join(FLAGS.cwd, 'log.txt'))
    
    FLAGS.rand_seed = U.seed_random(FLAGS.rand_seed)
    
    if FLAGS.id_dir is None:
        FLAGS.id_dir = FLAGS.data_dir
    else:
        FLAGS.id_dir = os.path.join(FLAGS.data_dir, FLAGS.id_dir).format(FLAGS.item_id)
        
    if FLAGS.attn_size>0:
        FLAGS.mean_pool = False
        if FLAGS.attn_type<0:
            FLAGS.attn_type=0
            
    if FLAGS.embed_type=='word':
        FLAGS.model_std = None
        FLAGS.attn_std = None
    
    #### test ids
    test_ids, test_id_file = None, None
    FLAGS.test_y, FLAGS.test_yint = None, None
    
    if FLAGS.test_pat is None:
        FLAGS.save_test = None
        FLAGS.load_test = None
    else:
        trait = ''
        if FLAGS.trait is not None:
            trait = '_{}'.format(FLAGS.trait)
        test_id_file = os.path.join(FLAGS.data_dir, FLAGS.test_pat).format(FLAGS.item_id, trait)
        #########################
        if FLAGS.load_test and U.check_file(test_id_file):
            
            #################################
            data = U.read_cols(test_id_file)
            test_ids = data[:,0]
            
            if test_ids.dtype.name.startswith('float'):
                test_ids = test_ids.astype('int32')
            test_ids = test_ids.astype('unicode')
                
            if data.shape[1]>1:
                FLAGS.test_yint = data[:,1].astype('int32')
                FLAGS.test_y = data[:,2].astype('float32')
            
        #########################
        if FLAGS.save_test and test_ids is not None:
            FLAGS.save_test = False
#     FLAGS.test_ids = set(test_ids) if test_ids is not None else []
    FLAGS.test_ids = test_ids if test_ids is not None else []
    FLAGS.test_id_file = test_id_file
    
    ''' don't overwrite MLT test ids!!! '''
    if 'test_ids' in FLAGS.test_id_file:
        FLAGS.save_test = False
        
    #### valid ids
    valid_ids, valid_id_file = None, None
    if FLAGS.valid_pat is None:
        FLAGS.save_valid = None
        FLAGS.load_valid = None
    else:
        trait = ''
        if FLAGS.trait is not None:
            trait = '_{}'.format(FLAGS.trait)
        valid_id_file = os.path.join(FLAGS.data_dir, FLAGS.valid_pat).format(FLAGS.item_id, trait)
        if FLAGS.load_valid:
            valid_ids = get_ids(valid_id_file)
        if FLAGS.save_valid and valid_ids is not None:
            FLAGS.save_valid = False
    #FLAGS.valid_ids = set(valid_ids) if valid_ids is not None else []
    FLAGS.valid_ids = valid_ids if valid_ids is not None else []
    FLAGS.valid_id_file = valid_id_file
    
    #### train ids
    train_ids, train_id_file =None, None
    if FLAGS.train_pat:
        trait = ''
        if FLAGS.trait is not None:
            trait = '_{}'.format(FLAGS.trait)
        train_id_file = os.path.join(FLAGS.data_dir, FLAGS.train_pat).format(FLAGS.item_id, trait)
        train_ids = get_ids(train_id_file, default=[])
    #FLAGS.train_ids = set(train_ids) if train_ids is not None else []
    FLAGS.train_ids = train_ids if train_ids is not None else []
    FLAGS.train_id_file = train_id_file
    
    ###################################
    FLAGS.embed = U.adict({'type':FLAGS.embed_type, 
                           'char':FLAGS.embed_type=='char', 
                           'word':FLAGS.embed_type=='word' })
    
    FLAGS.word_embed_dir = os.path.join(FLAGS.embed_dir, 'word')
    FLAGS.char_embed_dir = os.path.join(FLAGS.embed_dir, 'char')
    
    feats = ['kernel_widths','kernel_features','rnn_cells','rnn_sizes','rnn_bis','attn_sizes','attn_depths','attn_temps','pads','learning_rates']
    for feat in feats:
        if feat in FLAGS and FLAGS[feat]:
            FLAGS[feat] = eval(eval(FLAGS[feat]))
    FLAGS.wpad = FLAGS.pads[0]
    FLAGS.spad = None if len(FLAGS.pads)<2 else FLAGS.pads[1]
    
    if FLAGS.attn_depths[0]>1 or (len(FLAGS.attn_depths)>1 and FLAGS.attn_depths[1]>1):
        FLAGS.attn_vis=False
        
    if FLAGS.attn_sizes[0]<1:
        FLAGS.attn_vis=False
    
    if FLAGS.embed.char:
        FLAGS.attn_vis=False
    
    ###################################
    return FLAGS

def set_flags(FLAGS,i):
    flags = U.adict(FLAGS.copy())
    flags.rnn_cell = FLAGS.rnn_cells[i]
    flags.rnn_size = FLAGS.rnn_sizes[i]
    flags.bidirectional = FLAGS.rnn_bis[i]
    flags.attn_size = FLAGS.attn_sizes[i]
    flags.attn_depth = FLAGS.attn_depths[i]
    flags.attn_temp = FLAGS.attn_temps[i]
    flags.pad = FLAGS.pads[i]
    return flags

## demo parser...
def get_demo_parser():
    p = configargparse.ArgParser()#default_config_files=['config/model.conf'])
    
    p.add('-c', '--config', required=False, is_config_file=True, default='config/model.conf', help='config file path')
    
    p.add("-td", "--chkpt_dir", type=str, metavar='<str>', required=True, help="The path to the checkpoint dir")
    p.add("-dd", "--data_dir", type=str, metavar='<str>', required=True, help="The path to the data dir")
    p.add("-vf", "--vocab_file", type=str, metavar='<str>', required=True, help="The path to the vocab file")
    p.add("-rs", "--rnn_size", type=int, metavar='<int>', default=500, help='size of LSTM internal state')#500
    
    p.add("-bs", "--batch_size", type=int, metavar='<int>', default=128, help='number of sequences to train on in parallel')
    p.add("-rl", "--rnn_layers", type=int, metavar='<int>', default=2, help='number of layers in the LSTM')#2
    p.add("-hl", "--highway_layers", type=int, metavar='<int>', default=2, help='number of highway layers')#2
    p.add("-ces","--char_embed_size", type=int, metavar='<int>', default=15, help='dimensionality of character embeddings')
    p.add("-nus","--num_unroll_steps", type=int, metavar='<int>', default=35, help='number of timesteps to unroll for (word sequence length)')
    p.add("-me", "--max_epochs", type=int, metavar='<int>', default=50, help="")
    p.add("-mw", "--max_word_length", type=int, metavar='<int>', default=65, help="")

    p.add("--dropout", type=float, metavar='<float>', default=0.5, help="")
    
    p.add("-kw", "--kernel_widths", type=str, metavar='<str>', default='[1,2,3,4,5,6,7]', help="")
    p.add("-kf", "--kernel_features", type=str, metavar='<str>', default='[50,100,150,200,200,200,200]', help="")
    
    p.add("-lr", "--learning_rate", type=float, metavar='<float>', default=1.0, help='starting learning rate')
    p.add("-d",  "--learning_rate_decay", type=float, metavar='<float>', default=0.75, help='learning rate decay')
    p.add("-dw", "--decay_when", type=float, metavar='<float>', default=1.5, help='decay if validation perplexity does not improve by more than this much')
    p.add("-pi", "--param_init", type=float, metavar='<float>', default=0.05, help='initialize parameters at')
    p.add("-mn", "--max_grad_norm", type=float, metavar='<float>', default=5.0, help='normalize gradients at')
    
    p.add("-pe", "--print_every", type=int, metavar='<int>', default=50, help='how often to print current loss (batches)')
    p.add("-se", "--save_every", type=int, metavar='<int>', default=2, help='how often to validate AND save checkpoint (shards)')
    p.add("-vs", "--num_test_steps", type=int, metavar='<int>', default=100, help='num validation steps (batches)')
    p.add("-vpe","--test_print_every", type=int, metavar='<int>', default=10, help='how often to print current loss (batches)')
    p.add("-de", "--decay_every", type=int, metavar='<int>', default=50, help='how often to decay lr (shards)')
    
    p.add("--EOS", type=str, metavar='<str>', default='+', help='<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
    p.add("--seed", type=int, metavar='<int>', default=0, help="")
    
    return p


if __name__ == "__main__":

    config_file = None
    #config_file = 'chkpt/mod1_650-20/model.conf'
    FLAGS = get_config(config_file)
    pprint.pprint(FLAGS)
