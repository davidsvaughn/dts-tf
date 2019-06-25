from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#######################################

import os, sys
import re
# import tempfile
from backports import tempfile

import numpy as np
import tensorflow as tf
from distutils.dir_util import copy_tree

def print_map(m, name=None):
    if name:
        print(name)
    for k in m.keys():
        print('{}\t=> {}'.format(k, m[k]))
    print('')
    
def get_temp_dir():
    return tempfile.mkdtemp()
    
def get_temp_filename():
    default_tmp_dir = tempfile._get_default_tempdir()
    temp_name = next(tempfile._get_candidate_names())
    return os.path.join(default_tmp_dir, temp_name)

def copy_chkpt(src_chkpt_dir, dst_chkpt_dir):
    copy_tree(src_chkpt_dir, dst_chkpt_dir)

def reverse(s):
    return s[::-1]

def common_suffix(s1, s2):
    return reverse(os.path.commonprefix([reverse(s1), reverse(s2)]))

def longest_common_suffix(s, slist):
    sxs = [common_suffix(s, ss) for ss in slist]
    idx = np.argmax(map(len, sxs))
    return slist[idx], sxs[idx]

def map_rename_vars(new_vars, old_vars):
    vmap, rmap = {}, {}
    for new_var in new_vars:
        old_var, suffix = longest_common_suffix(new_var, old_vars)
        vmap[old_var] = new_var
        rmap[old_var[:-len(suffix)]] = new_var[:-len(suffix)]
    return vmap, rmap

def rename_all(checkpoint_dir, rmap, dry_run=False): 
    
    replace_from = rmap.keys()[0]
    replace_to = rmap[replace_from]

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Graph().as_default(), tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                #new_name = new_name.replace(replace_from, replace_to)
                new_name = re.sub('^{}'.format(replace_from), replace_to, new_name)

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)

def rename(chkpt_dir, variables_to_restore, dry_run=False):

    # [var.name for var in variables_to_restore]
    varnames_to_restore = [re.sub(':[0-9]$', '', var.name) for var in variables_to_restore]
    
    chkpt_vars = [var_name for var_name, _ in tf.contrib.framework.list_variables(chkpt_dir)]
    ###########
    print('\tVARIABLES IN SAVED MODEL...')
    [print('\t'+var) for var in chkpt_vars]
    print('')
    ###########
    
    vmap, rmap = map_rename_vars(varnames_to_restore, chkpt_vars)
#     print_map(rmap,'rmap')
#     print_map(vmap,'vmap')

    if len(rmap)!=1:
        print('ERROR: no unique variable rename mapping!')
        print_map(rmap,'rmap')
        print_map(vmap,'vmap')
        sys.exit()
        return
    
#     rename_all(chkpt_dir, rmap, dry_run=dry_run)

    checkpoint = tf.train.get_checkpoint_state(chkpt_dir)#, latest_filename=tf.train.latest_checkpoint(chkpt_dir))
    with tf.Graph().as_default(), tf.Session() as sess:
        for old_name in vmap.keys():
            var = tf.contrib.framework.load_variable(chkpt_dir, old_name)
            new_name = vmap[old_name]
            if dry_run:
                print('%s would be renamed to %s.' % (old_name, new_name))
            else:
                print('Renaming %s to %s.' % (old_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)
 
        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)
             
    print(chkpt_dir)

def rename_vars(variables_to_restore, src_chkpt_dir, dst_chkpt_dir=None):
    if dst_chkpt_dir is None:
        dst_chkpt_dir = get_temp_dir()
    copy_chkpt(src_chkpt_dir, dst_chkpt_dir)
    rename(dst_chkpt_dir, variables_to_restore)
    return dst_chkpt_dir


def restore_vars(session, variables_to_restore, chkpt_dir):
    with tempfile.TemporaryDirectory() as tmp_chkpt_dir:
        rename_vars(variables_to_restore, 
                    src_chkpt_dir=chkpt_dir,
                    dst_chkpt_dir=tmp_chkpt_dir)
        saver = tf.train.Saver(variables_to_restore)
        restore_chkpt = tf.train.latest_checkpoint(tmp_chkpt_dir)
        saver.restore(session, restore_chkpt)


def test1():
#     src_chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/mod2_600-15'
#     dst_chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/tmp'
    chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/mod2_600-15'
    varnames_to_restore = [
        'Model/FlatModel/char_embed_b/internal_embed/embeddings:0',
        'Model/FlatModel/TDNN/conv_2d/w:0',
        'Model/FlatModel/TDNN/conv_2d/b:0',
        'Model/FlatModel/TDNN/conv_2d_1/w:0',
        'Model/FlatModel/TDNN/conv_2d_1/b:0'
        ]
    with tempfile.TemporaryDirectory() as tmp_chkpt_dir:
        rename_vars(varnames_to_restore, 
                    src_chkpt_dir=chkpt_dir,
                    dst_chkpt_dir=tmp_chkpt_dir)
    print('done')

def test2():
    src_chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/mod2_600-15'
    dst_chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/tmp'

    varnames_to_restore = [
        'Model/HANModel/char_embed_b/internal_embed/embeddings:0',
        'Model/HANModel/TDNN/conv_2d/w:0',
        'Model/HANModel/TDNN/conv_2d/b:0',
        'Model/HANModel/TDNN/conv_2d_1/w:0',
        'Model/HANModel/TDNN/conv_2d_1/b:0'
        ]

    rename_vars(varnames_to_restore, 
                src_chkpt_dir=src_chkpt_dir,
                dst_chkpt_dir=dst_chkpt_dir)
    
    print('done')

if __name__ == '__main__':
#     test1()
    test2()
           
# Model/FlatModel/char_embed_b/internal_embed/embeddings:0
# Model/FlatModel/TDNN/conv_2d/w:0
# Model/FlatModel/TDNN/conv_2d/b:0
# Model/FlatModel/TDNN/conv_2d_1/w:0
# Model/FlatModel/TDNN/conv_2d_1/b:0

# varnames_to_restore = ['Model/FlatModel/TDNN/conv_2d/w:0','Model/FlatModel/TDNN/conv_2d/b:0','Model/FlatModel/TDNN/conv_2d_1/w:0','Model/FlatModel/TDNN/conv_2d_1/b:0']

# /tmp/tmpkyVUpL
# /tmp/tmpy7dhTP
# /tmp/tmpxvCboJ

# /tmp/tmpBiksCs