import os
import configargparse
import pprint

def get_parser(default_config=None):
    p = configargparse.ArgParser()#default_config_files=['config/model.conf'])
    
#     if default_config==None:
#         default_config = 'config/model.conf'
        
    p.add('--config', required=False, is_config_file=True, default=default_config, help='config file path')
    
    p.add('--model', type=str, required=True, help="Deep Learning Model")
    p.add('--batcher', type=str, required=True, help="Data Batcher")
    
    p.add("-id", "--item_id", type=str, required=True, help="itemID or modeID")
    p.add("--trait", type=str, required=False, default=None, help="trait")
    
    p.add("--chkpt_dir", type=str, required=True, help="The path to the checkpoint directory")
    p.add("--load_chkpt_dir", type=str, required=False, default=None, help="The path to pre-trained model chkpt")
    
    p.add("--load_model", action='store_true', help="")
    p.add("--save_model", action='store_true', help="")
    p.add("--roc", action='store_true', help="")
    p.add("--pickle_data", action='store_true', help="")
    p.add("--fast_sample", action='store_true', help="")
    
    p.add("--data_dir", type=str, required=True, help="The path to the data directory")
    p.add("--id_dir", type=str, required=False, default=None, help="The path to the id directory")
    p.add("--json_dir", type=str, required=False, default=None, help="The path to the json directory")
    p.add("--aux_dir", type=str, required=False, help="The path to the auxiliary data directory")
    
    p.add("--text_pat", type=str, required=True, help="format pattern for text file")
    p.add("--train_pat", type=str, required=False, default=None, help="format pattern for train ids")
    p.add("--src_pat", type=str, required=False, default=None, help="format pattern for source/prompt text")
    
    p.add("--test_pat", type=str, required=False, default=None, help="format pattern for test ids")
    p.add("--test_cut", type=float, default=0.2, required=False, help="")
    p.add("--save_test", action='store_true', help="")
    p.add("--load_test", action='store_true', help="")
    
    p.add("--valid_pat", type=str, required=False, default=None, help="format pattern for valid ids")
    p.add("--valid_cut", type=float, default=0.1, required=False, help="")
    p.add("--save_valid", action='store_true', help="")
    p.add("--load_valid", action='store_true', help="")
    
    p.add("--vocab_file", type=str, required=False, help="The path to the vocab file")
    p.add("--embed_path", type=str, required=False, help="The path to the glove embeddings")
    p.add("-e", "--embed_dim", type=int, default=50, help="Embeddings dimension (default=50)")
    p.add("--min_word_count", type=int, default=2, help="Min word frequency")
    
    p.add("-b", "--batch_size", type=int, default=32, help="Batch size (default=32)")
    p.add("--tensor_vol", type=int, default=0, help="max input tensor volume")
    
    p.add("--learning_rate", type=float, default=0.001, help="")
    p.add("--learning_rates", type=str, default=None, help="")
    
    p.add("--lr_decay", type=float, default=None, help="")
    p.add("--dropout", type=float, default=0.0, help="The dropout probability. To disable, give a negative number (default=0.5)")
    p.add("--rnn_dropout", action='store_true', help="")
    p.add("--drop_sign", type=float, default=1.0, help="")
    p.add("--max_grad_norm", type=float, default=1000.0, required=False, help="")
    p.add("--epochs", type=int, default=100, help="Number of epochs (default=50)")
    p.add("--optimizer", type=str, default='adam', required=False, help="optimizer")
    
    p.add("--min_cut", type=float, default=1.0, required=False, help="")
    p.add("--sample_upto", type=int, default=100000, help="")
    p.add("--model_std", type=float, default=None, required=False, help="")
    p.add("--epoch_unfreeze_filt", type=int, default=0, help="")
    p.add("--epoch_unfreeze_emb", type=int, default=0, help="")
    p.add("--epoch_unfreeze_word", type=int, default=0, help="")
    
    p.add("-r", "--rnn_size", type=int, default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
    p.add("-u", "--rnn_cell", type=str, default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
    p.add("-rl","--rnn_layers", type=int, default=2, help='number of layers in the LSTM')#2
    p.add("--bidirectional", action='store_true', help="")
    p.add("--train_initial_state", action='store_true', help="")
    
    p.add("--rnn_new", action='store_true', help="")
    p.add("--sparse_words", action='store_true', help="")

    p.add("--char_embed_size", type=int, metavar='<int>', default=15, help='dimensionality of character embeddings')
    p.add("-kw", "--kernel_widths", type=str, metavar='<str>', default='[1,2,3,4,5,6,7]', help="")
    p.add("-kf", "--kernel_features", type=str, metavar='<str>', default='[50,100,150,200,200,200,200]', help="")
    p.add("--char_embed_chkpt", type=str, metavar='<str>', default=None, help="")
    
    p.add("--embed_type", type=str, default='word', required=True, help="word|char")
    
    p.add("--trim_words", action='store_true', help="")
    p.add("--trim_chars", action='store_true', help="")
    
    p.add("--split_sentences", action='store_true', help="")
    
    p.add("--spad", type=str, metavar='<str>', default='pre', help="")
    p.add("--wpad", type=str, metavar='<str>', default='post', help="")
    p.add("--cpad", type=str, metavar='<str>', default='post', help="")
    
    p.add("--mean_pool", action='store_true', help="")
    p.add("--skip_connections", action='store_true', help="")
    p.add("--peepholes", action='store_true', help="")
    p.add("--tokenize", action='store_true', help="")
    p.add("--no_shuffle", action='store_true', help="")
    p.add("--keep_unk", action='store_true', help="")
    p.add("--spell_corr", action='store_true', help="")
    p.add("--no_clean", action='store_true', help="")
    
    p.add("--rhn_highway_layers", type=int, default=3, help="Number of highway layers in rhn")
    p.add("--rhn_inputs", action='store_true', help="")
    p.add("--rhn_kronecker", action='store_true', help="")
    
    p.add("--forget_bias", type=float, default=0.0, required=False, help="")
    p.add("--max_text_length", type=int, default=None, help="Max words in essay to consider")
    
    p.add("--rnn_sizes", type=str, default="'[100,100]'", help="")
    p.add("--rnn_cells", type=str, default="\"['gru','gru']\"", help="")
    p.add("--rnn_bis", type=str, default="'[True,True]'", help="")
    p.add("--pads", type=str, default="\"['post','pre']\"", help="")
          
    p.add("--attn_type", type=int, default=-1, help="")
    p.add("--attn_size", type=int, default=0, help="attention layer size")
    p.add("--attn_std", type=float, default=None, help="attention std init")
    p.add("--attn_depth", type=int, default=1, help="")
    p.add("--attn_temp", type=float, default=1., help="")
    p.add("--attn_coef", type=float, default=0., help="2D attention penalty coef")
    p.add("--attn_sizes", type=str, default="'[100,100]'", help="")
    p.add("--attn_depths", type=str, default="'[1,1]'", help="")
    p.add("--attn_temps", type=str, default="'[1,1]'", help="")
    p.add("--attn_vis", action='store_true', help="")
    
    p.add("--rand_seed", type=int, default=0, help="")
    p.add("--print_every", type=int, default=10, help="")
    p.add("-l", "--loss", type=str, default='mse', help="Loss function (mse|kappa) (default=mse)")
    
    p.add("--model_b", type=float, default=None, help="")
    p.add("--attn_b", type=float, default=None, help="")
    
    p.add("--bucket", type=float, default=None, help="")
    p.add("--tensorboard", action='store_true', help="")
    p.add("--print_shapes", action='store_true', help="")
    
    p.add("--ets18", action='store_true', help="")
    p.add("--new_split", action='store_true', help="")
    p.add("--augment", action='store_true', help="")

    
#     p.add("--maxlen", type=int, default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
#     p.add("--aggregation", type=str, default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
#     p.add("--num_highway_layers", type=int, default=0, help="Number of highway layers")
#     p.add("-a", "--algorithm", type=str, default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
#     p.add("-l", "--loss", type=str, default='kappa', help="Loss function (mse|kappa|mae) (default=kappa)")
#     p.add("-v", "--vocab_size", type=int, default=4000, help="Vocab size (default=4000)")
#     p.add("-c", "--cnn_dim", type=int, default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
#     p.add("-w", "--cnn_window_size", type=int, default=3, help="CNN window size. (default=3)")
#     p.add("--stack", type=int, default=1, help="how deep to stack core RNN")
#     p.add("--skip_emb_preload", action='store_true', help="Skip preloading embeddings")
#     p.add("--tokenize_old", action='store_true', help="use old tokenizer")
#     p.add("--run_mode", type=str, default='train', required=False, help="train/valid")
#     p.add("--vocab_path", type=str, help="(Optional) The path to the existing vocab file (*.pkl)")
#     p.add("--skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")

#     dir_path = os.path.dirname(os.path.realpath(__file__))# get path of this file
#     p.add("--code_dir", type=str, default=dir_path, required=False, help="The path to the code directory")
    
    return p


if __name__ == "__main__":
    
    from config import config
    # from nlp.util import config

    parser = get_parser()
    
    config_file = None
    #config_file = 'chkpt/mod1_650-20/model.conf'
    #config_file = 'config/ats.conf'
    config_file = 'config/mode.conf'
    
    FLAGS = config.get_config(config_file=config_file, parser=parser)
    
    pprint.pprint(FLAGS)
