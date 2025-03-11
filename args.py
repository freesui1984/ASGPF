import argparse


def get_args():
    parser = argparse.ArgumentParser('Train DCRNN on TUH data.')

    # General args
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument(
        '--load_model_path',
        type=str,
        default=None,
        help='Model checkpoint to start training/testing from.')
    parser.add_argument('--do_train',
                        default=False,
                        action='store_true',
                        help='Whether perform training.')
    parser.add_argument('--rand_seed',
                        type=int,
                        default=123,
                        help='Random seed.')
    parser.add_argument(
        '--task',
        type=str,
        default='detection',
        choices=(
            'detection',
            'classification',
            'SS-pre-training'),
        help="Seizure detection, seizure type classification, \
                            or SS-pre-training.")
    parser.add_argument(
        '--ssl_model',
        type=str,
        default='asgpf',
        choices=(
            'dcrnn',
            'asgpf'),
        help="dcgru or asgpf.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help='Whether to fine-tune pre-trained model.')

    # Input args
    parser.add_argument(
        '--graph_type',
        choices=(
            'individual',
            'combined',
            'aec'),
        default='individual',
        help='Whether use individual graphs (cross-correlation) or combined graph (distance).')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=12,
                        help='Maximum sequence length in seconds.')
    parser.add_argument('--num_edge_types',
                        type=int,
                        default=1,
                        help='Types of edges in adjacency matrices.')
    parser.add_argument('--n_steps',
                        type=int,
                        default=5,
                        help='GGNN iteration steps.')
    parser.add_argument(
        '--output_seq_len',
        type=int,
        default=12,
        help='Output seq length for SS pre-training, in seconds.')
    parser.add_argument('--time_step_size',
                        type=int,
                        default=1,
                        help='Time step size in seconds.')
    parser.add_argument('--input_dir',
                        type=str,
                        default=None,
                        help='Dir to resampled EEG signals (.h5 files).')
    parser.add_argument('--raw_data_dir',
                        type=str,
                        default=None,
                        help='Dir to TUH data with raw EEG signals.')
    parser.add_argument('--preproc_dir',
                        type=str,
                        default=None,
                        help='Dir to preprocessed (Fourier transformed) data.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=18,
        help='Top-k neighbors of each node to keep, for graph sparsity.')

    # Model args
    parser.add_argument("--model_name", type=str, default="dcrnn", choices=("dcrnn", "lstm", "densecnn", "cnnlstm", "sglc"))
    parser.add_argument("--dataset_name", type=str, default="tusz",
                        choices=("tusz"))
    parser.add_argument('--num_nodes',
                        type=int,
                        default=19,
                        help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers',
                        type=int,
                        default=2,
                        help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument(
        '--pretrained_num_rnn_layers',
        type=int,
        default=2,
        help='Number of RNN layers in encoder and decoder for SS pre-training.')
    parser.add_argument('--rnn_units',
                        type=int,
                        default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dcgru_activation',
                        type=str,
                        choices=('relu', 'tanh'),
                        default='tanh',
                        help='Nonlinear activation used in DCGRU cells.')
    parser.add_argument('--gcgru_activation',
                        type=str,
                        choices=('relu', 'tanh'),
                        default='tanh',
                        help='Nonlinear activation used in GCGRU cells.')
    parser.add_argument('--input_dim',
                        type=int,
                        default=100,
                        help='Input seq feature dim.')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=1,
        help='Number of classes for seizure detection/classification.')
    parser.add_argument('--output_dim',
                        type=int,
                        default=100,
                        help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step',
                        type=int,
                        default=2,
                        help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps',
                        type=int,
                        default=3000,
                        help='Scheduled sampling decay steps.')
    parser.add_argument(
        '--use_curriculum_learning',
        default=False,
        action='store_true',
        help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument(
        '--use_fft',
        default=False,
        action='store_true',
        help='Whether the input data is Fourier transformed EEG signal or raw EEG.')

    # Training/test args
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=48,
                        help='Training batch size.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=48,
                        help='Dev/test batch size.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Dropout rate for dropout layer before final FC.')
    parser.add_argument('--eval_every',
                        type=int,
                        default=1,
                        help='Evaluate on dev set every x epoch.')
    # GraphLearn args
    parser.add_argument(
        '--use_graph_learning',
        default=False,
        action='store_true',
        help='Whether to use graph learning.')
    parser.add_argument(
        '--graph_include_self',
        default=False,
        action='store_true',
        help='Whether include graph self.')
    parser.add_argument('--eps_adj',
                        type=float,
                        default=8e-3,
                        help='Graph learning threshold.')
    parser.add_argument('--graph_skip_conn',
                        type=float,
                        default=0.3,
                        help='Preserve the ratio of the original adj.')
    parser.add_argument('--update_adj_ratio',
                        type=float,
                        default=0.5,
                        help='Update the ratio of the graph.')
    parser.add_argument('--smoothness_ratio',
                        type=float,
                        default=0.5,
                        help='Smoothness loss of balance parameter.')
    parser.add_argument('--degree_ratio',
                        type=float,
                        default=0.01,
                        help='Degree loss of balance parameter.')
    parser.add_argument('--sparsity_ratio',
                        type=float,
                        default=0.3,
                        help='Sparse loss of balance parameter.')
    parser.add_argument('--max_iter',
                        type=int,
                        default=10,
                        help='Iterative rounds of graph learning.')
    parser.add_argument('--graph_learn_hidden_size',
                        type=int,
                        default=70,
                        help='Hidden size of graph learning.')
    parser.add_argument('--graph_learn_epsilon',
                        type=float,
                        default=0.3,
                        help='Graph learning threshold for mask.')
    parser.add_argument('--graph_learn_ratio',
                        type=float,
                        default=0.3,
                        help='Graph learning ratio.')
    parser.add_argument('--graph_learn_num_pers',
                        type=int,
                        default=8,
                        help='Number of attention heads.')


    parser.add_argument(
        '--metric_name',
        type=str,
        default='auroc',
        choices=(
            'F1',
            'acc',
            'loss',
            'auroc'),
        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr_init',
                        type=float,
                        default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=0,
                        # default=5e-4,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs for training.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='weighted',
                        help='weighted, micro or macro.')
    parser.add_argument('--data_augment',
                        default=False,
                        action='store_true',
                        help='Whether perform data augmentation.')
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs of patience before early stopping.')

    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F1', 'acc', 'auroc'):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError(
            'Unrecognized metric name: "{}"'.format(
                args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and not(args.do_train):
        raise ValueError(
            'For evaluation only, please provide trained model checkpoint in argument load_model_path.')

    # filter type
    if args.graph_type == "individual":
        # args.filter_type = "dual_random_walk"
        args.filter_type = "laplacian_1"
    if args.graph_type == "combined":
        args.filter_type = "laplacian"
    if args.graph_type == "aec":
        args.filter_type = "laplacian"

    return args
