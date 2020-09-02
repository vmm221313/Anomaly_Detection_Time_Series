import random
import numpy as np
import pandas as po

import tensorflow as tf

from preprocess import standardize_dataframe, train_val_test_split, make_windows

def main(args):

    # set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    if args.vars == 'uni':
        raise NotImplementedError
    elif args.vars == 'multi':
        df = po.read_csv('multivariate.csv', index=False)

    if args.scale == 'standardize':
        df, mean_std = standardize_dataframe(df)

    df_train, df_val, df_test = train_val_test_split(df)

    # make windows 
    X_train, y_train    = make_windows(df_train)
    X_val, y_val        = make_windows(df_val)
    X_test, y_test      = make_windows(df_test)

    # build model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Multivariate or Univariate 
    parser.add_argument('-vars', type=str, choices=['uni', 'multi'], default='uni')

    # Standardize/Normalize/None
    parser.add_argument('-scale', type=str, choices=['none', 'standardize'], default='standardize')

	# Train-Validation-Testing Distribution 
	parser.add_argument('-train_frac', type=float, default=0.6)	
	parser.add_argument('-val_frac', type=float, default=0.2)	
	parser.add_argument('-test_frac', type=float, default=0.2)	

	parser.add_argument('-train_seq_len', type=int, default=4*24*7)	
	parser.add_argument('-val_seq_len', type=int, default=4*24)	

	# Model to be used
	parser.add_argument('-model', type=str, choices=['nbeats'], default='nbeats')

    # Params for NBEATS
	parser.add_argument('-hidden_dim', type=int, default=100)
	parser.add_argument('-theta_1', type=int, default=96)
	parser.add_argument('-theta_2', type=int, default=96)
	parser.add_argument('-nb_blocks_per_stack', type=int, default=10)
    parser.add_argument('-seasonality', type=int, default=96)

    # Model Training Params
	parser.add_argument('-num_epochs', type=int, default=100)
	parser.add_argument('-batch_size', type=int, default=512)

	args = parser.parse_args()

    main(args)
