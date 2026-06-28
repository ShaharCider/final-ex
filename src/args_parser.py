import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Args for ex3: Fourier Transform and Individual Alpha Frequency',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    
    # General parameters

    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', # TODO: set the default data directory
                        help='Directory containing the data')
    
    return parser.parse_args()