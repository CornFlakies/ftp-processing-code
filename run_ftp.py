import argparse
from ftp_class import FtpClass

if __name__ == '__main__':
    '''
    If you run this from Spyder you should replace the add_arguments things in argparse
    with the relevant directories and stuff and then run it
    '''

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=str)
    argparser.add_argument('background_folder', type=str)
    argparser.add_argument('reference_folder', type=str)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args() 

    ftpclass = FtpClass()
    ftpclass.run_ftp(args.output_folder, args.background_folder, args.reference_folder, args.input_folder)
