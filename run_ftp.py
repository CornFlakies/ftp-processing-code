import yaml
import argparse
from ftp_class import FtpClass

if __name__ == '__main__':
    '''
    If you run this from Spyder you should replace the add_arguments things in argparse
    with the relevant directories and stuff and then run it
    '''

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('yaml_file', type=str)
    args = argparser.parse_args()

    param = yaml.load(
                    open(args.yaml_file), Loader=yaml.FullLoader
    )
    
    ftpclass = FtpClass()
    ftpclass.run_ftp(param)
