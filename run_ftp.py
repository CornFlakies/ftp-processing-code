import yaml
import argparse
from ftp_class import FtpClass

if __name__ == '__main__':
    '''
    If you run this from Spyder you should just directly input the yaml_file path
    '''

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('yaml_file', type=str)
    args = argparser.parse_args()

    param = yaml.load(
                    open(args.yaml_file), Loader=yaml.FullLoader
    )
    
    ftpclass = FtpClass()
    ftpclass.run_ftp(param)
