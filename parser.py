'''
hj04143@gmail.com
https://github.com/changhyeonnam/STRMF
'''
import argparse

parser = argparse.ArgumentParser(description="Seoul Tourism Recommendation using Matrix Factorzation")
parser.add_argument('-e','--epochs',default=1,type=int,help="epochs")
parser.add_argument('-b','--batch_size', default=512, type= int, help='batch size')
parser.add_argument('-lr','--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('-op','--optimizer',default='Adam',type=str, help='optimizer')
parser.add_argument('-topk','--topk', default=20, type = int, help='top k recommends')
parser.add_argument('-save','--save_model', default='True', type = str, help='save model')
parser.add_argument('-shu','--shuffle', default='True', type = str, help='shuffle')
parser.add_argument('-tar','--target', default= 'v', type = str, help='select v(vistor) or c(congestion)')
args = parser.parse_args()
