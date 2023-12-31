import argparse
from train import train
from test import test

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()

    args.add_argument('-lr', type=float, default=1e-4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-c1', type=float, default=0.5)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-clip', type=float, default=0.1)
    args.add_argument('-minibatch_size', type=int, default=96)
    args.add_argument('-batch_size', type=int, default=128)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-train', default='True', choices=('True','False'))
    args.add_argument('-env', default='MiniWorld-FourRooms-v0', choices=('MiniWorld-FourRooms-v0','MiniWorld-OneRoom'))
    #args.add_argument('-env', default='MiniWorld-OneRoom', choices=('MiniWorld-FourRooms-v0','MiniWorld-OneRoom'))
    args.add_argument('-episodes', type=int, default=3000)
    args.add_argument('-lam', type=float, default=1.0)
    args.add_argument('-in_channels', type=int, default=4)
    args.add_argument('-n_outputs', type=int, default=3)
    args.add_argument('-num_processes', type=int, default=12)
    args.add_argument('-alpha', type=float, default=0.1)
    args.add_argument('-beta', type=float, default=0.2)
  
    arguments = args.parse_args()

    if(arguments.train == 'True'):
        print('Modo: Entrenamiento')
        train(arguments)
    else:
        print('Modo: Testeo')
        test(arguments)