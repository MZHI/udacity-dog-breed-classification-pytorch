import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser('''Train classification model''')
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to folder where train/val/test directories exist")
    parser.add_argument('--checkpoints-dir', type=str, required=True,
                        help="Path to root checkpoints folder")
    parser.add_argument('--device', type=int, default=0,
                        help="ID of CUDA device, to specify device")
    parser.add_argument('--log-path')
    parser.add_argument('--batch-size')
    parser.add_argument('--num-epochs')
    parser.add_argument('--early-stopping', type=int, default=10)
    parser.add_argument('--num-workers', required=False, default=1)
    parser.add_argument('--num-classes', required=False)
    parser.add_argument('--mean')
    parser.add_argument('--std')
    parser.add_argument('--lr')
    parser.add_argument('--optim', type=str, required=False, default='SGD',
                        help="Type of optimizer. Select from: [SGD, Adam]")
    parser.add_argument('--momentum', type=float, required=False, default=0.9)
    parser.add_argument('--model-type', type=str, required=True,
                        help="Type of network model. Select from: [Base, AlexNet]")
    parser.add_argument('--prefix', type=str, required=False)
    parser.add_argument('--sheduler-patience', type=int, default=None)
    parser.add_argument('--scheduler-factor', type=float, default=0.5)
    parser.add_argument('--scheduler-cooldown', type=int, default=2)
    parser.add_argument('--save-last', type=int, default=1)
    parser.add_argument('--resume-train', type=int, default=False,
                        help="Is resume specific experiment or train from scratch")

    args = parser.parse_args()
    return args


def main(args):
    use_cuda = torch.cuda.is_available()


if __name__ == "__main__":
    args = get_args()
    main(args)