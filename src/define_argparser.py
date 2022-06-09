import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()

    # model_file_name
    p.add_argument('--model_fn', required=True)

    # basic arguments
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8) #test_ratio는 train_ratio에 따라 정해지도록 설정
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--learning_rate', type=int, default = 0.001)

    # model, opt, dataset, crit arguments
    p.add_argument('--model_name', type=str, default='bidkt')
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--dataset_name', type=str, default = 'assist2015')
    p.add_argument('--crit', type=str, default = 'binary_cross_entropy')

    # models' special arguments
    # bidkt's arguments
    p.add_argument('--max_seq_len', type=int, default=100)
    p.add_argument('--num_encoder', type=int, default=12)
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--num_head', type=int, default=16) #hidden_size가 나누어지도록
    p.add_argument('--output_size', type=int, default=1) #정답일 확률값만 알면 되므로, 0~1사이의 값
    p.add_argument('--dropout_p', type=int, default=.1)

    config = p.parse_args()

    return config