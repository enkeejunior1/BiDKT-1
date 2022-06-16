import pandas as pd
import numpy as np
import json

import torch
from torch.nn.utils.rnn import pad_sequence

from torch.optim import SGD, Adam

from torch.nn.functional import binary_cross_entropy

def collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []

    for q_seq, r_seq in batch:
        q_seqs.append(torch.Tensor(q_seq)) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개
        r_seqs.append(torch.Tensor(r_seq)) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개

    #가장 길이가 긴 seqs를 기준으로 길이를 맞추고, 길이를 맞추기 위해 그 자리에는 -1(pad_val)을 넣어줌
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )

    #각 원소가 -1이 아니면 Ture, -1이면 False로 값을 채움
    #mask_seqs는 실제로 문항이 있는 경우만을 추출하기 위해 사용됨(실제 문항이 있다면, True, 아니면 False, pad_val은 전체 길이를 맞춰주기 위해 사용됨)
    mask_seqs = (q_seqs != pad_val)

    #즉 전체를 qshft_seqs의 -1이 아닌 갯수만큼은 true(1)을 곱해서 원래 값을 부여하고, 아닌 것은 False(0)을 곱해서 0으로 만듦
    q_seqs, r_seqs = q_seqs * mask_seqs, r_seqs * mask_seqs

    return q_seqs, r_seqs, mask_seqs
    #|q_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|r_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|mask_seqs| = (batch_size, maximum_sequence_length_in_the_batch)

#get_optimizer 정의
def get_optimizers(model, config):
    if config.optimizer == "adam":
        optimizer = Adam(model.parameters(), config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = SGD(model.parameters(), config.learning_rate)
    #-> 추가적인 optimizer 설정
    else:
        print("Wrong optimizer was used...")

    return optimizer

#get_crit 정의
def get_crits(config):
    if config.crit == "binary_cross_entropy":
        crit = binary_cross_entropy
    #-> 추가적인 criterion 설정
    else:
        print("Wrong criterion was used...")

    return crit

#recoder
def recorder(train_auc_scores, test_auc_scores, highest_auc_score, record_time, config):

    model_fn = config.model_fn

    record_path = "../score_records/"

    # config를 json파일로 저장
    config_record_path = record_path + str(round(highest_auc_score, 6)) + "_" + record_time + "_" + model_fn + "_config" + ".json"
    config_dic = vars(config)
    with open(config_record_path,'w') as f:
        json.dump(config_dic, f)

    # train, test auc와 highest_auc_socre를 csv로 저장
    auc_record_path = record_path + str(round(highest_auc_score, 6)) + "_" + model_fn + "_auc" + ".csv"
    auc_scores = [train_auc + test_auc for train_auc, test_auc in zip(train_auc_scores, test_auc_scores)]
    auc_scores.append(['hightest_auc', 0, highest_auc_score])
    auc_df = pd.DataFrame(auc_scores, columns=['epochs', 'train_auc', 'test_auc'])
    auc_df.set_index('epochs', inplace=True)
    auc_df.to_csv(auc_record_path)