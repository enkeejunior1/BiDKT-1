import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint

# 15%에 <MASK> 적용
def Mlm4BertTrain(r_seqs):
    # |r_seqs| = (bs, n)
    
    # 15%에 해당하는 중복되지 않는 인덱스
    r_seqs_len = r_seqs.size(1) #r_seqs_len = 100
    mlm_idx = np.random.choice(r_seqs_len, int(r_seqs_len*0.15), replace=False)

    # 그냥 pad 신경쓰지 말고, mask에 해당되는 곳을 랜덤으로 정하기
    for idx in mlm_idx:
        if random() < 0.8: # 80%는 <MASK>
            r_seqs[:, idx] = 2 # <mask>는 2로 표시하기
        elif random() < 0.5: # 10%는 0과 1의 random 값으로 넣기
            r_seqs[:, idx] = randint(0, 1)

    # 명시적으로 변수 선언
    mlm_r_seqs = r_seqs

    # mlm_r_seqs: mask가 씌워진 r_seqs
    # mlm_idx: mask가 씌워진 idx 값
    return mlm_r_seqs, mlm_idx
    # |mlm_r_seqs| = (bs, n)

# real_r_seqs 마지막에 <MASK>
def Mlm4BertTest(r_seqs, mask_seqs):
    #|r_seqs| = (bs, n)

    # mask_seqs를 통해 <PAD>가 아닌 mask 길이를 알아내기
    masked_index = torch.sum(mask_seqs)
    # ex) r의 실제 길이가 3이면, 이것도 3

    mlm_idx = []

    for idx in range(r_seqs):
        mlm_idx.append(masked_index[idx] - 1)
        # r_seqs를 하나씩 불러오고, 마지막 차원의 값을 마스크값(2)로 변경
        r_seqs[idx][masked_index[idx] - 1] = 2

    #좀 더 명시적으로 보이도록 변수명 변경
    mlm_r_seqs = r_seqs
    mlm_idx = np.array(mlm_idx) #np array로 변경

    return mlm_r_seqs, mlm_idx
    # |mlm_r_seqs| = (bs, n)

class BidktTrainer():

    def __init__(self, model, optimizer, n_epochs, device, num_q, crit):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit

        print(self.model)
    
    def _train(self, train_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        for data in tqdm(train_loader):
            self.model.train()
            q_seqs, r_seqs, _, _, mask_seqs = data

            q_seqs = q_seqs.to(self.device) #|q_seqs| = (bs, sq)
            r_seqs = r_seqs.to(self.device) #|r_seqs| = (bs, sq)
            mask_seqs = mask_seqs.to(self.device) #|mask_seqs| = (bs, sq)

            mlm_r_seqs, mlm_idx = Mlm4BertTrain(r_seqs)
            mlm_r_seqs = mlm_r_seqs.to(self.device)

            y_hat = self.model(
                q_seqs.long(), 
                mlm_r_seqs.long(), # train을 위한 mlm된 r_seqs
                mask_seqs.long() # attn_mask
            )
            # |y_hat| = (bs, n, output_size)

            # 계산해서 나온 값 중에서 mlm_idx에 해당하는 부분의 값만 활용해서 성능 확인하기
            #####



            #####

            print("mlm_idx", mlm_idx)

            print("y_hat", y_hat) # 전부 nan값이 나옴, 학습이 안되고 있음

            # 여기서 mask를 masked_idx 값을 제외한 모든 부분에 씌워야 함

            # 예측값과 실제값
            y_hat = torch.masked_select(y_hat, mask_seqs)
            #|y_hat| = (bs, sq)
            correct = torch.masked_select(r_seqs, mask_seqs)
            #|correct| = (bs, sq)

            self.optimizer.zero_grad()
            #self.crit은 binary_cross_entropy
            loss = self.crit(y_hat, correct) #|loss|

            loss.backward() #여기서 error
            self.optimizer.step()

            y_trues.append(correct)
            y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy() #|y_tures| = () -> [0. 0. 0. ... 1. 1. 1.]
        y_scores = torch.cat(y_scores).detach().cpu().numpy() #|y_scores| = () ->  tensor(0.5552)

        auc_score += metrics.roc_auc_score( y_trues, y_scores ) #|metrics.roc_auc_score( y_trues, y_scores )| = () -> 0.6203433289463159

        return auc_score

    def _test(self, test_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, _, _, mask_seqs = data #collate에 정의된 데이터가 나옴
                #|r_seqs| = (bs, sq)
                test_masked_r_seqs = Mlm4BertTest(r_seqs)
                test_masked_r_seqs = test_masked_r_seqs.to(self.device)

                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)

                mask_seqs = mask_seqs.to(self.device)

                y_hat = self.model( q_seqs.long(), test_masked_r_seqs.long() )

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(r_seqs, mask_seqs)

                y_trues.append(correct)
                y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        return auc_score, y_trues, y_scores

    def train(self, train_loader, test_loader):
        
        highest_auc_score = 0
        best_model = None
        #시각화를 위해 받아오기
        y_true_record, y_score_record = [], []

        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            train_auc_score = self._train(train_loader)
            test_auc_score, y_trues, y_scores = self._test(test_loader)

            if test_auc_score >= highest_auc_score:
                highest_auc_score = test_auc_score
                best_model = deepcopy(self.model.state_dict())
                y_true_record, y_score_record = y_trues, y_scores

            print("Epoch(%d/%d) result: train_auc_score=%.4f  test_auc_score=%.4f  highest_auc_score=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_auc_score,
                test_auc_score,
                highest_auc_score,
            ))

        print("\n")
        print("The Highest_Auc_Score in Training Session is %.4f" % (
                highest_auc_score,
            ))
        print("\n")
        
        # 가장 최고의 모델 복구    
        self.model.load_state_dict(best_model)

        return y_true_record, y_score_record, highest_auc_score