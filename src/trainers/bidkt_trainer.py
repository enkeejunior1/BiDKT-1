import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint

# 15%에 <MASK> 적용
def Mlm4BertTrain(r_seqs, mask_seqs):
    # |r_seqs| = (bs, n)
    
    mlm_idxs = []

    # 그냥 pad 신경쓰지 말고, mask에 해당되는 곳을 랜덤으로 정하기
    for r_seq, mask_seq in zip(r_seqs, mask_seqs):
         # 중복되지 않는 인덱스의 수
        r_len = r_seq.size(0)
        # |r_len| = (n, )

        mask_len = mask_seq.sum()
        # |mask_len| = (n, )

        # 실제 r의 길이를 구함
        r_seq_len = r_len - mask_len
        # |r_seq_len| = (n, )

        r_seq_len = r_seq_len.cpu()

        # 실제 r의 길이인 r_seq_len의 15%에 해당하는 index 추출
        mlm_idx = np.random.choice(r_seq_len, int(r_seq_len*0.15), replace=False)
        # |mlm_idx| = (n, ), default = (15, )

        for idx in mlm_idx:
            if random() < 0.8: # 15% 중 80%는 <MASK>
                r_seq[idx] = 2 # <MASK>는 2로 표시하기
            elif random() < 0.5: # 15% 중 10%는 0과 1의 random 값으로 넣기
                r_seq[idx] = randint(0, 1)
            # 15% 중 10%는 원래 값 그대로

        # mask 만들기
        # r_len의 길이(전체 길이)만큼 zero vector로 만듦
        mlm_zeros = np.zeros(shape=(r_len, ))
        # mlm_idx에 해당하는 곳에 1을 넣음
        mlm_zeros[mlm_idx] = 1
        # mlm_idxs에 값을 더함
        mlm_idxs.append(mlm_zeros)
        
    # 명시적으로 변수 선언
    mlm_r_seqs = r_seqs
    # mlm_idxs를 np.array로 바꾸고, bool type으로 변경(이후 mask를 씌우기 위해)

    mlm_idxs = torch.BoolTensor(mlm_idxs)

    # mlm_r_seqs: mask가 씌워진 r_seqs
    # mlm_idx: mask가 씌워진 idx 값
    return mlm_r_seqs, mlm_idxs
    # |mlm_r_seqs| = (bs, n)
    # |mask_seqs| = (bs, n)

# real_r_seqs 마지막에 <MASK>
def Mlm4BertTest(r_seqs, mask_seqs):
    #|r_seqs| = (bs, n)

    mlm_idxs = []

    for r_seq, mask_seq in zip(r_seqs, mask_seqs):
        # mask_seqs를 통해 <PAD>가 아닌 mask 길이를 알아내기
        
        r_len = r_seq.size(0)
        # |r_len| = (n, )

        mask_len = torch.sum(mask_seq)
        # |mask_len| = (n, )

        # r_seqs의 실제값의 마지막 인덱스
        mlm_idx = r_len - mask_len - 1

        # r_seqs의 실제값의 마지막 인덱스를 2(<MASK>)로 바꿔줌
        r_seq[mlm_idx] = 2

        # mask를 위한 bool tensor 만들기
        mlm_zeros = np.zeros(shape=(r_len, ))
        mlm_zeros[mlm_idx] = 1
        mlm_idxs.append(mlm_zeros)
        
    # 명시적으로 보이도록 변수명 변경
    mlm_r_seqs = r_seqs
    # boolTensor로 변경
    mlm_idxs = torch.BoolTensor(mlm_idxs)

    return mlm_r_seqs, mlm_idxs
    # |mlm_r_seqs| = (bs, n)
    # |mask_seqs| = (bs, n)

class BidktTrainer():

    def __init__(self, model, optimizer, n_epochs, device, num_q, crit, max_seq_len):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit
        self.max_seq_len = max_seq_len
    
    def _train(self, train_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        for data in tqdm(train_loader):
            self.model.train()
            q_seqs, r_seqs, _, _, mask_seqs = data

            q_seqs = q_seqs.to(self.device) #|q_seqs| = (bs, n)
            r_seqs = r_seqs.to(self.device) #|r_seqs| = (bs, n)
            mask_seqs = mask_seqs.to(self.device) #|mask_seqs| = (bs, n)

            # correct에서 따로 사용하기 위해 clone 작성
            real_seqs = r_seqs.clone()

            # mlm_r_seqs: r_seqs에 Masked Language Model 구현을 위한 [MASK]를 씌움, [MASK]는 2로 표기 / mlm_idx: [MASK]의 위치
            mlm_r_seqs, mlm_idxs = Mlm4BertTrain(r_seqs, mask_seqs)
            # |mlm_r_seqs| = (bs, n)
            # |mlm_idxs| = (bs, n), True or False가 들어있어야 함

            mlm_r_seqs = mlm_r_seqs.to(self.device)
            mlm_idxs = mlm_idxs.to(self.device)

            y_hat = self.model(
                q_seqs.long(), 
                mlm_r_seqs.long(), # train을 위한 mlm된 r_seqs
                mask_seqs.long() # attn_mask
            ).to(self.device)
            # |y_hat| = (bs, n, output_size=1)

            y_hat = y_hat.squeeze()
             # |y_hat| = (bs, n)

            # 예측값과 실제값
            y_hat = torch.masked_select(y_hat, mlm_idxs)
            #|y_hat| = (bs * n - n_mlm_idxs)
            correct = torch.masked_select(real_seqs, mlm_idxs)
            #|correct| = (bs * n - n_mlm_idxs)

            self.optimizer.zero_grad()
            #self.crit은 binary_cross_entropy
            loss = self.crit(y_hat, correct)
            # |loss| = (1)
            loss.backward()
            self.optimizer.step()

            y_trues.append(correct)
            y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        return auc_score

    def _test(self, test_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, _, _, mask_seqs = data #collate에 정의된 데이터가 나옴
                #|r_seqs| = (bs, sq)
                
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                real_seqs = r_seqs.clone()

                mlm_r_seqs, mlm_idxs = Mlm4BertTest(r_seqs, mask_seqs)

                mlm_r_seqs = mlm_r_seqs.to(self.device)
                mlm_idxs = mlm_idxs.to(self.device)
                # |mlm_r_seqs| = (bs, n)
                # |mlm_idxs| = (bs, n), True or False가 들어있어야 함

                y_hat = self.model(
                    q_seqs.long(),
                    mlm_r_seqs.long(),
                    mask_seqs.long()
                ).to(self.device)

                y_hat = y_hat.squeeze()

                y_hat = torch.masked_select(y_hat, mlm_idxs)
                correct = torch.masked_select(real_seqs, mlm_idxs)

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