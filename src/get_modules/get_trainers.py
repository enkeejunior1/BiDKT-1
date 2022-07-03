from trainers.bidkt_trainer import BidktTrainer
from trainers.bert4kt_plus_trainer import Bert4ktPlusTrainer
from trainers.bert4kt_rasch_trainer import Bert4ktRaschTrainer
from trainers.albert4kt_plus_trainer import ALBert4ktPlusTrainer
from trainers.ma_bert4kt_plus_trainer import MonotonicBert4ktPlusTrainer
from trainers.bcaa_kt_trainer import BcaaKtTrainer
from trainers.nma_bert4kt_dualenc_kr_trainer import NmaBert4ktDualencKrTrainer
from trainers.ma_bert4kt_dualenc_kr_trainer import MaBert4ktDualencKrTrainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    #trainer 실행
    if config.model_name == "bidkt":
        trainer = BidktTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "bert4kt_plus":
        trainer = Bert4ktPlusTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "bert4kt_rasch":
        trainer = Bert4ktRaschTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "albert4kt_plus":
        trainer = ALBert4ktPlusTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "ma_bert4kt_plus":
        trainer = MonotonicBert4ktPlusTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "nma_bert4kt_dualenc_kr":
        trainer = NmaBert4ktDualencKrTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "ma_bert4kt_dualenc_kr":
        trainer = MaBert4ktDualencKrTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    elif config.model_name == "bcaa_kt":
        trainer = BcaaKtTrainer(
            model=model,
            optimizer=optimizer,
            n_epochs=config.n_epochs,
            device=device,
            num_q=num_q,
            crit=crit,
            max_seq_len=config.max_seq_len,
            grad_acc=config.grad_acc,
            grad_acc_iter=config.grad_acc_iter
        )
    else:
        print("wrong model was choosed..")

    return trainer
