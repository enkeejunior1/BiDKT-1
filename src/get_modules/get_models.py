from models.bidkt import Bidkt
from models.bert4kt_plus import Bert4ktPlus
from models.bert4kt_rasch import Bert4ktRasch
from models.albert4kt_plus import ALBert4ktPlus
from models.bcaa_kt import BCAA_KT
from models.ma_bert4kt_plus import MonotonicBert4ktPlus
from models.nma_bert4kt_dualenc_kr import NmaBert4ktDualencKr

def get_models(num_q, num_r, num_pid, device, config):

    if config.model_name == "bidkt":
        model = Bidkt(
            num_q=num_q,
            num_r=num_r,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model_name == "bert4kt_plus":
        model = Bert4ktPlus(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model_name == "bert4kt_rasch":
        model = Bert4ktRasch(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model_name == "albert4kt_plus":
        model = ALBert4ktPlus(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model_name == "ma_bert4kt_plus":
        model = MonotonicBert4ktPlus(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model_name == "nma_bert4kt_dualenc_kr":
        model = NmaBert4ktDualencKr(
            num_q=num_q,
            num_r=num_r,
            num_pid=num_pid,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
            num_head=config.num_head,
            num_encoder=config.num_encoder,
            max_seq_len=config.max_seq_len,
            device=device,
            use_leakyrelu=config.use_leakyrelu,
            dropout_p=config.dropout_p,
        ).to(device)
    elif config.model_name == "bcaa_kt":
        model = BCAA_KT(
            n_question=num_q,
            n_pid=num_pid,
            d_model=config.akt_d_model,
            n_blocks=config.akt_n_block,
            kq_same=config.akt_kq_same,
            dropout=config.akt_dropout_p,
            model_type="akt"
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model