from models.bidkt import Bidkt
from models.bert4kt_plus import Bert4ktPlus
from models.bert4kt_rasch import Bert4ktRasch
from models.albert4kt_plus import ALBert4ktPlus
from models.bcaa_kt import BcaaKt
from models.ma_bert4kt_plus import MonotonicBert4ktPlus
from models.nma_bert4kt_dualenc_kr import NmaBert4ktDualencKr
from models.ma_bert4kt_dualenc_kr import MaBert4ktDualencKr
from models.bigbird4kt_plus import Bigbird4ktPlus
from models.bert4kt_plus_time import Bert4ktPlusTime
from models.convbert4kt_plus import ConvBert4ktPlus
from models.monoconvbert4kt_plus import MonoConvBert4ktPlus

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
    elif config.model_name == "ma_bert4kt_dualenc_kr":
        model = MaBert4ktDualencKr(
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
        model = BcaaKt(
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
    elif config.model_name == "bigbird4kt_plus":
        model = Bigbird4ktPlus(
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
            config=config,
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model_name == "bert4kt_plus_time":
        model = Bert4ktPlusTime(
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
    elif config.model_name == "convbert4kt_plus":
        model = ConvBert4ktPlus(
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
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model_name == "monoconvbert4kt_plus":
        model = MonoConvBert4ktPlus(
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
            dropout_p=config.dropout_p
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model