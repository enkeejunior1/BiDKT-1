from models.bidkt import Bidkt
from models.bert4kt_plus import Bert4ktPlus

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
            use_gelu=config.use_gelu,
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
    else:
        print("Wrong model_name was used...")

    return model