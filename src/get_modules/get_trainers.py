from trainers.bidkt_trainer import BidktTrainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    #trainer 실행
    if config.model_name == "bidkt":
        trainer = BidktTrainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit
        )
    else:
        print("wrong model was choosed..")

    return trainer
