import torch
import logging

from machine.tasks import get_task
from machine.trainer import SupervisedTrainer
from machine.loss import NLLLoss
from machine.metrics import SequenceAccuracy

from data import get_iters
from model import get_baseline_model


SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logging():
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, 'INFO'))


NUM_EPOCHS = 10
HIDDEN_SIZE = 128
init_logging()

# Get data
train_iter, valid_iter, test_iters, src, tgt = get_iters()

# Prepare model
baseline_seq2seq = get_baseline_model(src, tgt, HIDDEN_SIZE)
baseline_seq2seq.to(device)

# Prepare training
pad = tgt.vocab.stoi[tgt.pad_token]
losses = [NLLLoss(ignore_index=pad).to(device)]
metrics = [SequenceAccuracy(ignore_index=pad)]
trainer = SupervisedTrainer(expt_dir='runs/models/baseline')

# Train
logging.info("Training")
seq2seq, logs = trainer.train(baseline_seq2seq, train_iter,
                              dev_data=valid_iter,
                              monitor_data=test_iters,
                              num_epochs=NUM_EPOCHS,
                              optimizer='adam',
                              checkpoint_path='runs/models/baseline',
                              losses=losses, metrics=metrics,
                              checkpoint_every=100,
                              print_every=100)
