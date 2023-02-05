import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from PadCollate import PadCollate
from dataset import Dataset
from losses import RankListWiseLoss , QPPLoss
import torch
from model import M_QPPF
from transformers import get_linear_schedule_with_warmup
import time
from utils import format_time
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")



def train(config):
    tokenizer = BertTokenizer.from_pretrained(config['bertModel'])
    model = M_QPPF(config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=config['epsilon_parameter'])

    random.seed(config['seed_val'])
    np.random.seed(config['seed_val'])
    torch.manual_seed(config['seed_val'])
    torch.cuda.manual_seed_all(config['seed_val'])

    datasetTrain = Dataset(dataPath=config['dataPath'], DOCperQuery=config['Doc_per_query'], toknzerName=config['bertModel'], phase='train')

    print('{:>5,} training samples'.format(len(datasetTrain)))

    trainDataloader = DataLoader(dataset=datasetTrain, batch_size=config['batch'], shuffle=True,
                                 collate_fn=PadCollate(tokenizer.pad_token_id, tokenizer.pad_token_type_id))


    total_steps = len(trainDataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['num_warmup_steps'], num_training_steps=total_steps)

    total_t0 = time.time()

    for epoch_i in range(0, config['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config['epochs']))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(trainDataloader):
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(trainDataloader), elapsed))
            inputs, query, docSet, relevance_grades, MRRScore = batch
            bsz, gsz, _ = inputs["input_ids"].size()

            inputs = {
                k: v.view(bsz * gsz, -1)
                for k, v in inputs.items()
                }
            inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs.get("token_type_ids")
            }
            model.zero_grad()
            rank_logits, qpp_logits = model.forward(inputs['input_ids'].to(device), inputs['attention_mask'].to(device),
                                                    inputs['token_type_ids'].to(device))

            ranker_loss = RankListWiseLoss(device).loss(rank_logits, relevance_grades)
            qpp_loss = QPPLoss(device).loss(qpp_logits, MRRScore)
            total_loss = ranker_loss + qpp_loss
            total_train_loss += total_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(trainDataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        torch.save(model, config['outputPath'] + 'M-QPPF' + str(epoch_i) + '.model')

    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

if __name__ == '__main__':
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)  # Reading the file
        print(config)
        jsonfile.close()
    train(config)