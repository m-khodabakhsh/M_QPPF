import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from PadCollate import PadCollate
from dataset import Dataset
import torch
from utils import correlation, save
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")


def test(config):
    tokenizer = BertTokenizer.from_pretrained(config['bertModel'])
    model = torch.load(config['outputPath'] + 'M-QPPF' + str(0) + '.model')


    model.to(device)
    print(next(model.parameters()).device)

    random.seed(config['seed_val'])
    np.random.seed(config['seed_val'])
    torch.manual_seed(config['seed_val'])
    torch.cuda.manual_seed_all(config['seed_val'])

    datasetVal = Dataset(dataPath=config['dataPath'], DOCperQuery=config['Doc_per_query'], toknzerName=config['bertModel'], phase='dev')

    print('{:>5,} validation samples'.format(len(datasetVal)))

    valDataloader = DataLoader(dataset=datasetVal, batch_size=config['batch'], shuffle=True,collate_fn=PadCollate(tokenizer.pad_token_id, tokenizer.pad_token_type_id))

    Run = {}
    QPP = {}
    MRR = {}
    for batch in valDataloader:
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
        with torch.no_grad():
            rank_logits, qpp_logits = model.forward(inputs['input_ids'].to(device), inputs['attention_mask'].to(device),inputs['token_type_ids'].to(device))

        for q in range(0, len(query)):
            for d in range(0, config['Doc_per_query']):
                if query[q] in Run:
                    listRun = Run[query[q]]
                    listRun.append((docSet[q][d], float(rank_logits[q][d].item())))
                    Run[query[q]] = listRun
                else:
                    Run[query[q]] = [(docSet[q][d], float(rank_logits[q][d].item()))]
        for q in range(0, len(query)):
            MRR[query[q]] = float(MRRScore[q][0])
            QPP[query[q]] = float(qpp_logits[q].item())

    pearsonr, pearsonp, kendalltauCorrelation, kendalltauPvalue, spearmanrCorrelation, spearmanrPvalue = correlation(QPP, MRR)

    print("  pearsonrCorrelation: {0:.3f}".format(pearsonr))
    print("  pearsonrPvalue: {0:.3f}".format(pearsonp))
    print("  kendalltauCorrelation: {0:.3f}".format(kendalltauCorrelation))
    print("  kendalltauPvalue: {0:.3f}".format(kendalltauPvalue))
    print("  spearmanrCorrelation: {0:.3f}".format(spearmanrCorrelation))
    print("  spearmanrPvalue: {0:.3f}".format(spearmanrPvalue))


    save(config, Run)

if __name__ == '__main__':
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)  # Reading the file
        print(config)
        jsonfile.close()
    test(config)