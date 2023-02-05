from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from typing import List
from irtools.pad import pad_batch


class Dataset(Dataset):
    def __init__(self, dataPath, DOCperQuery, toknzerName, phase):
        self.dataPath = dataPath
        self.phase = phase
        self.tokenizer = BertTokenizer.from_pretrained(toknzerName)
        self.DOCperQuery=DOCperQuery
        self._pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": self.tokenizer.pad_token_type_id,
            "special_tokens_mask": 1,
        }
        self.read()

    def __getitem__(self,index):
        outputs: List[Dict[str, torch.Tensor]] = []
        keys = [*self.Qrel]
        queryID = keys[index]
        queryText = self.queryDic[queryID]
        list = self.Qrel[queryID]
        docset = []
        relevance_grades = []
        for element in list[0:self.DOCperQuery]:
            docset.append(element[0])
            relevance_grades.append(element[1])


        for doc in range(0, self.DOCperQuery):
            output = self.tokenizer.encode_plus(
                queryText,
                self.collectionDic[docset[doc]],
                add_special_tokens=True,
                max_length=216,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                truncation="longest_first",
                return_special_tokens_mask=False,
            )
            assert all(v.size(0) == 1 for v in output.values())
            outputs.append({k: v[0] for k, v in output.items()})
        encoded_output: Dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            padded = pad_batch(
                [one[key] for one in outputs], value=self._pad_values[key]
            )
            tensor = torch.stack(padded)
            encoded_output[key] = tensor

        return encoded_output , queryID, docset, relevance_grades ,[self.MRRScore[queryID]]

    def __len__(self):
        return (len(self.Qrel))

    def read(self):
        import csv
        self.queryDic = {}
        self.collectionDic = {}
        self.Qrel = {}
        self.MRRScore = {}

        collection_file = open(self.dataPath + "/collection.tsv")
        read_tsv = csv.reader(collection_file, delimiter="\t")
        for row in read_tsv:
            self.collectionDic[int(row[0])] = row[1]

        if self.phase == 'train':
            query_file = open(self.dataPath + "/queries.train_filtered.tsv")
        else:
            query_file = open(self.dataPath+"/queries.dev.small.tsv")
        read_tsv = csv.reader(query_file, delimiter="\t")
        for row in read_tsv:
            self.queryDic[int(row[0])]=row[1]

        MRRScore_file = open(self.dataPath + "/"+self.phase+"_query_mrr"+".tsv")
        read_tsv = csv.reader(MRRScore_file, delimiter="\t")
        for row in read_tsv:
            self.MRRScore[int(row[0])] = float(row[2])


        qrel_file = open(self.dataPath + "/RetrievedwithRelevance" + self.phase + str(self.DOCperQuery))
        read_tsv = csv.reader(qrel_file, delimiter="\t")
        for row in read_tsv:
            if int(row[0]) in self.Qrel:
                list = self.Qrel[int(row[0])]
                list.append((int(row[1]),float(row[2])))
                self.Qrel[int(row[0])] = list
            else:
                self.Qrel[int(row[0])] = [(int(row[1]),float(row[2]))]