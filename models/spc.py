import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_NUM = 20
BATCH_SIZE = 8
SEQUENCE_LENGTH = 140

def tokenize(examples):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")

def load_data():
    dataset = load_dataset("glue", "rte")
    return dataset["train"], dataset["validation"], dataset["test"]

def get_batch(dataset, batch_size=BATCH_SIZE):
    n_batchs = len(dataset)
    for i in range(0, n_batchs * batch_size, batch_size):
        if i != (n_batchs - 1) * batch_size:
            X1, X2, Y = dataset["sentence1"][i: i + batch_size], dataset["sentence2"][i: i + batch_size], dataset["label"][i: i + batch_size]
            yield X1, X2, np.array(Y)
        # else:
        #     X1, X2, Y = dataset["sentence1"][i:], dataset["sentence2"][i:], dataset["label"][i:]



class SPC(nn.Module):

    def __init__(self):
        super(SPC, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, hypo, prem):
        inputs = self.tokenizer(hypo, prem, return_tensors="pt", truncation=True, padding="max_length")
        print(len(hypo))
        print(len(prem))
        inputs.to(DEVICE)
        hidden = self.model(**inputs, output_hidden_states=True)
        max_hidden = torch.max(hidden.hidden_states[-1], dim=1)[0]
        out1 = self.linear1(max_hidden)
        out2 = self.linear2(out1)
        return out2

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = load_data()

    model = SPC()
    model = model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS_NUM):
        sum_loss = 0
        sum_acc = 0
        for index, (hypo, prem, y) in enumerate(get_batch(train_dataset, BATCH_SIZE)):
            y = torch.from_numpy(y).long()
            y = y.to(DEVICE)
            optimizer.zero_grad()
            predict = model(hypo, prem)
            loss = criterion(predict, y)
            sum_loss += loss
            loss.backward()
            optimizer.step()
            acc = torch.eq(y, predict.argmax(dim=1)).sum().float()
            sum_acc += acc
            if index % 10 == 0:
                print("epoch:", "%04d" % (epoch), "cost:", "{:.6f}".format(sum_loss / 1000),
                      "index:", "%04d" % (index), "acc:", (sum_acc * 1.0 / (10 * predict.size()[0])).item())
                sum_loss = 0
                sum_acc = 0
        torch.save(model, "scc_model.pkl")