import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW

BATCH_SIZE = 16
CLASSES_NUM = 10
EPOCHS_NUM = 15
HIDDEN_SIZE = 768
SEQUENCE_LENGTH = 140
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_json(path1, path2, length=None):
    x, y = [], []
    with open(path1).readlines() as f:
        for line in f.readlines():



def get_batch(x, y, batch_size=BATCH_SIZE):
    n_batchs = len(x) // batch_size
    for i in range(0, n_batchs * batch_size, batch_size):
        if i != (n_batchs - 1) * batch_size:
            X, Y = x[i: i + batch_size], y[i: i + batch_size]
        else:
            X, Y = x[i:], y[i:]
        yield  X, Y

class DomainClassification(nn.Module):

    def __init__(self):
        super(DomainClassification, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-chinanews-chinese")
        self.model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-chinanews-chinese")
        self.linear1 = nn.Linear(HIDDEN_SIZE * 4, 256)
        self.linear2 = nn.Linear(256, CLASSES_NUM)

    def forward(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=SEQUENCE_LENGTH)
        inputs = inputs.to(DEVICE)
        hidden = self.model(**inputs)
        max_hidden_8 = torch.max(hidden.hidden_states[-4], dim=1)[0]
        max_hidden_9 = torch.max(hidden.hidden_states[-3], dim=1)[0]
        max_hidden_10 = torch.max(hidden.hidden_states[-2], dim=1)[0]
        max_hidden_11 = torch.max(hidden.hidden_states[-1], dim=1)[0]
        max_pooled = torch.cat((max_hidden_8, max_hidden_9, max_hidden_10, max_hidden_11), dim=1)
        out1 = self.linear1(max_pooled)
        out2 = self.linear2(F.leaky_relu(out1))
        return out2


if __name__ == '__main__':
    model = DomainClassification()
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-7)
    x_train, y_train = load_data_from_json("./data/real_release_all.json", "./data/fake_release_all.json", None)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_setps=0.05,
                                                num_training_steps=len(x_train) // BATCH_SIZE * EPOCHS_NUM)

    for epoch in range(EPOCHS_NUM):
        sum_loss = 0
        sum_acc = 0
        for index, (x, y) in enumerate(get_batch(x_train, y_train)):
            y = torch.from_numpy(y).long()
            y = y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            sum_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            predict = torch.argmax(output, dim=1)
            out = torch.eq(predict, y)
            sum_acc += out.sum()
            if index % 100 == 0:
                print("epoch: ", "%04d" % (epoch), "cost: ", "{:.6f}".format(sum_loss / 1000),
                      "index = ", "%04d" % (index), "acc = ", (sum_acc * 1.0 / (100 * predict.size()[0])).item())
                sum_loss = 0
                sum_acc = 0
        torch.save(model, "domain_cls_{}.pkl".format(epoch))
    torch.save(model, "domain_cls_model.pkl")
