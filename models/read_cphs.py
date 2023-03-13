import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForQuestionAnswering

BATCH_SIZE = 4
EPOCHS_NUM = 10
MAX_LENGTH = 384
STRIDE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReadCphs(nn.Module):
    def __init__(self):
        super(ReadCphs, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        self.model = BertForQuestionAnswering.from_pretrained("bert-bast-cased", return_dict=True)

    def forward(self, inputs):
        inputs = self.tokenizer(
            inputs["question"],              # 问题文本
            inputs["context"],               # 篇章文本
            truncation="only_second",        # 截断只发生在第二部分，即篇章
            max_length=MAX_LENGTH,           # 设定一条文本最大长度384
            stride=128,                      # 设定篇章切片步长为128
            return_overflowing_tokens=True,  # 返回超出最大长度的标记，将篇章切成多片
            return_offsets_mapping=True,     # 返回偏置信息，用于对齐答案位置
            padding="max_length"             # 按最大长度进行补齐
        )

datasets = 1

if __name__ == '__main__':
    model = ReadCphs()
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS_NUM):
        for index in len(train_dataset) // BATCH_SIZE:
            input = get_data(train_dataset, index, BATCH_SIZE)
            predict = model(**input)
            loss = loss_function(predict, label)
            loss.backword()
            optimizer.step()

            accuracy = torch.
