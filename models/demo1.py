import torch
from torch import nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# 驱动选择
device = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.zeros((26, 26), dtype=torch.float32).to(device=device)
labels = []
for i in range(26):
    labels.append((i + 1) % 26)
    X[i][i] = 1.
labels = torch.tensor(labels)
dataset = Dataset.from_dict({'x': X, 'labels': labels})


# 残差网络
class RN(nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(26, 64),
            nn.Hardsigmoid(),
            nn.Linear(64, 26),
            nn.Hardsigmoid(),
        )

        self.linear_stack_2 = nn.Sequential(
            nn.Linear(26, 64),
            nn.Hardsigmoid(),
            nn.Linear(64, 64),
            nn.Hardsigmoid(),
        )

        self.output_layer = nn.Linear(64, 26)

        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, x, labels, mode='train'):
        y = self.linear_stack(x)
        # 残差
        y = y + x
        y = self.linear_stack_2(y)
        y = self.output_layer(y)

        if mode is 'train':
            return {
                'loss': self.loss_f(y, labels),
                'predictions': y
            }

        return y


# 生成模型实例
model = RN().to(device=device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (labels == preds).sum() / len(labels)
    return {
        'accuracy': acc,
    }


training_args = TrainingArguments(
    output_dir='./results',  # output directory 结果输出地址
    num_train_epochs=1000,  # total # of training epochs 训练总批次
    per_device_train_batch_size=1,  # batch size per device during training 训练批大小
    per_device_eval_batch_size=1,  # batch size for evaluation 评估批大小
    logging_dir='./logs/rn_log',  # directory for storing logs 日志存储位置
    learning_rate=1e-3,  # 学习率
    save_steps=False,  # 不保存检查点
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained 需要训练的模型
    args=training_args,  # training arguments, defined above 训练参数
    train_dataset=dataset,  # training dataset 训练集
    eval_dataset=dataset,  # evaluation dataset 测试集
    compute_metrics=compute_metrics  # 计算指标方法
)

trainer.train()
trainer.evaluate()
