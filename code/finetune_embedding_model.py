import os
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TranslationEvaluator
from opencc import OpenCC

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# 3. Load a dataset to finetune on
cc = OpenCC('s2tw')
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除標點符號
    text = text.translate(str.maketrans(
        '，。！？【】（）％＃＠＆１２３４５６７８９０',
        ',.!?[]()%#@&1234567890'
    ))  # 轉換全角字符到半角字符
    text = text.strip()  # 移除多餘空格
    return cc.convert(text)

# 設置根資料夾
root_folder = '双语数据'  # 修改為您的根資料夾路徑

anchors = []
positives = []

# 遍歷根資料夾
for dirpath, dirnames, files in os.walk(root_folder):
    if 'bitext.txt' in files:
        file_path = os.path.join(dirpath, 'bitext.txt')
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('古文：'):
                    anchors.append(clean_text(line[len('古文：'):]))
                    print(clean_text(line[len('古文：'):]))
                elif line.startswith('现代文：'):
                    positives.append(clean_text(line[len('现代文：'):]))

# Split the dataset into training and evaluation sets
train_anchors, eval_anchors, train_positives, eval_positives = train_test_split(
    anchors, positives, test_size=0.1, random_state=42
)

# 創建訓練 Dataset
train_dataset = Dataset.from_dict({
    "anchor": train_anchors,
    "positive": train_positives,
})
print("創建 train Dataset OK")

# 創建評估 Dataset
eval_dataset = Dataset.from_dict({
    "anchor": eval_anchors,
    "positive": eval_positives,
})

print("創建 eval Dataset OK")

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)
print("Define a loss function OK")

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/Guwen-nomic-embed-text-v1.5",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if GPU can't handle FP16
    bf16=False,  # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_steps=100,  # Log every 100 steps
    run_name="Guwen-nomic-embed-text-v1.5",  # Used in W&B if `wandb` is installed
    load_best_model_at_end=True
)

print("Specify training arguments OK")

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    loss=loss,
)
trainer.train()

print("Create a trainer & train OK")

# 8. Save the trained model
model.save_pretrained("models/Guwen-nomic-embed-text-v1.5/final")

print("Save the trained model OK")