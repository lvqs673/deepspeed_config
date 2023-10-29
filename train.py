import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed


class MyDataset(Dataset):
    def __init__(self, sentence_path, tokenizer):
        super().__init__()
        sents = []
        with open(sentence_path, encoding='utf-8') as f:
            for line in f:
                sent = line.strip()
                if sent:
                    sents.append(sent)

        input_ids, target_ids, mask = [], [], []
        res = tokenizer(sents, truncation=True, max_length=256)
        input_ids = res['input_ids']
        mask = res['attention_mask']
        target_ids = []
        for i, ids in enumerate(input_ids):
            input_ids[i] = ids[:-1]
            target_ids.append(ids[1:])
            mask[i].pop()
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.mask = mask

    def __getitem__(self, i):
        return self.input_ids[i], self.target_ids[i], self.mask[i]

    def __len__(self):
        return len(self.input_ids)


def pad_to_length(sents, length, pad_token_id=0):
    res = []
    for sent in sents:
        sent = sent+[pad_token_id]*(length-len(sent))
        res.append(sent)
    return torch.tensor(res, dtype=torch.int64)


def collate_fn(batch):
    input_ids, target_ids, mask = zip(*batch)
    max_len = max(len(input_ids) for input_ids in input_ids)
    input_ids = pad_to_length(input_ids, max_len, 0)
    target_ids = pad_to_length(target_ids, max_len, -100)
    mask = pad_to_length(mask, max_len, 0)
    return input_ids, target_ids, mask


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='deepspeed training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--nepoch', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='./data/sentences.txt')
    parser.add_argument('--model_save_dir', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='epoch_{}.pt')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train():
    args = parse_arguments()
    deepspeed.init_distributed()

    # load model and dataset
    model_name = "bloom-3b"
    model_path = os.path.expanduser(f'~/local_transformers/{model_name}')
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = MyDataset(sentence_path=args.dataset, tokenizer=tokenizer)

    # init engine
    engine, _, training_dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        collate_fn=collate_fn,
    )

    # train
    engine.train()
    for epoch in range(args.nepoch):
        for step, (xs, ys, mask) in enumerate(training_dataloader, 1):
            xs = xs.to(device=engine.device, dtype=torch.long)
            ys = ys.to(device=engine.device, dtype=torch.long)
            mask = mask.to(device=engine.device, dtype=torch.long)

            logits = engine(input_ids=xs, attention_mask=mask).logits
            loss = F.cross_entropy(
                logits.transpose(1, 2), ys, ignore_index=-100)
            engine.backward(loss)
            engine.step()

        # model path example: ./model/epoch_1.pt
        model_name = args.model_name.format(epoch+1)
        engine.save_16bit_model(args.model_save_dir, model_name)
    


if __name__ == '__main__':
    train()
