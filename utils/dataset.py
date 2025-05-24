import json
import os
import torch
import pandas as pd
import numpy as np
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd, RandCoarseShuffled, RandRotated,
                              RandZoomd, Resized, ToTensord, LoadImaged, EnsureChannelFirstd, Flipd, Rotated)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import spacy
import re



class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224],
                 max_phrases=5, max_length=5):

        super(QaTa, self).__init__()

        self.mode = mode
        self.data = pd.read_excel(csv_path)

        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if mode == 'train':
            self.image_list = self.image_list[:int(0.8*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8*len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
        else:
            pass   # for mode is 'test'

        self.root_path = root_path
        self.image_size = image_size
        self.max_phrases = max_phrases  
        self.max_length = max_length  
        self.nlp = spacy.load("en_core_web_sm")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)
        image = os.path.join(self.root_path, 'Images', self.image_list[idx].replace('mask_', ''))
        gt = os.path.join(self.root_path, 'GTs', self.image_list[idx])
        caption = self.caption_list[idx]


        sentence_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')

        sentence_token, sentence_mask = sentence_output['input_ids'].squeeze(dim=0),sentence_output['attention_mask'].squeeze(dim=0)


        phrases = re.split(r',\s*', caption)
       
        final_phrases = []
        for phrase in phrases:
            doc = self.nlp(phrase)
           
            chunks = [chunk.text for chunk in doc.noun_chunks]
            if not chunks:
                final_phrases.append(phrase)
            else:
                
                for chunk in chunks:
                    if "lung" in chunk:
                       
                        sub_phrases = re.split(r' and ', chunk)
                        final_phrases.extend(sub_phrases)
                    else:
                        final_phrases.append(chunk)
        phrases = final_phrases
        phrases = phrases[:self.max_phrases] + [""] * (self.max_phrases - len(phrases))  # 填充到 max_phrases 数量


        
        input_ids = torch.zeros((self.max_phrases, self.max_length), dtype=torch.long)
        attention_masks = torch.zeros((self.max_phrases, self.max_length), dtype=torch.long)
        for i, phrase in enumerate(phrases):
            encoded = self.tokenizer.encode_plus(
                phrase,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids[i] = encoded["input_ids"].squeeze(0)
            attention_masks[i] = encoded["attention_mask"].squeeze(0)




        data = {'image': image, 'gt': gt,'sen_token': sentence_token, 'sen_mask': sentence_mask,  'token': input_ids, 'mask': attention_masks}
        data = trans(data) 



       
        image, gt, sentence_token, sentence_mask, token, mask = data['image'], data['gt'],data['sen_token'], data['sen_mask'], data['token'], data['mask']

        gt = torch.where(gt == 255, 1, 0)
        rate = torch.mean(gt.float())




        
        text = {'sen_token': sentence_token, 'sen_mask': sentence_mask, 'input_ids': token, 'attention_mask': mask}
        return ([image, text], gt, rate)

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                Flipd(keys=["image", "gt"], spatial_axis=0),
                Rotated(keys=["image"], angle=np.pi * (1 / 2), mode='bicubic', keep_size=True),
                Rotated(keys=["gt"], angle=np.pi * (1 / 2), mode='nearest', keep_size=True),
                ToTensord(["image", "gt","sen_token", "sen_mask",  "token", "mask"]),
            ])

        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                Flipd(keys=["image", "gt"], spatial_axis=0),
                Rotated(keys=["image"], angle=np.pi * (1 / 2), mode='bicubic', keep_size=True),
                Rotated(keys=["gt"], angle=np.pi * (1 / 2), mode='nearest', keep_size=True),
                ToTensord(["image", "gt","sen_token", "sen_mask",  "token", "mask"]),

            ])

        return trans


