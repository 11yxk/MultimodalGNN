import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
# from layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
from vig_pytorch.pyramid_vig import pvig_ti_224_gelu

import torch.nn.functional as F


class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()

        # 加载 BERT 模型并定义投影层
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )

        # 冻结 BERT 参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sen_token, sen_mask, input_ids, attention_mask):


        sen_output = self.model(input_ids=sen_token, attention_mask=sen_mask)
        sen_embed = sen_output.last_hidden_state[:, 0, :]


        # 获取输入的批次大小和短语数量
        batch_size, max_phrases, max_length = input_ids.size()

        # 将输入调整为 (batch_size * max_phrases, max_length)
        input_ids = input_ids.view(-1, max_length)
        attention_mask = attention_mask.view(-1, max_length)


        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        embed = output.last_hidden_state[:, 0, :]
        embedding_dim = embed.size(-1)
        batch_embeddings = embed.view(batch_size, max_phrases, embedding_dim)
        # 进行维度变换并池化


        batch_embeddings_fusion = torch.cat([sen_embed.unsqueeze(dim =1), batch_embeddings], dim=1)
        # 将嵌入通过投影层
        embed = self.project_head(batch_embeddings_fusion)  # (batch_size * max_phrases, project_dim)



        return embed


# class BERTModel(nn.Module):
#     def __init__(self, bert_type, project_dim):
#         super(BERTModel, self).__init__()
#
#         # 加载 BERT 模型并定义投影层
#         self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
#         self.project_head = nn.Sequential(
#             nn.Linear(768, project_dim),
#             nn.LayerNorm(project_dim),
#             nn.GELU(),
#             nn.Linear(project_dim, project_dim)
#         )
#
#         # 冻结 BERT 参数
#         for param in self.model.parameters():
#             param.requires_grad = False
#
#     def forward(self, sen_token, sen_mask, input_ids, attention_mask):
#
#
#         sen_output = self.model(input_ids=sen_token, attention_mask=sen_mask)
#         # print(111111111111)
#         # print(sen_output.last_hidden_state.shape)
#         sen_embed = sen_output.last_hidden_state[:, 0, :]
#
#
#         # 获取输入的批次大小和短语数量
#         batch_size, max_phrases, max_length = input_ids.size()
#
#         # 将输入调整为 (batch_size * max_phrases, max_length)
#         input_ids = input_ids.view(-1, max_length)
#         attention_mask = attention_mask.view(-1, max_length)
#
#
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask)
#
#         embed = output.last_hidden_state[:, 0, :]
#         embedding_dim = embed.size(-1)
#         batch_embeddings = embed.view(batch_size, max_phrases, embedding_dim)
#         # 进行维度变换并池化
#
#
#         batch_embeddings_fusion = torch.cat([sen_embed.unsqueeze(dim =1), batch_embeddings], dim=1)
#
#         # 将嵌入通过投影层
#         embed = self.project_head(batch_embeddings_fusion)  # (batch_size * max_phrases, project_dim)
#
#
#
#         return embed
class KNNRegressionNet(nn.Module):
    def __init__(self, patch_size, channels):
        super(KNNRegressionNet, self).__init__()
        # 定义一个简单的卷积层
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels//4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels//4, out_channels=channels//8, kernel_size=3, padding=1)
        # 定义全连接层，将卷积层输出映射到最终的回归输出
        self.fc = nn.Linear((channels//8) * patch_size, 1)

    def forward(self, x):
        # 输入 x 的 shape 为 (batch_size, channels, patch_size)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))  # 卷积层并应用激活函数
        x = F.relu(self.conv2(x))  # 卷积层并应用激活函数
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层输出
        return torch.sigmoid(x)


# class VisionModel(nn.Module):
#     def __init__(self):
#         super(VisionModel, self).__init__()
#
#         self.encoder = pvig_ti_224_gelu()
#         checkpoint = torch.load('/mnt/data1/RIS/LanGuideMedSeg-MICCAI2023-main-knnnet/pvig_ti_78.5.pth.tar')
#         self.encoder.load_state_dict(checkpoint)
#
#
#         print('===========================')
#         print('using VIG')
#         print('===========================')
#
#     def forward(self, x):
#         # 提取每个层级的特征
#         features = self.encoder(x)
#
#         return features


# class VisionModel(nn.Module):
#     def __init__(self):
#         super(VisionModel, self).__init__()
#
#         self.encoder = pvig_ti_224_gelu()
#
#         pretrained_dict = torch.load('/mnt/data1/RIS/LanGuideMedSeg-MICCAI2023-main-knnnet/pvig_ti_78.5.pth.tar')
#         new_state_dict = self.encoder.state_dict()
#         # 过滤出名字匹配的参数
#         matched_dict = {k: v for k, v in pretrained_dict.items() if k in new_state_dict}
#         # 更新新网络的 state_dict
#         new_state_dict.update(matched_dict)
#         # 加载更新后的 state_dict
#         self.encoder.load_state_dict(new_state_dict, strict=False)
#
#         print('===========================')
#         print('using VIG')
#
#     def forward(self, x):
#         # 提取每个层级的特征
#         features = self.encoder(x)
#
#         return features


class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()

        # 初始化模型
        self.encoder = pvig_ti_224_gelu()

        # 加载预训练模型参数
        pretrained_path = '/mnt/data1/RIS/LanGuideMedSeg-MICCAI2023-main-knnnet/pvig_ti_78.5.pth.tar'
        pretrained_dict = torch.load(pretrained_path)

        # 当前模型的参数字典
        current_state_dict = self.encoder.state_dict()

        # 过滤出名字匹配的参数
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in current_state_dict and v.size() == current_state_dict[k].size()}

        # 更新当前模型的参数字典
        current_state_dict.update(matched_dict)

        # 加载更新后的参数
        self.encoder.load_state_dict(current_state_dict, strict=False)

        print('===========================')
        print('Using VIG with partial loading of matched weights.')

        # print('===========================')
        # print('Using VIG wo pretrain')

    def forward(self, x):
        # 提取每个层级的特征
        features = self.encoder(x)
        return features


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=768):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel()
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [384,240,96,48]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0])
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1])
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2])
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.k_estimate = KNNRegressionNet(49,feature_dim[0])

    def forward(self, data):


        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_features = self.encoder(image)

        text_output = self.text_encoder(text['sen_token'],text['sen_mask'], text['input_ids'],text['attention_mask'])

        image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features]
        k = self.k_estimate(image_features[3]).squeeze(-1)

        os32 = image_features[3]

        os16 = self.decoder16(os32,image_features[2], text_output,k)
        os8 = self.decoder8(os16,image_features[1], text_output,k)
        os4 = self.decoder4(os8,image_features[0], text_output,k)

        # os16 = self.decoder16(os32,image_features[2], None,k)
        # os8 = self.decoder8(os16,image_features[1], None,k)
        # os4 = self.decoder4(os8,image_features[0], None,k)

        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out, k

if __name__ == '__main__':
    from thop import profile

    image = torch.randn(1, 3, 224, 224)

    text = {'sen_token': torch.randint(0,10000,(1, 1)), 'sen_mask':torch.randint(0,1,(1, 1)), 'input_ids': torch.randint(0,10000,(1, 5, 5)), 'attention_mask':torch.randint(0,1,(1,5, 5))}

    data = (image,text)
    model = LanGuideMedSeg(bert_type = 'microsoft/BiomedVLP-CXR-BERT-specialized',vision_type = 'facebook/convnext-tiny-224')
    # out = model(data)
    # print(out.shape)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {total_params/1e6}")

    flops, params = profile(model, inputs=(data,))
    print(flops/1e9, params/1e6)

    vig = VisionModel()
    total_params = sum(p.numel() for p in vig.parameters() if p.requires_grad)
    print(f"可训练参数数量: {total_params/1e6}")


