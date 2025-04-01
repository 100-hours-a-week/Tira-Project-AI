# (Input) --> [Shared Encoder] --> [SwitchGate] --> [Experts] --> [Weighted Sum] --> [Final Output]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models

resnet = models.resnet50(pretrained=True)



### Shared Encoder 정의 ###
class SharedEncoder(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True, hidden_dim=512):
        super().__init__()
        
        # 모델 선택
        self.encoder = self._get_pretrained_model(model_name, pretrained)
        
        # Feature Extractor로 사용 (Fully Connected Layer 제거)
        if hasattr(self.encoder, "fc"):  # ResNet, EfficientNet, MobileNet
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  # 마지막 FC 제거
        elif hasattr(self.encoder, "classifier"):  # VGG, MobileNet
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])  
        
        # Output projection layer (필요시 추가)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(self._get_feature_dim(model_name), num_class)

        # self.projection = nn.Linear(self._get_feature_dim(model_name), hidden_dim)
        # 수정 (feature map projection)
        self.projection = nn.Conv2d(self._get_feature_dim(model_name), hidden_dim, kernel_size=1)


    def _get_pretrained_model(self, model_name, pretrained):
        model_dict = {
            "resnet50": models.resnet50,
            "mobilenetv2": models.mobilenet_v2, # b, d, 7, 7
            "mobilenetv3": models.mobilenet_v3_large,
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "shufflenetv2": models.shufflenet_v2_x1_0
        }
        return model_dict[model_name](pretrained=pretrained)
    
    def _get_feature_dim(self, model_name):
        feature_dim_dict = {
            "resnet50": 2048,
            "mobilenetv2": 1280,
            "mobilenetv3": 1280,
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "shufflenetv2": 1024
        }
        return feature_dim_dict[model_name]

    def forward(self, x):
        x = self.encoder(x) 
        x = self.projection(x)  # Projection to hidden dim

        return x

### 라우팅
class SwitchGate(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        ######## hard routing # Switch Transformer, Switch MoE 논문 기반
        gate_scores = F.softmax(self.w_gate(x), dim=-1) 
        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        mask = torch.zeros_like(gate_scores).scatter_(1, top_k_indices, 1)
        masked_gate_scores = gate_scores * mask
        
        # ######## soft routng # 여러가지 expert 출력을 확률적 비율로 섞어씀
        # gate_scores = F.softmax(self.w_gate(x), dim=-1)


        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = (masked_gate_scores / denominators) * int(self.capacity_factor * x.size(0))

        if use_aux_loss:
            load = gate_scores.sum(0)
            importance = gate_scores.sum(1)
            loss = ((load - importance) ** 2).mean()
            return gate_scores, loss

        # return gate_scores, None
        return gate_scores


class SwitchGate(nn.Module):
    def __init__(self, dim, num_experts, capacity_factor=1.0, epsilon=1e-6):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, domain_label=None, use_aux_loss=False):
        """
        Args:
            x (Tensor): input feature (B, D)
            domain_label (Tensor): domain label (B,) - 0: non-disease, 1: disease
        Returns:
            gate_scores (Tensor): routing score after domain mask
        """

        raw_scores = self.w_gate(x)  # (B, num_experts)

        if domain_label is not None:
            # 도메인 라벨을 기준으로 마스크 생성
            B = raw_scores.size(0)
            mask = torch.full_like(raw_scores, float('-inf'))  # 초기 -inf

            for i in range(B):
                if domain_label[i] == 0:
                    mask[i, :self.num_experts // 2] = 0  # non-disease experts
                else:
                    mask[i, self.num_experts // 2:] = 0  # disease experts

            raw_scores = raw_scores + mask 

        gate_scores = F.softmax(raw_scores, dim=-1) 

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter(1, top_k_indices, 1)
        masked_gate_scores = gate_scores * mask

        # normalize
        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = (masked_gate_scores / denominators) * int(self.capacity_factor * x.size(0))

        return gate_scores




class SwitchMoE(nn.Module):
    def __init__(self, dim, input_dim, num_experts, mult=4, use_aux_loss=False):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.use_aux_loss = use_aux_loss
        self.gate = SwitchGate(dim, num_experts)


        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(),
            nn.Linear(dim * mult, dim)
        ) for _ in range(num_experts)])


    def forward(self, x: Tensor):
        #1. gate_scores: 각 샘플이 어떤 expert로 라우팅될지 확률 벡터
        # hard routing: 0 or 1 
        # gate_scores, loss = self.gate(x, use_aux_loss=self.use_aux_loss)
        x_vec = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # [B, 512]
        gate_scores = self.gate(x_vec, use_aux_loss=self.use_aux_loss)

        #2. expert sub network만들기
        expert_outputs = [expert(x) for expert in self.experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)

        # (B, 1, N) * (B, D, N) -> (B, D, N) 브로드캐스팅 :
        # 각 expert의 출력에 해당 샘플의 expert 확률을 곱함
        moe_output = torch.sum(gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1)
        return moe_output


### 최종 모델 (Shared Encoder + MoE) ###
class MoEWithSharedEncoder(nn.Module):
    def __init__(self, shared_model, shared_hidden_dim, expert_input_dim, num_experts, num_class):
        super().__init__()
        self.shared_encoder = SharedEncoder(model_name=shared_model, pretrained=True, hidden_dim=512)
        self.moe = SwitchMoE(shared_hidden_dim, expert_input_dim, num_experts)
        self.to_out = nn.Sequential(
            nn.LayerNorm(expert_input_dim),
            nn.Linear(expert_input_dim, num_class)
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.shared_encoder(x)  # 공통 인코딩
        # moe_output, aux_loss = self.moe(encoded)  # MoE 처리
        moe_output = self.moe(encoded)  # MoE 처리
        out = self.to_out(moe_output)  # 최종 출력 변환
        # return out, aux_loss
        return out



        