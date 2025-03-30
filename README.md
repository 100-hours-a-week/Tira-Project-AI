# Tira-Project-AI

## 프로젝트명
Masked Mixture of Experts (Masked-MoE)

## 설명
본 프로젝트는 다양한 도메인의 이미지 데이터를 활용하여 효율적인 연산과 높은 분류 정확도를 동시에 달성하기 위해 Mixture of Experts (MoE) 구조를 적용한 실험이다.
특히, 각 입력 이미지가 어떤 도메인에 속하는지를 판별하고, 그 결과를 바탕으로 해당 도메인에 적합한 전문가(Expert) 네트워크만을 활성화하는 방식으로 효율을 극대화하고자 하였다.

## 데이터셋 개요
| 도메인 구분 | 데이터셋         | 클래스 수 | 이미지 수 (샘플링 전 → 후) | 사용 모델            |
|-------------|------------------|------------|-----------------------------|-----------------------|
| non-disease | fruit & vegetable | 36         | 3,466 → 1,296               | Pretrained ResNet     |
| non-disease | vegetable         | 15         | 18,000 → 1,200              | Pretrained EfficientNet |
| non-disease | flower            | 5          | 3,670 → 1,250               | Basic MLP             |
| non-disease | pistachio         | 2          | 2,148 → 1,200               | Basic MLP             |
| non-disease | rice              | 5          | 75,000 → 1,250              | Basic MLP             |
| non-disease | almond            | 4          | 1,556 → 1,200               | Basic MLP             |
| disease     | rice disease      | 3          | 4,684 → 1,200               | Basic MLP             |
| disease     | tomato disease    | 11         | 32,535 → 1,210              | Pretrained EfficientNet |
| disease     | orange disease    | 4          | 1,090 → 1,090               | Basic MLP             |

> 총 9개 도메인, 약 14만 장의 이미지에서 언더샘플링하여 실험 진행


## 목표
- 이미지가 속한 도메인 분류 (disease / non-disease)
- 각 도메인 내에서의 세부 클래스 분류
- 효율적인 연산 구조 구현: 입력에 따라 필요한 전문가만 활성화
- 불필요한 연산을 제거하는 Masked Softmax 적용

## 모델 구조
### Shared Encoder
- 사전학습된 MobileNetV2 기반
- 이미지에서 공통 feature 추출

### Gate1
- Disease / Non-disease 이진 분류
- 단순 MLP

### Expert Network
- 도메인별 Classifier
- 각 도메인에 최적화된 분류기 (예: tomato disease → leaf mold, blight 등)

### Masked Softmax
- 입력 이미지가 속하지 않는 도메인의 Expert는 non-trainable 처리
- 파라미터 효율 및 학습 안정성 향상

## 실험 환경
항목	내용
환경	Google Colab Pro
GPU	A100, 80GB
OS	Linux
Python	3.7
로그 저장	MongoDB Atlas (PyMongo 활용)

## 실험 결과
### Shared Encoder 기반 MoE vs 일반 모델
도메인	구성	Val Acc (%)	Test Acc (%)
fruit & veg	Only Encoder	96.01	97.40
fruit & veg	Gate1 + Expert	96.58	98.27
vegetable	Only Encoder	99.10	99.03
vegetable	Gate1 + Expert	99.90	99.86
orange disease	Only Encoder	93.94	97.28
orange disease	Gate1 + Expert	96.97	99.19
→ Gate1을 통해 라우팅된 Expert를 사용할 경우 성능 향상이 일관적으로 나타남.

### Gate1 성능 (도메인 분류 정확도)
- Disease / Non-disease 구분 이진 분류에서 초기 학습부터 99% 이상의 정확도 달성
- 시각적 힌트 기반 학습 가능성이 있으므로 t-SNE 분석 필요성 제기

### 결론
- Shared Encoder 기반의 Masked MoE 구조는 효율적인 라우팅 확인
- Sub Expert 분류기 학습에 대한 실험은 완료되지 않았으며, 향후 라우팅 이후 분류 정확도 분석 및 전체 연산 효율성 분석이 필요
- Gate1이 성능을 잘 내고 있지만, 단순한 시각적 차이에 의존했을 가능성이 있어 feature 기반의 일반화 표현 학습 여부를 시각화 기법으로 분석할 예정

### 참고자료
Switch Transformer (Fedus et al., 2021)
Sparse MoE Layer (Shazeer et al., 2017)
Adaptive Mixtures of Local Experts (Hinton et al., 1991)
Parameter-Efficient MoE (Gao et al., 2022)
