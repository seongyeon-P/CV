# LoRA

LoRA(Low-Rank Adaptation)는
대규모 언어모델 Fine-tuning 효율화를 위한 방법이다.

## 핵심 아이디어

- 기존 pretrained weight freeze
- ΔW = BA 형태의 low-rank update만 학습
- 적은 파라미터만 학습하여 메모리 절약

## 발표 내용

- Full Fine-tuning 문제점
- Adapter / Prefix tuning 비교
- Low-rank decomposition
- Transformer 적용 방식

## Original Paper

https://arxiv.org/abs/2106.09685