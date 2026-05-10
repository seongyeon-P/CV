# QLoRA

QLoRA는 Quantized LLM Fine-tuning 방법으로,
4bit Quantization과 LoRA를 결합하여
적은 GPU 메모리로 대규모 언어모델을 학습할 수 있게 한다.

## 핵심 아이디어

- Base model을 4bit로 양자화
- LoRA adapter만 학습
- 메모리 사용량 감소
- Full Fine-tuning 대비 효율적

## 발표 내용

- 4bit Quantization
- NF4
- Double Quantization
- Guanaco example 코드 분석
- PEFT 기반 adapter 구조

## Original Paper

https://arxiv.org/abs/2305.14314