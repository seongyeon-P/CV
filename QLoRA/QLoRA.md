# QLoRA

기간: 2026년 4월 3일
역할: 코드 리뷰
URL: QLORA: Efficient Finetuning of Quantized LLMs

## 프로젝트 관련 코드 간단 설명

---

- QLoRA의 example 코드
    - 학습된 모델을 실제로 사용하는 코드
    - 모델 로드 → 입력 → generate
    - 4bit base model + LoRA adapter 구조
    - 
- QLoRA/examples
    
    ├── guanaco_7B_demo_colab.ipynb      ← 학습/데모 예시
    ├── guanaco_generate.py                        ← 모델 사용(추론) 예시
    

## **guanaco_7B_demo_colab.ipynb**

```python
# Install latest bitsandbytes & transformers, accelerate from source
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
# Other requirements for the demo
!pip install gradio
!pip install sentencepiece
```

### 모델 로드

```python
# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

model_name = "decapoda-research/llama-7b-hf"
adapters_name = 'timdettmers/guanaco-7b'

print(f"Starting to load the model {model_name} into memory")

# 베이스 LLaMA 모델을 불러옴 (QLoRA의 base model)
m = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

# LoRA 어댑터를 베이스 모델에 결합 (QLoRA 구조)
m = PeftModel.from_pretrained(m, adapters_name)

# LoRA를 base model에 병합하여 하나의 모델로 사용
m = m.merge_and_unload()

# 토크나이저 설정
# 입력 문장을 토큰으로 변환하기 위한 tokenizer
tok = LlamaTokenizer.from_pretrained(model_name) 

# LLaMA 모델 호환을 위한 설정 
tok.bos_token_id = 1

stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")
```

### Gradio UI 구성

사용자 입력 → 프롬포트 구성 → 토큰화 → GPU 전달

```python
# Setup the gradio Demo.

import datetime
import os
from threading import Event, Thread
from uuid import uuid4

import gradio as gr
import requests

max_new_tokens = 1536
start_message = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def convert_history_to_text(history):
    text = start_message + "".join(
        [
            "".join(
                [
                    f"### Human: {item[0]}\n",
                    f"### Assistant: {item[1]}\n",
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    f"### Human: {history[-1][0]}\n",
                    f"### Assistant: {history[-1][1]}\n",
                ]
            )
        ]
    )
    return text

def log_conversation(conversation_id, history, messages, generate_kwargs):
    logging_url = os.getenv("LOGGING_URL", None)
    if logging_url is None:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "history": history,
        "messages": messages,
        "generate_kwargs": generate_kwargs,
    }

    try:
        requests.post(logging_url, json=data)
    except requests.exceptions.RequestException as e:
        print(f"Error logging conversation: {e}")

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    print(f"history: {history}")
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = convert_history_to_text(history)

    # Tokenize the messages string
    input_ids = tok(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(m.device)
    streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    stream_complete = Event()

    def generate_and_signal_complete():
        m.generate(**generate_kwargs)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()
        log_conversation(
            conversation_id,
            history,
            messages,
            {
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            },
        )

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    t2 = Thread(target=log_after_stream_complete)
    t2.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        yield history

def get_uuid():
    return str(uuid4())

# Gradio demo (UI)
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown(
        """Guanaco Demo
"""
    )
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.7,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.9,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=0,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                with gr.Column():
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.0,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
    with gr.Row():
        gr.Markdown(
            "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. The model was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=2)
```

### 마지막 실행

```python
# Launch your Guanaco Demo!
# 웹 기반 챗봇 형태로 모델 실행
demo.launch()
```

# guanaco_generate.py

QLoRA로 학습된 Guanaco-7B 어댑터를, 4-bit LLAMA-7B 베이스 모델 위에 붙여 질문에 답변을 생성하는 추론 예제 코드

```python
# local에 학습 중이던 checkpoint가 있으면 last checkpoint를 찾아주는 함수
def get_last_checkpoint(checkpoint_dir): 
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training
```

### 1. 설정 (생성 파라미터)

```python
# TODO: Update variables
max_new_tokens = 64    # 최대 생성길이
top_p = 0.9            # 확률이 높은 후보들 위주로 샘플링
temperature=0.7        # 답변 다양성 조절
user_question = "What is Einstein's theory of relativity?". # 예시 질문
```

### **2. 모델 준비 (Base model** + **Adapter)**

```python
# Base model
model_name_or_path = 'huggyllama/llama-7b'        # 원래의 베이스 LLaMA-7B
# Adapter name on HF hub or local checkpoint path.
# adapter_path, _ = get_last_checkpoint('qlora/output/guanaco-7b')
adapter_path = 'timdettmers/guanaco-7b'           # QLoRA로 학습된 Guanaco LoRA Adapter

# 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 초반 LLaMA 변환 이슈를 맞춰주기 위한 보정
tokenizer.bos_token_id = 1
```

### 3. 모델 로드 (4bit + NF4)

```python
# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=True,                           # 베이스 모델을 4bit로 불러옴
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,   # 저장은 4bit이지만, 계산은 16bit
        bnb_4bit_use_double_quant=True,          # Double Quantization
        bnb_4bit_quant_type='nf4',               # NF4
    )
)
```

### 4. LoRA 결합

```python
# LoRA Adapter 결합
model = PeftModel.from_pretrained(model, adapter_path) # PEFT 방식으로 결합
model.eval()                                           # 추론 모드 전환
```

### 5. 입력 구성 (prompt)

```python
# prompt 형식 제작 (대화형 모델이기에, 질문을 단순 문장이 아닌 Human과 Assistant 형식)
prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "### Human: {user_question}"
    "### Assistant: "
)
```

### 6. 출력 생성 (generate)

```python
# generate함수에서 실제 답변 생성
# QLoRA로 구성된 모델(4bit base + LoRA adapter)을 사용한 추론 과정

# 사용자 질문 -> 프롬포트 생성 -> 토큰화 -> GPU
def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

    # 실제 답변 생성
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )

    # 생성된 토큰을 사람이 읽을 수 있는 text로 전환 후 출력
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

# 실제 실행
generate(model, user_question)

# 디버깅용-> 코드 실행을 멈추고 상태를 보는 용도 
import pdb; pdb.set_trace()
```