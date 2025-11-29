---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - 오디오 인식 모델 파인튜닝"
date:   2025-11-29 00:10:22 +0900
categories: Huggingface_Audio
---

# 오디오 인식 모델 파인튜닝

- 실제 음성 인식 모델을 학습하기 위한 전체 파이프라인

Whisper는 이미 68만 시간의 음성-텍스트 데이터로 사전학습된 강력한 모델이다.
그러나 Whisper를 특정 언어(특히 저자원 언어), 특정 도메인(회의 녹음, 상담 콜, 유튜브, 의료, 법률 등)
또는 특정 음성 스타일에 맞게 만들기 위해서는 fine-tuning이 필요하다.

아래는 Whisper-small 모델을 Common Voice 13의 Dhivehi 데이터로 직접 학습시키는 전체 과정이다.
동일한 흐름으로 어떤 언어, 어떤 ASR 데이터셋에도 적용할 수 있는 범용 가이드이기도 하다.

## 준비 단계

### Hugging Face Hub 연결

훈련 중 생성되는 체크포인트를 Hub에 업로드하면 다음 장점이 있다.

- 모델 버전 관리
- 훈련 중간중간 자동 백업
- TensorBoard 로그 제공
- 협업 및 공유 용이
- Colab 환경에서도 안전하게 저장 가능

```python
from huggingface_hub import notebook_login
notebook_login()
```

## 데이터 로드

Common Voice 13에는 Dhivehi 음성이 약 10시간 정도 존재한다.
ASR fine-tuning에서 10시간은 매우 적은 양이지만 Whisper는 강력한 multilingual representation을 이미 갖추고 있어
저자원 언어에서도 좋은 결과를 낼 수 있다.

### train + validation -> train

Dhivehi는 특히 저자원이므로 train과 validation을 합쳐서 전부 학습에 사용한다.

```python
from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="train+validation"
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "dv", split="test"
)
```

### 필요한 컬럼만 남기기

ASR fine-tuning에서 필요한 것은 단 두 가지다.

- audio (raw waveform)
- sentence (정답 텍스트)

```python
common_voice = common_voice.select_columns(["audio", "sentence"])
```

## Processor 구성

WhisperProcessor는

- feature extractor (오디오 -> log-mel spectrogram)
- tokenizer (텍스트 -> 토큰)

를 묶어 하나의 인터페이스로 제공한다.

Whisper는 multilingual이므로
processor를 생성할 때 language와 task를 명시해야 한다.

### Dhivehi는 Whisper 사전학습 언어에 없음

Whisper는 96개 언어로 사전학습되었지만, Dhivehi는 포함되어 있지 않다.

이 경우 Whisper에게 가장 가까운 언어를 선택해야 한다.

공식 언어 문헌에 따르면
Dhivehi는 **Sinhala(스리랑카어)**와 계통적으로 가장 가깝다.
Whisper는 Sinhala를 지원하므로 language="sinhalese"로 세팅한다.

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="sinhalese",
    task="transcribe"
)
```

이 방식은 Whisper에게 새로운 언어를 가르치는 과정이다.
Cross-lingual transfer 덕분에 사전학습된 언어들로부터 패턴을 학습해 빠르게 적응한다.

## 오디오 전처리 (sampling rate)

Whisper가 받는 입력은 16kHz log-mel spectrogram이다.
그러므로 데이터셋의 오디오가 48kHz이면 반드시 16kHz로 리샘플링해야 한다.

Datasets의 cast_column을 사용하면 내부적으로 자동 resampling이 적용된다.

```python
from datasets import Audio

sampling_rate = processor.feature_extractor.sampling_rate
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate))
```

## 데이터 전처리 함수 작성

다음 작업을 수행한다.

1. 오디오 로드
2. Whisper feature extractor 적용 -> log-mel spectrogram 생성
3. tokenizer로 label 텍스트 토큰화
4. input 길이 계산하여 long audio 제거에 사용

```python
def prepare_dataset(example):
    audio = example["audio"]
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    return example
```

모든 샘플에 적용

```python
common_voice = common_voice.map(
    prepare_dataset,
    # remove_columns=["audio", "sentence"]와 같은 의미
    remove_columns=common_voice.column_names["train"],
    # 멀티프로세싱 병렬 처리 개수
    num_proc=1
)
```

## 30초 초과 오디오 필터링

Whisper는 30초 단위 입력을 기준으로 설계되어 있다.
30초 이상은 Truncate되므로 학습 안정성 문제가 발생할 수 있다.

```python
max_input_length = 30.0

def is_audio_in_length_range(length):
    return length < max_input_length

common_voice["train"] = common_voice["train"].filter(
    is_audio_in_length_range,
    input_columns=["input_length"],
)
```

## Data Collator 정의

ASR의 Data Collator는 일반 NLP와 다르다.
이유는 입력(feature)와 출력(label)의 구조가 완전히 다르기 때문이다.

- input_features: 이미 패딩 완료된 log-mel spectrogram
- labels: 길이가 다 다르므로 tokenizer로 padding 필요
- padding된 label token은 loss에 포함되면 안 되므로 -100으로 변환

```python
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    # # WhisperProcessor(feature_extractor + tokenizer)를 외부에서 받아옴
    processor: Any

    def __call__(self, features):
        """
        features: DataLoader가 dataset에서 뽑아온 샘플들의 리스트.
        각 샘플은 prepare_dataset()에서 생성한 dict 구조.

        예시:
        {
            "input_features": [...],   # log-mel spectrogram
            "labels": [...],           # tokenized label ids
            "input_length": ...
        }
        """
        # Whisper feature_extractor는 다음과 같은 형태를 기대한다. [{"input_features": feature}, {"input_features": feature}, ...]
        # 그런데 prepare_dataset에서 "input_features"의 shape이 [1,] (리스트 1개짜리)라 f["input_features"][0]로 첫 번째 요소를 꺼내서 전달한다.
        input_features = [
            {"input_features": f["input_features"][0]} for f in features
        ]

        # feature_extractor.pad()는 이미 길이가 동일한 Whisper spectrogram을 단순히 PyTorch tensor로 묶어주는 역할을 한다.
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Whisper tokenizer는 문자열(label)을 token ID로 변환해놨지만, 길이가 제각각이라 pad()로 동일 길이로 맞춰야 함.
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        # labels_batch:
        # {
        #   "input_ids": padded_label_ids,
        #   "attention_mask": mask_for_labels
        # }

        # label padding token의 위치(attention_mask != 1)를 -100으로 바꿈.
        # CrossEntropyLoss는 ignore_index=-100을 기본값으로 사용하므로, -100 위치는 loss 계산에서 제외됨.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # tokenizer.pad()가 BOS(<|startoftranscript|>)을 자동으로 붙이는 경우가 있음.
        # 그러나 Whisper 모델은 forward() 내부에서 BOS를 다시 prepend하므로 label의 첫 토큰이 BOS라면 제거해야 중복이 생기지 않음.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            # 모든 샘플에서 첫 토큰이 BOS라면 해당 토큰 제거
            labels = labels[:, 1:]

        # batch dict에 labels 추가하여 Trainer로 반환
        batch["labels"] = labels
        return batch
```

## 평가 메트릭 정의 (WER)

이미 설명했듯, ASR의 표준 지표는 Word Error Rate(WER)다.

```python
import evaluate
metric = evaluate.load("wer")
```

Whisper 전용 텍스트 정규화도 적용한다.

```python
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()
```

정규화된 WER과 orthographic WER 둘 다 계산한다.

```python
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(
        predictions=pred_str, references=label_str
    )

    pred_str_norm = [normalizer(p) for p in pred_str]
    label_str_norm = [normalizer(l) for l in label_str]

    pred_str_norm = [
        p for p, l in zip(pred_str_norm, label_str_norm) if len(l) > 0
    ]
    label_str_norm = [
        l for l in label_str_norm if len(l) > 0
    ]

    wer = 100 * metric.compute(
        predictions=pred_str_norm, references=label_str_norm
    )

    return {"wer_ortho": wer_ortho, "wer": wer}
```

## 모델 로드 및 설정

```python
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.use_cache = False   # gradient checkpointing과 충돌
```

Whisper는 generation 시 언어/태스크 토큰을 필요로 한다.
학습 중에는 필요 없지만, inference에서는 필수다.

```python
from functools import partial

model.generate = partial(
    model.generate,
    language="sinhalese",
    task="transcribe",
    use_cache=True
)
```

10. TrainingArguments 설정

중요 파라미터 설명

1. learning_rate: 1e-5 ~ 2e-5 권장
2. max_steps: GPU 자원에 따라 조절
- Colab Free: 500
- Colab Pro+ 또는 개인 GPU: 4000
3. gradient_checkpointing: 메모리 절약
4. fp16: GPU 메모리 절약
5. generation_max_length: Whisper 토큰 길이 제한
6. push_to_hub=True: 모델 자동 업로드

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-dv",
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
```

## Trainer 구성 및 학습 시작

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()
```

Whisper-small을 500 step 훈련하면 약 45분이 걸린다.
(CPU/Low GPU에서는 더 오래 걸릴 수 있다.)

12. 결과 해석

예시 결과


| Step | WER(ortho) | WER(normalised) |
| ---- | ---------- | --------------- |
| 500  | 63.9%      | 14.1%           |


정규화된 WER 14%는
baseline Whisper-small의 126% 대비 112% 개선이다.

저자원 언어에서 7시간 데이터로 이 정도 성능을 얻는다는 것은
Whisper의 multilingual 사전학습 효율이 매우 높다는 뜻이다.

## 결과 공유 및 Hub 업로드

```python
trainer.push_to_hub(**kwargs)
```

이제 모델은 누구나 다음처럼 사용 가능하다:

```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="your-username/your-model-name")
```

14. 더 높은 성능을 위한 개선 방향

GPU가 충분하다면 다음을 고려할 수 있다.

- max_steps를 4000 이상으로 증가
- lr_scheduler_type="linear"
- learning_rate 조절
- dropout 조정
- Whisper medium 또는 large 모델로 확장
- 정규화 커스텀
- 도메인 특화 추가 데이터 수집

Whisper는 기본적으로 대규모 사전학습의 효과 덕분에
적은 데이터로도 큰 성능 향상이 가능하다.

## 결론

이 글은 Whisper ASR fine-tuning의 전 과정을 실제로 수행 가능한 코드와 함께 정리한 것이다.

핵심 포인트는 다음과 같다.

- WhisperProcessor는 feature extractor + tokenizer 통합
- 48kHz -> 16kHz downsampling 필수
- 30초 초과 샘플 잘라내기
- Data Collator는 input과 label 패딩 방식을 완전히 다르게 처리
- WER + normalised WER 모두 계산
- Hub 업로드를 통한 버전 관리

이제 어떤 언어든, 어떤 도메인이든 Whisper를 자신만의 ASR 모델로 fine-tuning할 수 있다.

참고자료
Huggingface, Audio Course, https://huggingface.co/learn