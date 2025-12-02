---
layout: post
title:  "허깅페이스 오디오 트랜스포머 코스 - TTS 모델 파인튜닝"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_Audio
---

# TTS 모델 파인튜닝

이 글에서는 microsoft/speecht5_tts(영어 기반 사전학습 모델)를
**네덜란드어 음성 데이터(VoxPopuli nl subset)**로 파인튜닝하는 과정을 정리한다.

## 환경 준비

### GPU 확인

```python
# 현재 머신에 NVIDIA GPU가 있는지, 어떤 모델인지, 메모리는 얼마나 되는지 확인
nvidia-smi
```

TTS는 오디오 -> 멜 스펙트로그램 -> Transformer -> 멜 스펙트로그램
이 과정을 모두 거치기 때문에,
텍스트만 다루는 LLM보다 훨씬 메모리와 연산량이 크다.

- Colab Free: 10~15시간 분량 데이터 + 적은 step 정도가 현실적
- 로컬 GPU 16GB 이상: 예제처럼 40시간 근처까지 가능

### 필요한 패키지 설치

```python
# TTS 파인튜닝에 필요한 주요 라이브러리 설치
pip install transformers datasets soundfile speechbrain accelerate
```

- transformers : SpeechT5 모델과 Trainer 사용
- datasets : VoxPopuli 로딩, 전처리
- soundfile : 오디오 입출력 처리에 사용
- speechbrain : X-vector 기반 Speaker Embedding 추출
- accelerate : Trainer의 분산 / 하드웨어 가속에 사용

### Hugging Face Hub 로그인

```python
from huggingface_hub import notebook_login

# 브라우저에 토큰을 입력해서 로그인하면, 나중에 Trainer가 push_to_hub로 모델 업로드 가능
notebook_login()
```

## 데이터셋 준비 - VoxPopuli (네덜란드어)

### VoxPopuli nl subset 로딩

```python
from datasets import load_dataset, Audio

# VoxPopuli 네덜란드어(nl) subset의 train split 로드
dataset = load_dataset("facebook/voxpopuli", "nl", split="train")

# 전체 샘플 개수 확인
len(dataset)
```

- facebook/voxpopuli : 유럽 의회 발언 음성 데이터
- "nl" : 네덜란드어
- split="train" : 학습용 split

### 샘플링 레이트 16kHz로 통일

```python
# SpeechT5는 16kHz 오디오에 맞춰 사전학습되었으므로,
# datasets의 Audio 기능을 이용해 "audio" 컬럼을 16kHz로 리샘플
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

오디오 모델은 샘플링 속도가 불일치하면
스펙트로그램 패턴이 완전히 달라져 학습이 깨질 수 있다.

반드시 **모델이 기대하는 속도(여기서는 16kHz)**로 맞춰야 한다.

## Processor & Tokenizer 로드

SpeechT5는 Processor 하나에 **토크나이저 + 피처 추출기(멜 스펙트로그램)**가 모두 들어있다.

```python
from transformers import SpeechT5Processor

# 사용할 SpeechT5 TTS 체크포인트 이름
checkpoint = "microsoft/speecht5_tts"

# Processor 로드: 내부에 tokenizer + feature_extractor 포함
processor = SpeechT5Processor.from_pretrained(checkpoint)

# 텍스트 전처리/토큰화를 위해 tokenizer만 따로 꺼내놓기
tokenizer = processor.tokenizer
```

## 텍스트 전처리 — 지원하지 않는 문자 처리

### 데이터 예시 확인

```python
# 첫 번째 샘플 확인해서 어떤 필드들이 있는지 체크
dataset[0]
```

주요 필드

- raw_text : 원본 자막
- normalized_text : 숫자를 문자로 풀어쓴 정규화 텍스트
- audio : 오디오 경로 + waveform array + sampling_rate
- speaker_id : 화자 ID
- gender : 성별

SpeechT5 tokenizer는 숫자 토큰이 없다.
따라서 "2024" 같은 숫자보다는 "two thousand and twenty four" 식으로
풀어 쓴 normalized_text가 더 잘 맞는다.
네덜란드어에서도 normalized_text는 숫자를 문자로 변경한 상태라, 이쪽을 사용한다.

### 데이터 전체에서 사용된 문자 집합 추출

먼저 데이터셋 전체에 어떤 문자들이 나오는지 확인한다.

```python
def extract_all_chars(batch):
    """
    batch: 여러 샘플이 한번에 들어오는 배치(dict of lists).
    
    1) 이 배치의 normalized_text들을 공백으로 이어 붙여 하나의 긴 문자열(all_text)을 만들고
    2) 그 문자열에 등장하는 문자들을 set()으로 모아서 vocab 리스트로 만든다.
    
    반환 형식은 datasets.map을 위한 딕셔너리:
    - "vocab": [문자 리스트 하나]
    - "all_text": [전체 텍스트 하나]
    """
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))  # set으로 unique 문자 추출 후, 리스트로 변환
    return {"vocab": [vocab], "all_text": [all_text]}


# batched=True, batch_size=-1 -> 데이터셋 전체를 한 번에 함수에 전달
vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,                # 메모리에 유지 (속도 향상)
    remove_columns=dataset.column_names # 원래 컬럼은 제거하고 vocab/all_text만 유지
)

# 데이터셋에서 사용된 문자 집합
dataset_vocab = set(vocabs["vocab"][0])

# 토크나이저가 알고 있는 문자 집합
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

### 토크나이저가 모르는 문자 찾기

```python
# 데이터에는 있지만, tokenizer에는 없는 문자들
dataset_vocab - tokenizer_vocab
```

예시 출력

```python
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

여기서,

- 공백 ' '은 tokenizer에서 내부적으로 ▁로 처리되므로 신경 안 써도 됨
- 나머지 악센트 문자들은 SpeechT5가 모르는 문자 -> `<unk>`로 변환될 것

그래서 이 문자들을 비슷한 글자로 매핑하는 함수를 만든다.

### 문자 치환 함수 정의

```python
# (문제 문자, 대체 문자) 쌍 리스트
replacements = [
    ("à", "a"),
    ("ç", "c"),
    ("è", "e"),
    ("ë", "e"),
    ("í", "i"),
    ("ï", "i"),
    ("ö", "o"),
    ("ü", "u"),
]


def cleanup_text(inputs):
    """
    하나의 샘플(dict)에서 normalized_text에 포함된 특수 문자를
    위에서 정의한 replacements 규칙에 따라 치환한다.
    
    - inputs: {"normalized_text": "...", ...} 형태의 딕셔너리
    - 반환: 수정된 inputs
    """
    for src, dst in replacements:
        inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
    return inputs


# 전체 데이터셋에 cleanup_text 적용
dataset = dataset.map(cleanup_text)
```

이렇게 해야 `<unk>` 토큰이 무분별하게 많아지는 것을 막을 수 있고,
발음/리듬이 훨씬 안정적으로 나온다.

## 화자 분포 확인 및 필터링

VoxPopuli는 화자가 많다.
하지만 어떤 화자는 샘플이 10개도 안 되고, 어떤 화자는 1,000개 넘게 있을 수 있다.

이 상태 그대로 학습하면,

- 샘플이 많은 소수 화자 중심으로 모델이 오버피팅
- 샘플이 적은 화자는 목소리 재현이 불안정

그래서 적당한 범위의 화자만 남긴다.

### 화자별 샘플 개수 세기

```python
from collections import defaultdict

# speaker_id → 해당 화자의 예제 수
speaker_counts = defaultdict(int)

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1
```

### 히스토그램으로 화자 당 샘플 수 시각화

```python
import matplotlib.pyplot as plt

plt.figure()
plt.hist(speaker_counts.values(), bins=20)  # 화자별 예제 수 분포
plt.ylabel("Speakers")   # y축: 화자 수
plt.xlabel("Examples")   # x축: 각 화자가 가진 예제 개수
plt.show()
```

결과적으로,

- 약 1/3의 화자는 예제 수 < 100
- 약 10명 정도는 예제 수 > 500

### 너무 적거나 너무 많은 화자 제외

```python
def select_speaker(speaker_id):
    """
    speaker_id를 받아서, 
    그 화자의 예제 개수가 100개 이상 400개 이하인 경우만 True 반환.
    
    - True인 샘플만 데이터셋에 남기게 된다.
    """
    return 100 <= speaker_counts[speaker_id] <= 400


# speaker_id 컬럼을 기준으로 필터 적용
dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

필터링 적용 후의 화자 수와 예제 수

```python
# 남은 고유 화자 수
len(set(dataset["speaker_id"]))

# 남은 전체 예제 수
len(dataset)
```

- 약 40명 내외의 화자
- 예제 약 1만 개 미만 -> 멀티스피커 TTS 학습에 적당한 규모

## Speaker Embeddings (X-vector) 생성

이제 각 오디오에 대해 **Speaker Embedding (X-vector)**를 만들어야 한다.
이 벡터가 “이 화자의 목소리”를 대표한다.

### SpeechBrain X-vector 모델 로드

```python
import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

# GPU가 있으면 "cuda", 없으면 "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# X-vector 추출용 사전학습 화자 인식 모델 로드
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,                        # 사용할 모델 이름
    run_opts={"device": device},                  # 사용할 디바이스
    savedir=os.path.join("/tmp", spk_model_name), # 모델 가중치를 저장할 로컬 경로
)
```

### X-vector 생성 함수

```python
def create_speaker_embedding(waveform):
    """
    주어진 waveform(1D numpy array 또는 list)을 입력으로 받아
    SpeechBrain X-vector 모델을 통해 512차원의 speaker embedding을 반환.
    
    - waveform: 오디오 샘플 배열 (16kHz)
    - 반환: (512,) shape의 numpy array
    """
    with torch.no_grad():  # 추론 모드, gradient 계산 비활성화
        # encode_batch는 (batch, time) 형태 텐서를 기대하므로,
        # torch.tensor(waveform)으로 변환 후 내부에서 배치 차원 추가
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        
        # speaker embedding을 L2 normalization (길이를 1로 정규화)
        # 이렇게 하면 embedding이 구면 위에 분포 -> 거리 계산에 안정적
        speaker_embeddings = torch.nn.functional.normalize(
            speaker_embeddings, dim=2
        )
        
        # (1, 1, 512) 같은 형태라면 squeeze()로 (512,)로 축소
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings
```

실제로는 X-vector 모델이 VoxCeleb(영어)로 학습되었지만,
화자의 음색, 성문 특성은 언어와 어느 정도 독립적이므로
네덜란드어 데이터에서도 충분히 쓸 수 있다.

### 독자적인 X-vector 모델 작성 템플릿

```python
import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def process(args):
    wavlst = []
    for split in args.splits.split(","):
        wav_dir = os.path.join(args.arctic_root, split)
        wavlst_split = glob.glob(os.path.join(wav_dir, "wav", "*.wav"))
        print(f"{split} {len(wavlst_split)} utterances.")
        wavlst.extend(wavlst_split)

    spkemb_root = args.output_root
    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=args.speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[args.speaker_embed]
    for utt_i in tqdm(wavlst, total=len(wavlst), desc="Extract"):
        # TODO rename speaker embedding
        utt_id = "-".join(utt_i.split("/")[-3:]).replace(".wav", "")
        utt_emb = f2embed(utt_i, classifier, size_embed)
        numpy.save(os.path.join(spkemb_root, f"{utt_id}.npy"), utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", "-i", required=True, type=str, help="LibriTTS root directory.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Output directory.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker emebdding.")
    parser.add_argument("--splits",  type=str, help="Split of four speakers seperate by comma.",
                        default="cmu_us_bdl_arctic,cmu_us_clb_arctic,cmu_us_rms_arctic,cmu_us_slt_arctic")
    args = parser.parse_args()
    print(f"Loading utterances from {args.arctic_root}/{args.splits}, "
        + f"Save speaker embedding 'npy' to {args.output_root}, "
        + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    process(args)

if __name__ == "__main__":
    """
    python utils/prep_cmu_arctic_spkemb.py \
        -i /root/data/cmu_arctic/CMUARCTIC \
        -o /root/data/cmu_arctic/CMUARCTIC/spkrec-xvect \
        -s speechbrain/spkrec-xvect-voxceleb
    """
    main()
```

## SpeechT5를 위한 최종 전처리 함수

이제 하나의 예제를 SpeechT5가 기대하는 형태로 바꾸는 함수를 만든다.

### prepare_dataset 함수

```python
def prepare_dataset(example):
    """
    하나의 example(dict)을 입력받아
    - 텍스트 -> input_ids (토큰)
    - 오디오 -> 멜 스펙트로그램(labels)
    - Speaker Embedding -> speaker_embeddings
    를 생성해서 반환한다.
    
    반환 딕셔너리 키:
    - "input_ids": 텍스트 토큰(1D tensor)
    - "labels":   log-mel spectrogram (T, 80) tensor
    - "stop_labels": 종료 토큰용 라벨(현재 사용 X)
    - "speaker_embeddings": (512,) numpy array
    """
    audio = example["audio"]  # {"array": waveform, "sampling_rate": 16000, ...}
    
    # SpeechT5Processor를 사용해
    # - text: normalized_text를 텍스트 입력으로 사용
    # - audio_target: 오디오 배열을 멜 스펙트로그램으로 변환
    # - sampling_rate: 샘플링 레이트를 명시
    # 반환값에는:
    #   - "input_ids": 텍스트 토큰
    #   - "labels": [0]에 (T, 80) log-mel spectrogram (배치 차원 포함)
    #   - "stop_labels": 종료 시점 예측용
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,  # 여기서는 attention_mask는 사용하지 않음
    )
    
    # processor가 batch 차원을 추가해서 반환하므로 labels는 shape (1, T, 80)
    # 우리는 배치 없이 (T, 80) 형식으로만 저장하면 되므로 첫 번째 차원 제거
    example["labels"] = example["labels"][0]
    
    # Speaker Embedding 생성 (X-vector)
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
    
    return example
```

### 한 샘플에 대해 확인

```python
processed_example = prepare_dataset(dataset[0])

# 어떤 키들이 있는지 확인
list(processed_example.keys())
# ['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']

# speaker_embeddings shape 확인 → (512,)
processed_example["speaker_embeddings"].shape

# labels는 (T, 80) 형태의 log-mel spectrogram
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(processed_example["labels"].T)  # (T, 80)를 (80, T)로 transpose 후 이미지로 보여줌
plt.title("Log-mel Spectrogram (80 mel bins)")
plt.show()
```

## 데이터 전체에 전처리 적용 및 길이 필터링

### 전체 데이터셋에 prepare_dataset 적용

```python
# 기존 컬럼들은 모두 제거하고,
# prepare_dataset이 반환한 "input_ids", "labels", "stop_labels", "speaker_embeddings"만 남긴다.
dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset.column_names,
)
```

이 과정은 오디오 멜 스펙트로그램 변환까지 포함하기 때문에
약 5~10분 정도 걸릴 수 있다.

### 너무 긴 입력 제거

SpeechT5는 최대 600 토큰까지 처리하도록 되어 있지만,
여기서는 배치 크기를 키우기 위해 200 토큰 이상은 제거한다.

```python
def is_not_too_long(input_ids):
    """
    input_ids(토큰 리스트)의 길이가 200 미만일 때만 True.
    길이가 200 이상이면 학습에서 제외.
    """
    input_length = len(input_ids)
    return input_length < 200

# input_ids 기준으로 필터링
dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])

len(dataset)  # 필터링 후 남은 샘플 수
```

### Train/Test 분할

```python
# 10%를 test로 사용
dataset = dataset.train_test_split(test_size=0.1)
```

- dataset["train"] : 90% 학습용
- dataset["test"] : 10% 검증/테스트용

## Data Collator

Transformer에 학습 데이터를 넣으려면
여러 샘플을 하나의 batch 텐서로 묶어야 한다.

문제:

- 각 샘플의 input_ids 길이가 다 다름
- 각 샘플의 labels (멜 스펙트로그램 길이)도 다 다름
- Speaker Embedding만 고정 길이(512)

그래서 다음을 해야 한다.

- input_ids를 padding해서 길이 맞추기
- labels를 padding해서 길이 맞추기
- padding된 부분을 -100으로 채워 loss 무시
- reduction_factor=2에 맞게 길이를 조정하는 로직

이 모든걸 `__call__` 안에서 처리하는 커스텀 collator를 만든다.

### Data Collator 정의

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class TTSDataCollatorWithPadding:
    """
    Text-to-Speech용 커스텀 Data Collator.
    
    - processor: SpeechT5Processor (토크나이저 + 피처 패딩 기능 포함)
    
    __call__ 메서드는 Trainer가 자동으로 호출하며,
    List[feature_dict] -> batch_dict 형태로 변환한다.
    """
    processor: Any

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        features: 개별 샘플들이 들어 있는 리스트.
        각 샘플은 prepare_dataset에서 만든 dict 형식이다.
        
        반환: 배치 텐서를 담은 dict
          - input_ids: (B, T_text)
          - labels:    (B, T_mel, n_mels)
          - speaker_embeddings: (B, 512)
        """
        # 1) 텍스트 토큰 추출: processor.pad가 기대하는 형식으로 dict 안에 "input_ids" 키를 둔다.
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        
        # 2) 멜 스펙트로그램 라벨 추출: processor가 "input_values"라는 키를 기대하므로 이름 변환
        label_features = [{"input_values": feature["labels"]} for feature in features]
        
        # 3) Speaker Embedding은 그대로 리스트로 모은다. (각 원소 shape: (512,))
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # 4) processor.pad를 사용해
        #    - input_ids는 토큰 padding
        #    - labels는 음성(target) padding
        #    - return_tensors="pt"로 PyTorch 텐서 반환
        batch = self.processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        # 5) labels padding 부분을 -100으로 채우기
        #    - decoder_attention_mask: 1이면 실제 데이터, 0이면 padding
        #    - masked_fill(..., -100): padding 위치를 -100으로 설정
        batch["labels"] = batch["labels"].masked_fill(
            batch["decoder_attention_mask"].unsqueeze(-1).ne(1),  # True: 패딩 위치
            -100,
        )

        # 6) decoder_attention_mask는 파인튜닝 시 loss 계산에는 쓰지 않으므로 제거
        del batch["decoder_attention_mask"]

        # 7) reduction_factor에 맞춰 target 길이 조정
        #    - SpeechT5는 config.reduction_factor=2로,
        #      decoder가 두 프레임씩 묶어서 처리한다.
        #    - 그래서 target 길이가 항상 reduction_factor의 배수가 되도록 잘라줘야 한다.
        if model.config.reduction_factor > 1:
            # 각 샘플의 target 길이(=멜 스펙트로그램 길이)를 모으기
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            # 각 길이를 reduction_factor의 배수로 내림
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            
            # 배치 전체의 labels를 max_length까지 잘라서
            # 모든 샘플의 타겟 길이를 동일하게 맞춘다.
            batch["labels"] = batch["labels"][:, :max_length]

        # 8) Speaker Embedding을 (B, 512) 텐서로 변환해서 배치에 추가
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


# collator 인스턴스 생성
data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## 모델 로드 및 학습 설정

### SpeechT5 모델 로드

```python
from transformers import SpeechT5ForTextToSpeech

# 사전학습된 SpeechT5 TTS 모델 로드
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

### use_cache 설정

```python
from functools import partial

# 학습 시에는 gradient checkpointing과 use_cache가 충돌할 수 있음.
# 따라서 학습 동안은 cache 비활성화
model.config.use_cache = False

# 추론(generation)을 할 때는 다시 cache를 켜서 속도를 올린다.
# model.generate를 partial로 싸서 use_cache=True를 기본 인자로 준다.
model.generate = partial(model.generate, use_cache=True)
```

### TrainingArguments 설정

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="speecht5_finetuned_voxpopuli_nl",  # 체크포인트 저장 위치 (또는 Hub repo 이름)
    per_device_train_batch_size=4,                 # GPU 한 장당 batch size
    gradient_accumulation_steps=8,                 # 4*8=32 효과 (메모리 절약)
    learning_rate=1e-5,                            # TTS에서는 작은 lr 권장
    warmup_steps=500,                              # 워밍업 스텝 수
    max_steps=4000,                                # 전체 학습 스텝 수
    gradient_checkpointing=True,                   # 메모리 절약용
    fp16=True,                                     # half precision으로 속도 & 메모리 절약
    eval_strategy="steps",                         # 일정 스텝마다 eval
    per_device_eval_batch_size=2,                  # 평가용 배치 크기
    save_steps=1000,                               # 1000 step마다 체크포인트 저장
    eval_steps=1000,                               # 1000 step마다 평가
    logging_steps=25,                              # 25 step마다 로그 기록
    report_to=["tensorboard"],                     # TensorBoard 로깅
    load_best_model_at_end=True,                   # validation loss 기준 best 모델 로드
    greater_is_better=False,                       # loss는 낮을수록 좋으므로 False
    label_names=["labels"],                        # Trainer가 loss 계산 시 참조할 라벨 이름
    push_to_hub=True,                              # 학습 종료 후 Hub에 업로드
)
```

### Trainer 생성 및 학습 시작

```python
from transformers import Seq2SeqTrainer

# HuggingFace Trainer 설정
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,  # 우리가 만든 custom collator
    tokenizer=processor,          # 로그/저장을 위해 processor 전달
)

# 학습 시작 (몇 시간 걸릴 수 있음)
trainer.train()

# 최종 모델을 Hugging Face Hub에 업로드
trainer.push_to_hub()
```

만약 CUDA out-of-memory 에러가 난다면
per_device_train_batch_size를 반으로 줄이고
gradient_accumulation_steps를 두 배로 늘리는 방식으로 조정한다.

## Fine-tuned 모델로 Inference 수행

### Hub에서 Fine-tuned 모델 로드

```python
from transformers import SpeechT5ForTextToSpeech

# "YOUR_ACCOUNT"를 자신의 HF 계정 이름으로 변경
model = SpeechT5ForTextToSpeech.from_pretrained(
    "YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl"
)
```

### 테스트 예제에서 Speaker Embedding 가져오기

```python
import torch

# 테스트 데이터셋의 한 예제를 가져옴
example = dataset["test"][304]

# example에는 이미 "speaker_embeddings"가 포함되어 있음 (prepare_dataset에서 추가)
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
# shape: (1, 512) -> 배치 차원 추가
```

### 입력 텍스트 전처리

```python
# 네덜란드어 문장 예제
text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"

# processor를 사용해 텍스트를 토큰화 (input_ids 생성)
inputs = processor(text=text, return_tensors="pt")
```

### Vocoder 로드 및 음성 생성

```python
from transformers import SpeechT5HifiGan

# HiFi-GAN 기반 vocoder 로드
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# generate_speech:
# - input_ids: 텍스트 토큰
# - speaker_embeddings: 화자 임베딩
# - vocoder: 멜 스펙트로그램 → waveform 변환
speech = model.generate_speech(
    inputs["input_ids"],
    speaker_embeddings,
    vocoder=vocoder
)
```

### 결과 재생

```python
from IPython.display import Audio

# SpeechT5는 16kHz를 사용
Audio(speech.numpy(), rate=16000)
```

## 결과 품질이 안 좋을 때 체크할 것들

1. Speaker Embedding 품질
- 너무 짧은 음성으로 X-vector를 만든 경우
- 발화 내용이 너무 특이하거나 잡음이 심한 경우

다른 샘플에서 speaker embedding을 뽑아보자.

2. 학습 스텝 부족

- 4000 step은 실험용 수준
- 8000~20000 step까지 늘리면 훨씬 좋아질 수 있다.

3. reduction_factor

- model.config.reduction_factor = 1로 변경 후 다시 학습
- 더 세밀한 타임스텝으로 멜 스펙트로그램 예측 가능

4. 데이터 품질

- 잡음이 많은 샘플 필터링
- 발음이 너무 극단적이거나 오류가 있는 자막 제거

## 마무리

- 텍스트 전처리: tokenizer 미지원 문자 처리 필수
- Speaker Embedding(X-vector): 멀티스피커 TTS의 핵심
- Processor: 텍스트 + 오디오를 한 번에 처리
- Data Collator: padding, masking, reduction_factor 처리
- Trainer: 작은 학습률 + gradient accumulation + fp16
- Inference: speaker_embeddings + HiFi-GAN vocoder로 waveform 생성


참고자료
Huggingface, Audio Course, https://huggingface.co/learn