---
layout: post
title:  "Style-Bert-VITS2 학습기 3"
date:   2025-12-10 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 TextEncoder

## 목표

- TextEncoder 코드에서 “이 줄을 주석 처리하면 어떤 소리가 망가질지” 식별
- BERT / style / tone / language가 어디서 합쳐지고, 왜 거기서 합쳐지는지 확인
- 파인튜닝 시 TextEncoder를 freeze 할지 말지 판단

## TextEncoder가 등장하는 위치

Style-Bert-VITS2에서 TextEncoder는 보통 이런 형태로 선언된다
(파일명은 구현마다 다를 수 있으나 구조는 거의 동일).

```python
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
        **kwargs
    ):
```
이 클래스의 책임

```
“텍스트 + 조건들을 받아서,
이후 음성 생성에 필요한 ‘발화 설계도’를 만든다.”
```

## __init__ 한 줄씩 해부

### 기본 임베딩: 텍스트 토큰

```python
self.emb = nn.Embedding(n_vocab, hidden_channels)
```

이 줄의 의미

- 텍스트 토큰 ID -> 벡터
- “단어를 의미 벡터로 바꾸는 첫 단계”
- shape: (B, T_text) -> (B, T_text, hidden_channels)

이 줄이 없으면?

- 텍스트가 숫자로만 남음
- Transformer가 아무 의미도 못 잡음
- 발음 이전에 모델이 붕괴

### Tone 임베딩 (일본어 억양)

```python
self.tone_emb = nn.Embedding(num_tones, hidden_channels)
```

이 줄의 의미

- 일본어의 고저 악센트 힌트
- 같은 글자라도 억양이 달라질 수 있음

이 줄이 없으면?

- 일본어가 “외국인 억양”처럼 들릴 가능성 증가
- 문장 끝 상승/하강이 단조로워짐

### Language 임베딩

```python
self.lang_emb = nn.Embedding(num_languages, hidden_channels)
```

이 줄의 의미

- 다국어 모델 대비
- 일본어 단일이면 거의 상수

이 줄이 없으면?

- 단일 언어 모델이면 사실상 큰 문제 없음
- 다국어 모델에서는 발음 규칙 충돌 가능

### BERT 특징 투영 (중요)

```python
self.bert_proj = nn.Conv1d(
    bert_dim,
    hidden_channels,
    kernel_size=1
)
```

이 줄의 의미

- BERT 출력

```python
(B, T_text, bert_dim)
```

- Conv1d로:

```python
(B, bert_dim, T_text) -> (B, hidden_channels, T_text)
```

왜 Linear가 아니라 Conv1d인가?

- 시간축(T_text)을 유지하면서 채널만 바꾸기 위함
- Transformer 입력과 정렬 맞추기 쉬움

이 줄이 없으면?

- BERT 특징을 못 씀
- 말줄임표, 망설임, 문맥 차이 무시
- 캐릭터 연기 거의 사라짐

### Style Vector 투영

```python
self.style_proj = nn.Linear(style_dim, hidden_channels)
```

이 줄의 의미

- style vector (256-d 등)를
- TextEncoder hidden 공간으로 변환

중요한 특징

- style vector는 시간 축이 없음
- 이후에 expand로 T_text만큼 늘려짐

이 줄이 없으면?

- 모든 문장이 항상 같은 분위기
- “불안 / 속삭임 / 분노” 같은 차이 사라짐

### Transformer Encoder 본체

```python
self.encoder = attentions.Encoder(
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout
)
```

이 줄의 의미

- 앞에서 만든 모든 조건을 서로 관계 맺게 만드는 핵심

Transformer가 하는 질문

“이 단어는 앞 단어와 어떤 관계지?”
“지금 스타일이 이 단어에 영향을 주나?”
“문장 끝에서 톤을 올릴까?”

이 줄이 없으면?

- 모든 토큰이 독립적으로 행동
- 리듬, 억양, 감정 전부 붕괴

### Prior 출력 레이어

```python
self.proj = nn.Conv1d(
    hidden_channels,
    out_channels * 2,
    kernel_size=1
)
```

이 줄의 의미

- TextEncoder의 출력 -> 분포 파라미터
- out_channels * 2 이유:
  - 평균 m
  - 로그 분산 logs

이 줄이 없으면?

- posterior/flow와 연결 불가
- 음성 생성 자체가 안 됨

## forward() 한 줄씩 해부

### 텍스트 임베딩

```python
x = self.emb(text)
```

shape:

- (B, T_text, hidden)

### tone / language 임베딩 추가
x = x + self.tone_emb(tone) + self.lang_emb(lang)

여기서 중요한 개념

- “조건은 더하는 것”
- concat이나 gating이 아닌 같은 공간에서 합산
- 이 구조 덕분에 tone이 특정 단어에만 영향을 줄 수 있음

### BERT 특징 추가

```python
bert = self.bert_proj(bert_feats.transpose(1, 2))
x = x + bert.transpose(1, 2)
```

여기서 무슨 일이 일어나는가

- BERT 문맥 정보가
- 텍스트 토큰 임베딩에 직접 더해짐

“이 단어는 망설이는 맥락이다”
-> 그 정보가 해당 토큰 벡터에 섞인다

### Style Vector 추가

```python
style = self.style_proj(style_vec)
style = style.unsqueeze(-1).expand(-1, -1, x.size(2))
x = x + style
```

이 코드의 의미

- style은 문장 전체에 동일하게 적용
- 분위기/감정은 “전역 상태”

이 줄이 왜 중요한가

- 문장 중간에서 감정 바뀌는 걸 방지
- 캐릭터 일관성 유지

### Transformer 통과

```python
x = self.encoder(x, x_mask)
```

여기서 일어나는 일

- 단어 <-> 단어 관계
- 스타일 <-> 단어 상호작용
- 문맥 흐름 형성

여기서 발화 리듬이 결정된다

### Prior 분포 생성

```python
stats = self.proj(x)
m, logs = torch.split(stats, self.out_channels, dim=1)
```

의미

- 이제 TextEncoder의 역할 끝
- “이 텍스트는 이런 분포의 소리를 가질 것이다”