---
layout: post
title:  "Style-Bert-VITS2 해부 시리즈 – 일본어 G2P와 억양(tone): 문자열이 모델 입력 텐서가 되기까지"
date:   2026-01-19 00:10:22 +0900
categories: Style-Bert-VITS2
---

# Style-Bert-VITS2 해부 시리즈 — 일본어 G2P와 억양(tone): 문자열이 모델 입력 텐서가 되기까지

## 0. 목표

이 문장 하나를 끝까지 따라간다.

```
なんとなく、今日は静かな朝だと思った。
```

최종적으로 다음을 얻게 된다.

- phones: 음소 문자열 리스트
- tones: 억양/악센트 정수 리스트
- word2ph: 단어 -> 음소 길이 매핑
- phone_ids: phoneme vocab 기준 정수 ID
- tone_ids
- bert_features (음소 길이에 맞춰 확장됨)
- x_mask, lengths

## 1. 진입점: 텍스트가 처음으로 “처리”되는 위치

파일

```
style_bert_vits2/data_utils.py
```

- 핵심 흐름 (개념 -> 실제)
- Dataset이 filelist 한 줄을 읽는다
- text, language를 추출
- 언어별 text processing으로 위임

논리적으로는 이런 형태다:

# 의사 코드 (구조 설명용)

```python
if lang == "jp":
    phones, tones, word2ph = japanese_g2p(text)
```

**일본어 처리의 실질적인 시작점은 japanese_g2p**다.

## 2. 일본어 G2P 엔트리 포인트

파일

```
style_bert_vits2/nlp/japanese/g2p.py
```

핵심 함수

```python
def g2p(text: str, use_jp_extra: bool = False):
    ...
```

이 함수 하나가 일본어 텍스트 전처리 2편의 본체다.

## 3. Step 1 — 문자열 정규화 (normalize)
담당 파일

```
style_bert_vits2/nlp/japanese/normalizer.py
```

입력

```
なんとなく、今日は静かな朝だと思った。
```

정규화에서 하는 일

- 전각/반각 통일
- 이상한 기호 제거
- 장음/문장부호 유지
- G2P가 못 먹는 문자 제거

출력 (예시)

```
なんとなく、今日は静かな朝だと思った。
```

※ 이 문장은 비교적 깨끗해서 거의 그대로 통과한다.

## 4. Step 2 — OpenJTalk 기반 G2P
내부에서 실제로 호출되는 것

```python
pyopenjtalk.extract_fullcontext(text)
```

이 단계에서 언어학적 마술이 일어난다.

## 5. Step 3 — phoneme / tone 생성

### 결과 1: phoneme (문자열)

```python
phones = [
  "n","a","N","n","o","t","o","k","u",
  "pau",
  "k","y","o","o",
  "w","a",
  "sh","i","z","u","k","a","n","a",
  "a","s","a",
  "d","a",
  "t","o",
  "o","m","o","t","t","a"
]
```

설명:

- N : ん (비음)
- pau : 쉼(、)
- 장음/모음 연장은 그대로 phoneme으로 반영
- 문장부호는 pau로 치환됨 (중요)

### 결과 2: tone (억양 정보)

```python
tones = [
  0,0,1,1,0,0,1,0,0,
  0,
  1,0,0,
  0,
  0,1,0,0,1,0,0,
  1,0,
  0,
  1,0,
  0,0,1,1,0
]
```

의미:

- 0 / 1 (혹은 -1 포함)은 일본어 pitch accent
- phoneme 단위로 1:1 대응
- 길이가 phones와 정확히 같아야 함

> 여기서 길이 mismatch 나면 즉사

## 6. Step 4 — word2ph (단어 -> 음소 매핑)

예시:

```python
word2ph = [9, 1, 4, 2, 8, 2, 2, 5]
```

의미:


| 단어    | 대응 phoneme 개수 |
| ----- | ------------- |
| なんとなく | 9             |
| 、     | 1             |
| 今日    | 4             |
| は     | 2             |
| 静かな   | 8             |
| 朝     | 2             |
| だ     | 2             |
| 思った   | 5             |


이게 없으면 BERT를 음소 길이에 맞춰 늘릴 수 없다.

## 7. Step 5 — phoneme -> ID 변환

phoneme vocab

```
style_bert_vits2/text/phoneme_symbols.py
```

예시:

```python
"a"  -> 5
"N"  -> 32
"pau"-> 1
```

변환 결과

```python
phone_ids = [5,12,32,12,8,9,8,15,6,1, ...]
```

이제 문자열이 완전히 숫자 텐서가 된다.

## 8. Step 6 — tone / language ID 텐서화

```python
tone_ids = torch.LongTensor(tones)
lang_ids = torch.full_like(phone_ids, LANG_ID_JP)
```

## 9. Step 7 — BERT 특징 생성 & 확장

입력

- 원문 텍스트 (토큰 단위)
- BERT tokenizer

출력

```
bert_token_features: [num_tokens, hidden_dim]
```

확장

```
bert_phoneme_features: [num_phonemes, hidden_dim]
```

이때 word2ph를 사용해 token -> phoneme으로 repeat한다.

> **Style-Bert-VITS2의 “Style-Bert” 핵심**

## 10. 최종 모델 입력 텐서 묶음

```python
{
  "phones": phone_ids,
  "tones": tone_ids,
  "lang_ids": lang_ids,
  "bert": bert_phoneme_features,
  "lengths": len(phone_ids),
  "x_mask": mask
}
```

이게 TextEncoder로 들어가는 실제 입력이다.

참고 자료 
https://github.com/litagin02/Style-Bert-VITS2