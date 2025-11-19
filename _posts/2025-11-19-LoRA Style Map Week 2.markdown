---
layout: post
title:  "CLIP 임베딩 기반 스타일 분석 프로젝트 2주차"
date:   2025-11-19 00:10:22 +0900
categories: LoRA Style Map
---

# 들어가며

이전 글에서는 CivitAI LoRA 이미지들을 CLIP 임베딩으로 바꾸는 과정까지 정리했다.  

이번 글에서는 드디어 그 임베딩을 가지고 실제로 스타일 군집화를 수행한 과정을 기록한다.

---

# 고차원 임베딩을 낮추는 이유

CLIP 임베딩은 512차원이다.
하지만 군집화(KMeans), 시각화(UMAP)를 하기 위해서는 차원 축소가 필수다.

나는 다음 파이프라인을 사용했다.

512D (CLIP) -> 50D (PCA) -> 2D (UMAP) -> 군집화(KMeans)

---

# PCA – 차원 축소

먼저 PCA로 차원을 512에서 50로 압축한다.

```python
pca = PCA(n_components=50, random_state=42)
pca_features = pca.fit_transform(features)
```


50차원으로 압축하면 노이즈가 줄고
핵심 스타일 축만 남게 된다.

---

# UMAP – 시각화 가능한 스타일 맵 만들기

UMAP은 t-SNE보다 빠르면서도 “지형학적 구조”를 잘 살리는 기법이다.

```python
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_features = umap_model.fit_transform(pca_features)
```

시각화를 통해 "스타일 맵(Style Map)"을 형성할 수 있다.

---

# KMeans 군집화 — 스타일 카테고리 자동 생성

이 지도 위에서 KMeans로 군집을 나눴다.

```python
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(umap_features)
```


몇 개 군집이 가장 적절한지는 Elbow/Silhouette도 테스트했는데, 테스트 결과 8개가 균형점이었다.

---

# Centroid를 저장하는 이유

각 클러스터의 중심점을 다음과 같이 저장했다:

cluster_centroids.npy

이것이 왜 필요할까?

바로 다음 기능을 위해서다:

“프롬프트를 입력하면 가장 비슷한 스타일의 LoRA 클러스터를 추천”

텍스트 프롬프트 -> CLIP 텍스트 임베딩 ->
각 클러스터 중심점과 코사인 유사도 -> 가장 가까운 군집 추천

이 파이프라인은 프롬프트 기반 스타일 추천의 기반이 된다.

---

# Streamlit UI 구현

프로토타입 UI는 네 가지 탭으로 구성된다.

- 프롬프트 기반 스타일 추천
- LoRA ID 기반 클러스터링
- UMAP 스타일 맵
- 클러스터 예시 브라우저

---

# 마무리

LoRA는 embedding 공간에서 좀 더 효과적으로 모인다

CLIP 텍스트 임베딩과 이미지 임베딩은 꽤 정교하게 alignment 되어 있다

---

# 추후 확장 가능성

- LoRA Fusion Lab

원하면 이 과정들도 계속 기록할 계획이다.