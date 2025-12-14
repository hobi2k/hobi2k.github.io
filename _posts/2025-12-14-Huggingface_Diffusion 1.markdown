---
layout: post
title:  "허깅페이스 디퓨전 코스 - Unit 1 & Unit 2"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_diffusion
---

# Diffusion Course Unit 1 & Unit 2

## 목표

- Unit 1: Diffusers로 **DDPM을 scratch 학습**하고, pipeline/샘플링 루프까지 직접 구성  
- Unit 2: **pretrained DDPM 로드 → DDIM으로 빠른 샘플링 → fine-tuning → guidance(추론 시 loss로 조종)**


## 공통 전제: Diffusers 코드가 “모듈 분리”를 강제하는 이유

Diffusers에서 diffusion 시스템은 보통 다음 3개로 쪼갠다.

- `UNet2DModel` (또는 pipeline 내부 `unet`): **모델**, 입력(노이즈 포함 이미지 + t) -> 예측(대개 noise ε) 
- `DDPMScheduler` / `DDIMScheduler`: **규칙 엔진**, `add_noise()`(학습용 forward)와 `step()`(추론용 reverse 업데이트) 담당 
- `DDPMPipeline`: 위 둘을 묶어 “제품”처럼 호출 가능하게 만든 실행 래퍼

즉, “모델이 학습되는 대상”과 “확률 과정 업데이트 규칙”을 분리해두면

- 학습은 같게 유지하면서
- 샘플링 방법(DDPM/DDIM/Euler 등)을 바꾸는 실험을 쉽게 한다.

# Unit 1: Introduction to Diffusers (코드 독해/재구현)

## Dataset -> Transform -> DataLoader

원문은 butterfly subset을 쓰고, **32×32**로 다운샘플링해 학습 시간을 줄인다. 

```python
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
image_size = 32
batch_size = 64

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # (H,W,C) PIL -> (C,H,W) float [0,1]
    transforms.Normalize([0.5], [0.5]), # [0,1] -> [-1,1]
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

구현 포인트

- diffusion 학습/샘플링은 대부분 [-1, 1] 정규화를 기대한다. (Normalize([0.5],[0.5])) 
- DataLoader가 내놓는 배치는 batch["images"]로 접근한다. 
- 텐서 shape는 (B,3,32,32).

## Scheduler 정의: “노이즈를 어떻게 더하고/빼는가”

Unit 1은 학습/샘플링 모두 DDPM 기본 스케줄러로 시작한다.

```python
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
Hugging Face
```

핵심 API

- add_noise(x0, noise, t): 학습용 forward process (x0 -> xt)
- step(model_pred, t, xt).prev_sample: 추론용 reverse update (xt → x(t-1))

## UNet2DModel 정의: “무엇을 예측하는 모델인가”

원문 구성(채널/블록/attention 포함)이 그대로 아래와 같다. 

```python
model = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
).to(device)
```

여기서 “out_channels=3”의 의미

Unit 1 루프에서 모델 output은 noise ε(= 입력에 더한 noise와 동일 shape)를 맞추도록 학습된다.
그래서 출력도 입력과 같은 채널(3)로 둔다. (loss가 mse(noise_pred, noise)) 

## Training Loop: “noise 예측 MSE”가 표준 목적함수

Unit 1의 루프는 diffusion 학습의 정석 템플릿이다. 

```python
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

for epoch in range(30):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device) # x0
        noise = torch.randn(clean_images.shape, device=device)  # ε
        bs = clean_images.shape[0]

        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=device
        ).long() # t ~ Uniform

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)  # xt

        noise_pred = model(noisy_images, timesteps, return_dict=False)[0] # ε̂

        loss = F.mse_loss(noise_pred, noise) # MSE(ε̂, ε)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

다시 짤 때 반드시 보는 디테일

- timesteps는 배치마다 “샘플별로” 다르게 뽑는다. (shape=(B,)) 
- add_noise는 내부적으로 t별 α/β를 적용해서 xt를 만든다.
- model(noisy_images, timesteps)에서 timestep이 conditioning으로 들어간다.
- return_dict=False면 튜플로 나오며 [0]이 sample이다. 

## Pipeline 구성/저장: “재현 가능한 artifact” 만들기

Unit 1은 학습한 model과 noise_scheduler를 pipeline으로 묶는다. 

```python
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
image_pipe.save_pretrained("my_pipeline")
```

저장 구조가 바로 핵심이다:

- my_pipeline/unet/{config.json, diffusion_pytorch_model.bin}
- my_pipeline/scheduler/...
- my_pipeline/model_index.json 

즉, “코드 없이도” pipeline을 재구성할 수 있는 최소 메타데이터가 저장된다.

## Sampling Loop: pipeline 내부가 결국 이 for-loop

Unit 1은 pipeline을 쓰지 않고도, 핵심 sampling 루프를 직접 보여준다. 

```python
sample = torch.randn(8, 3, 32, 32).to(device) # x_T
for i, t in enumerate(noise_scheduler.timesteps): # T -> 0
    with torch.no_grad():
        residual = model(sample, t).sample  # ε̂
    sample = noise_scheduler.step(residual, t, sample).prev_sample  # x_{t-1}
```

이걸 이해하면 pipeline을 “블랙박스”로 쓰지 않고,

- 중간 단계 시각화
- custom guidance 삽입
- sampler 교체 같은 확장이 가능해진다.

# Unit 2: Fine-Tuning and Guidance (코드 독해/재구현)

## Pretrained pipeline 로드

```python
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)
images = image_pipe().images
```

포인트

- image_pipe.unet은 이미 학습된 UNet
- image_pipe.scheduler도 같이 들어있다
- 기본 호출은 편하지만 느리다(기본 DDPM step 수가 큼) 

## Faster Sampling: DDIMScheduler로 샘플링 루프 재작성

Unit 2의 핵심은 “scheduler가 sampler다”를 코드로 증명하는 것. 

```python
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)

x = torch.randn(4, 3, 256, 256).to(device)
for i, t in enumerate(scheduler.timesteps):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample
```

여기서 중요한 디테일

- scale_model_input(x, t)는 scheduler마다 필요한 정규화/스케일링을 적용한다. 
- scheduler.step(...) 결과는 객체이며, prev_sample로 업데이트한다. 
- 일부 scheduler는 pred_original_sample도 제공한다(중간 복원 시각화에 유용). 

## Fine-tuning loop: 바뀌는 건 “optimizer target” 뿐

Unit 2는 pretrained pipeline의 unet만 업데이트 대상으로 잡는다. 

```python
optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=1e-5)
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        noise = torch.randn(clean_images.shape, device=device)
        bs = clean_images.shape[0]

        timesteps = torch.randint(
            0, image_pipe.scheduler.num_train_timesteps, (bs,), device=device
        ).long()

        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]
        # (이 아래에서 loss/backprop/optimizer.step이 이어진다)
```

구현 체크포인트

- Unit 1과 거의 동일하되, noise_scheduler 대신 image_pipe.scheduler를 쓰는 이유:
    - pretrained pipeline이 가진 학습 설정/스케줄과 정합성을 유지하기 위해서다. 
- 학습이 잘 되는지 확인하려면 “loss만” 보지 말고 중간 샘플을 주기적으로 생성해야 한다.

## Fine-tuning 스크립트화 포인트

Unit 2는 notebook 루프를 스크립트로 뽑아 logging을 붙인 finetune_model.py를 별도 제공한다고 명시한다. 

그리고 CLI 실행 예시도 있다:

```bash
python finetune_model.py \
  --image_size 128 --batch_size 8 --num_epochs 16 \
  --grad_accumulation_steps 2 \
  --start_model "google/ddpm-celebahq-256" \
  --dataset_name "Norod78/Vintage-Faces-FFHQAligned" \
  --log_samples_every 100 --save_model_every 1000 \
  --model_save_name "vintageface"
```

## Save/Load/Upload: Unit 1과 완전히 같은 방식

```python
image_pipe.save_pretrained("my-finetuned-model")
```

Hub 업로드 코드 또한 Unit 1과 거의 동일한 패턴으로 제시된다. 

## Guidance: “추론 시 loss의 gradient로 x를 조종”

Unit 2는 guidance를 “모델을 바꾸지 않고 sampling 중간 변수 x를 loss gradient로 밀어주는 기법”으로 설명하고,
색상 loss 예제로 시작한다. 

### conditioning loss 정의

```python
def color_loss(images, target_color=(0.1, 0.9, 0.5)):
    target = torch.tensor(target_color).to(images.device) * 2 - 1  # [-1,1]로 맞춤
    target = target[None, :, None, None]
    return torch.abs(images - target).mean()
```

### guided sampling loop의 핵심: x.requires_grad_(True) + autograd.grad(loss, x)

```python
x0 = scheduler.step(noise_pred, t, x).pred_original_sample
loss = color_loss(x0) * guidance_loss_scale
cond_grad = -torch.autograd.grad(loss, x)[0]
x = x.detach() + cond_grad
x = scheduler.step(noise_pred, t, x).prev_sample
```

- x는 “현재 timestep의 latent(여기서는 pixel-space) 샘플”이다.
- 모델은 noise_pred만 예측한다.
- 그런데 우리는 “원하는 속성(color)”을 만족시키고 싶다.
- 그래서 pred_original_sample(예측된 x0)에 loss를 걸고,
- 그 loss의 gradient를 x 방향으로 역전파해 x를 조정한 뒤,
- 그 조정된 x로 다시 scheduler step을 진행한다.
- 즉, guidance는 구조적으로 “sampling loop를 커스터마이징”하는 기술이다.


참고자료
Huggingface, Diffusion Course, https://huggingface.co/learn