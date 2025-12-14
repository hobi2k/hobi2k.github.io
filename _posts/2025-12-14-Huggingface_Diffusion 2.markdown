---
layout: post
title:  "허깅페이스 디퓨전 코스 - Unit 3"
date:   2025-12-02 00:10:22 +0900
categories: Huggingface_diffusion
---

# Diffusion Course Unit 3

## Unit 3 범위

이 글에서는 기존 pipelines를 활용해 생성/편집을 수행하는 방법, 
그리고 pipeline 내부의 핵심 구성요소를 “보이는 수준”까지 꺼내서 확인한다. 

1. StableDiffusionPipeline로 text-to-image 생성 + 주요 인자 실험
2. pipeline 구성요소 확인: VAE / tokenizer+text encoder / UNet / scheduler
3. pipeline 구성요소로 DIY sampling loop 재현
4. Img2Img, Inpainting, Depth2Img 파이프라인 사용
5. Inpainting은 **DIY inpainting loop(수동 구현)**

## Setup: 설치/임포트/데모 이미지 로드

### 설치

노트북은 diffusers + ftfy + accelerate를 설치하고, Depth2Img에 필요한 최신 transformers를 소스 설치한다. 

```python
%pip install -Uq diffusers ftfy accelerate
%pip install -Uq git+https://github.com/huggingface/transformers
```

실전 팁(재현성):

- “소스 설치”가 들어가면 로컬 환경/커밋 시점에 따라 동작이 달라질 수 있으니, 
작업 시점의 transformers 버전(커밋/태그)을 기록해두는 게 좋다.

### 파이프라인 import + 데모 이미지 다운로드

Unit 3는 여러 파이프라인을 다루므로 한꺼번에 import한다. 

```python
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline,
)

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/..."
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/..._mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))
```

### device 선택

```python
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
```

## Text-to-Image: StableDiffusionPipeline 호출을 “인자 의미”까지 분해

### pipeline 로드

이 글에서는 SD 2.1 base 모델을 예시로 로드한다. 

```python
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
```

### 메모리 부족 대응 옵션

노트북이 제시하는 대표 옵션: FP16 로드, attention slicing, 해상도 축소 

FP16 로드 예:

```python
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
).to(device)
```

attention slicing:

```python
pipe.enable_attention_slicing()
```

### 생성 호출: 핵심 인자 6개를 ‘설계 파라미터’로 보기

```python
generator = torch.Generator(device=device).manual_seed(42)

pipe_output = pipe(
    prompt="Palette knife painting of an autumn cityscape",
    negative_prompt="Oversaturated, blurry, low quality",
    height=480,
    width=640,
    guidance_scale=8,
    num_inference_steps=35,
    generator=generator,
)
img = pipe_output.images[0]
```

1. prompt

- 텍스트 조건(Conditioning). SD에서는 prompt가 text encoder의 임베딩으로 변환되어 UNet에 전달된다. 

2. negative_prompt

- CFG(Classifier-Free Guidance)에서 “unconditional branch”를 구성할 때 사용되는 텍스트다.

즉, 단순 “마이너스 프롬프트”가 아니라 CFG 수식의 한쪽 입력이다. 

3. height, width (+ “8로 나누어 떨어져야 함”)

- VAE가 latent 공간으로 압축/복원을 하기 때문에, 해상도는 특정 단위 제약이 걸린다(“8로 나누어 떨어져야 한다”). 
- 이 제약은 뒤의 DIY loop에서 latents = (1, 4, 64, 64) 같은 형태로 드러납니다. (512/8=64) 

4. guidance_scale

- CFG 강도. 높을수록 prompt에 더 강하게 맞추지만 과하면 과포화/부자연이 생길 수 있다. 
- 이 값은 DIY sampling loop에서 정확히 어떤 연산으로 적용되는지를 직접 확인할 수 있다.

5. num_inference_steps

- denoising step 수. 일반적으로 step이 많을수록 품질이 좋아지나 속도는 느려진다. 
- Unit 2에서 “scheduler가 sampler다”를 배웠다면, Unit 3에서는 latent diffusion + CFG + scheduler scaling이 결합된 형태로 step loop가 돌아간다. 

6. generator(seed 고정)

- 동일한 prompt라도 seed가 달라지면 결과가 달라진다.
- “캐릭터 시스템”에서는 일관성 유지(동일 인물/구도) 레버로 매우 중요하다.

## Text conditioning: Tokenizer -> Text Encoder -> Embeddings를 “손으로” 확인

### prompt -> input_ids -> text_embeddings

노토큰 ID를 찍어보고, text encoder 출력의 shape를 확인한다.

핵심 흐름:

1. tokenizer로 텍스트를 토큰 ID 시퀀스로 변환
2. text encoder(CLIP 계열)로 hidden states 생성
3. 이 hidden states가 UNet의 encoder_hidden_states로 들어감 

shape 예시:

- torch.Size([1, 8, 1024]) 
(= 배치 1, 토큰 길이 8, 임베딩 차원 1024)

### _encode_prompt는 “실전용 래퍼”

pipe._encode_prompt(...)를 사용해 최종 임베딩을 얻을 수 있다. 

```python
text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)
```

중요 포인트(나중에 DIY loop 작성 시 필수):

- CFG를 쓰면 “uncond + text” 두 임베딩이 필요하고, _encode_prompt가 그걸 내부에서 준비해준다. 

## Scheduler: 노이즈 스케줄 확인 + 교체 실험

스케줄의 alphas_cumprod를 플롯으로 확인하고, scheduler를 교체한다. 

## scheduler 교체(핵심은 from_config)

```python
from diffusers import LMSDiscreteScheduler
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
```

설계 관점:

- “모델(UNet)은 그대로 두고 sampler만 바꿔서 결과/속도/스타일을 조절”한다. 

## Latent Diffusion: 왜 latent가 (1,4,64,64)인가 (VAE 관점)

DIY sampling loop에서 latent를 예를 들어 다음과 같이 초기화한다. 

```python
latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
latents *= pipe.scheduler.init_noise_sigma
```

여기서 구조적으로 중요한 사실:

- SD는 “픽셀 공간(512×512×3)”에서 diffusion을 돌리지 않고 **VAE latent 공간(64×64×4)**에서 diffusion을 돌립니다. 
- 그래서 빠르고 메모리 효율적이며, 마지막에만 VAE decode로 픽셀 이미지로 복원합니다. 

## 핵심: DIY Sampling Loop (Pipeline을 분해한 “구현 템플릿”)

여기서는

- CFG를 직접 넣고/빼고
- 중간 단계에서 custom guidance를 삽입하고
- latent를 외부 조건(캐릭터 상태)로 조작한다.

DIY loop 핵심 코드는 아래 순서로 전개된다. 

### prompt encoding + latent init + scheduler 준비

```python
guidance_scale = 8
num_inference_steps = 30
prompt = "Beautiful picture of a wave breaking"
negative_prompt = "zoomed in, blurry, oversaturated, warped"

text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
latents *= pipe.scheduler.init_noise_sigma

pipe.scheduler.set_timesteps(num_inference_steps, device=device)
```

### step loop: “CFG가 실제로는 이 한 줄 연산”

```python
for i, t in enumerate(pipe.scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)  # CFG: uncond/text 두 배치로 확장
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    with torch.no_grad():
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
```


여기서 반드시 이해할 포인트(재구현 체크리스트)

1. 왜 torch.cat([latents]*2)인가?

- CFG는 한 step에서 “unconditional 예측”과 “text 조건 예측”을 둘 다 얻어야 한다.
- 그래서 입력 배치를 2배로 만들어 UNet을 한 번만 호출하고, 결과를 chunk(2)로 나눈다. 

2. scale_model_input

- scheduler마다 모델 입력에 요구되는 스케일이 다를 수 있어 step마다 적용한다.

3. CFG 수식이 코드로 명확히 고정됨

- uncond + s*(text-uncond)
- 이게 곧 guidance_scale의 의미. 

### decode: latent -> image

```python
with torch.no_grad():
    image = pipe.decode_latents(latents.detach())
pil = pipe.numpy_to_pil(image)[0]
```

## Img2Img: “초기 이미지를 latent로 인코딩” + strength로 timestep 구간 선택

Img2Img는 두 방식으로 다룰 수 있다.

- Pipeline 사용: StableDiffusionImg2ImgPipeline 
- DIY Img2Img loop: init_image -> VAE encode -> 일부 timestep만 denoise 

### Pipeline 사용(가장 쉬운 형태)

```python
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)
result = img2img_pipe(prompt="...", image=init_image, strength=0.6).images[0]
```

### DIY Img2Img loop의 핵심: init_image를 latent로 만들기

init_image를 직접 tensor로 바꾸고 [-1,1]로 정규화한 뒤 VAE encode한다.

```python
init_image_tensor = torch.from_numpy(np.array(init_image).transpose(2, 0, 1)).float() / 255.0
init_image_tensor = 2.0 * init_image_tensor - 1.0
init_image_tensor = init_image_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    init_image_latents = pipe.vae.encode(init_image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
```

scaling_factor가 중요한 이유

- SD의 VAE latent는 통상 특정 스케일로 학습되어 있어, encode/decode 시 scaling_factor를 곱/나눠 정합성을 맞춘다.

### strength: “전체 step 중 일부만 사용”한다는 의미

DIY Img2Img loop는 strength로 “얼마나 원본을 유지할지”를 timestep 관점으로 정의한다. 

- init_timestep = int(num_inference_steps * strength)
- timesteps = scheduler.timesteps[t_start:]

즉, strength가 클수록 더 많은 노이즈를 원본 latent에 섞고(더 많이 파괴), 더 강한 변형을 허용한다.

## Inpainting: “마스크 영역만 생성하고, 나머지는 원본 latent 유지”를 코드로 구현

1. DIY Inpainting loop로 “로직”을 직접 구현 
2. 그 다음 inpainting 전용으로 fine-tuned된 pipeline을 사용 

### DIY Inpainting loop의 핵심 아이디어

- step마다 latents를 denoise(생성)하되
- 마스크 밖 영역은 “원본 이미지 latent에 해당 step 노이즈를 섞은 값”으로 덮어써서 유지. 

아래처럼 “배경(background)”을 원본 latent에서 만들고 마스크로 합성한다. 

```python
# latents: 현재 생성 중인 latent
# init_image_latents: 원본 이미지 latent
# mask_image_latent_size: latent 해상도에 맞춘 mask

if i < len(pipe.scheduler.timesteps) - 1:
    noise = torch.randn(init_image_latents.shape, generator=generator, device=device, dtype=torch.float32)
    background = pipe.scheduler.add_noise(
        init_image_latents, noise, torch.tensor([pipe.scheduler.timesteps[i + 1]])
    )

    latents = latents * mask_image_latent_size
    background = background * (1 - mask_image_latent_size)
    latents += background
```

이 부분이 중요한 이유(실전 확장 포인트):

- “inpainting은 UNet이 특별해서가 아니라, sampling loop에서 latent를 마스크로 매 step 합성하는 방식”이라는 점이 드러난다.
- 즉, 원하는 커스텀 편집(예: 캐릭터 얼굴만 유지, 의상만 변경 등)은 결국 mask + 합성 로직을 어떻게 설계하느냐로 정리된다.

### inpainting decode

DIY inpainting loop 마지막에는 VAE scaling_factor를 나눠 decode하고, [-1,1]을 [0,1]로 되돌리는 후처리를 한다. 

```python
latents_norm = latents / pipe.vae.config.scaling_factor
with torch.no_grad():
    inpainted = pipe.vae.decode(latents_norm).sample

inpainted = (inpainted / 2 + 0.5).clamp(0, 1)
```

### Inpainting 전용 pipeline 사용

“stabilityai/stable-diffusion-2-inpainting”을 사용한다. 

```python
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(device)
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```


## Depth2Img: “깊이맵을 추가 conditioning으로 넣는” 파이프라인

- Img2Img는 강도(strength) 조절이 어렵고 색이 너무 유지될 수 있다
- Depth2Img는 depth estimation으로 깊이맵을 만들고, 이를 UNet에 추가 조건으로 넣어 구조(깊이/형태)를 보존하며 재질/색을 바꾸는 방향을 노립니다. 

```python
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth").to(device)
image = pipe(prompt="An oil painting of a man on a bench", image=init_image).images[0]
```


참고자료
Huggingface, Diffusion Course, https://huggingface.co/learn