# 텍스트 생성 모델 단계별 학습 가이드

## 📋 개요

이 프로젝트는 **GPT2 기반 한국어 텍스트 생성 모델**을 단계별로 학습할 수 있도록 설계된 교육용 자료입니다. 수강생들은 5단계에 걸쳐 텍스트 생성의 핵심 개념과 실무 구현 방법을 체계적으로 학습할 수 있습니다.

## 🎯 학습 목표

- **GPT2 언어 모델**의 구조와 작동 원리 이해
- **다양한 생성 전략**과 파라미터 효과 체험
- **모델 Fine-tuning** 과정과 Transfer Learning 개념 학습
- **실시간 텍스트 생성 웹 서비스** 구축 능력 획득
- **창의적 AI 애플리케이션** 개발 기초 습득

## 📚 단계별 구성

### 🔷 Step 1: 텍스트 생성 기본 개념
**파일**: `step1_basic_generation.py`

**학습 내용**:
- GPT2 모델 구조와 AutoRegressive Generation 이해
- 기본 텍스트 생성 파이프라인 구현
- Greedy Search의 특징과 한계 체험

**핵심 TODO**:
```python
# 한국어 GPT2 모델 로드
pretrained = # TODO: "skt/kogpt2-base-v2" 지정

# 모델과 토크나이저 초기화
model = # TODO: GPT2LMHeadModel.from_pretrained() 사용
tokenizer = # TODO: PreTrainedTokenizerFast 로드

# 기본 텍스트 생성
generated_ids = # TODO: model.generate() 호출
```

**학습 포인트**:
- `GPT2LMHeadModel`: Causal Language Model의 대표적 구현
- `do_sample=False`: 확정적 생성 (Greedy Search)
- `eos_token="</s>"`: 한국어 GPT2의 문장 종료 토큰
- AutoRegressive: 이전 토큰들을 조건으로 다음 토큰 예측

---

### 🔷 Step 2: 다양한 생성 전략
**파일**: `step2_generation_strategies.py`

**학습 내용**:
- Greedy Search vs Beam Search vs Sampling 비교
- `do_sample` 파라미터의 역할과 중요성
- 각 전략의 장단점과 적용 시나리오

**핵심 TODO**:
```python
# Beam Search 실험
generated_ids = # TODO: num_beams=3 추가

# Top-k Sampling 실험  
generated_ids = # TODO: do_sample=True, top_k=50

# Top-p (Nucleus) Sampling 실험
generated_ids = # TODO: do_sample=True, top_p=0.92

# 동일 설정 반복 실행으로 다양성 확인
for i in range(3):
    generated_ids = # TODO: 샘플링 기법으로 3회 생성
```

**학습 포인트**:
- **Beam Search**: 전체적으로 더 좋은 확률의 문장 탐색
- **Top-k Sampling**: 상위 k개 토큰 중 확률적 선택
- **Top-p Sampling**: 누적 확률 p까지의 토큰들 중 선택
- **다양성 vs 품질**: 생성 전략 간 트레이드오프

---

### 🔷 Step 3: 생성 파라미터 조정
**파일**: `step3_parameter_tuning.py`

**학습 내용**:
- Temperature, Top-k, Top-p 파라미터 심화 실험
- 반복 방지 기법 (repetition penalty, no_repeat_ngram_size)
- 실무 최적화 파라미터 조합 발견

**핵심 TODO**:
```python
# 반복 방지 실험
generated_ids = # TODO: no_repeat_ngram_size=3 적용

# Temperature 효과 실험
generated_ids = # TODO: temperature 값 변화시켜 비교

# 최적 파라미터 조합
generated_ids = # TODO: 통합 최적화 설정 적용
```

**학습 포인트**:
- **Temperature**: 확률 분포의 날카로움 조절 (창의성 vs 안정성)
- **Repetition Penalty**: 이미 사용된 토큰의 재출현 억제
- **N-gram 반복 방지**: 자연스러운 문장 흐름 보장
- **파라미터 조합**: 도메인별 최적 설정 탐색

---

### 🔷 Step 4: 모델 Fine-tuning
**파일**: `step4_model_training.py`

**학습 내용**:
- NSMC 데이터를 활용한 GPT2 Fine-tuning
- Transfer Learning 개념과 실무 적용
- ratsnlp 라이브러리를 활용한 효율적 학습 파이프라인

**핵심 TODO**:
```python
# 학습 설정 구성
args = GenerationTrainArguments(
    pretrained_model_name=# TODO: 모델명,
    downstream_corpus_name=# TODO: 데이터셋명,
    batch_size=# TODO: 배치 크기,
)

# 데이터셋과 데이터로더 구성
train_dataset = # TODO: GenerationDataset 생성
train_dataloader = # TODO: DataLoader 설정

# 모델 학습 실행
trainer.fit(# TODO: task, dataloaders 전달)
```

**학습 포인트**:
- **NSMC 데이터**: 네이버 영화 리뷰로 감정 표현 풍부한 텍스트 학습
- **Fine-tuning**: 일반 언어 모델을 특정 도메인에 특화
- **GenerationTask**: Lightning 기반 자동화된 학습 파이프라인
- **Transfer Learning**: 사전학습된 지식을 새로운 태스크에 활용

---

### 🔷 Step 5: 웹 서비스 구현
**파일**: `step5_web_service.py`

**학습 내용**:
- Flask 기반 실시간 텍스트 생성 API 구현
- 학습된 모델을 실제 서비스로 배포
- 사용자 친화적 웹 인터페이스 제공

**핵심 TODO**:
```python
# 배포 환경 설정
device = # TODO: GPU/CPU 디바이스 설정
args = # TODO: GenerationDeployArguments 구성

# Fine-tuned 모델 로드
model = # TODO: 체크포인트에서 모델 복원
model.load_state_dict(# TODO: 가중치 로드)

# 추론 함수 구현
def inference_fn(prompt, **kwargs):
    generated_ids = # TODO: 파라미터를 반영한 생성
    return # TODO: 결과 반환

# 웹 서비스 실행
app = # TODO: Flask 앱 생성
app.run(# TODO: 서버 실행)
```

**학습 포인트**:
- **Model Deployment**: 학습된 모델의 실제 서비스 배포
- **Flask API**: RESTful 웹 서비스 구현
- **실시간 추론**: 사용자 요청에 대한 즉시 응답 시스템
- **파라미터 조정 UI**: 사용자가 생성 품질을 직접 조절

## 🚀 실행 방법

### 1단계부터 순차적 학습:
```bash
# Step 1: 기본 텍스트 생성 이해
python step1_basic_generation.py

# Step 2: 생성 전략 비교 (TODO 완성 후)
python step2_generation_strategies.py

# Step 3: 파라미터 튜닝 (TODO 완성 후)
python step3_parameter_tuning.py

# Step 4: 모델 학습 (TODO 완성 후)
python step4_model_training.py

# Step 5: 웹 서비스 (TODO 완성 후)
python step5_web_service.py
# 브라우저에서 http://localhost:9001 접속
```

## 🎓 교육적 장점

### 1. **체계적 난이도 증가**
- Step 1: 기본 개념 → Step 5: 실무 시스템
- 각 단계별 명확한 학습 목표와 실습 과제

### 2. **실무 중심 커리큘럼**
- Production 레벨의 모델 학습과 배포 경험
- 실제 서비스에서 사용 가능한 코드와 아키텍처
- 다양한 비즈니스 시나리오 대응 능력 배양

### 3. **즉시 확인 가능한 결과**
- 각 단계에서 바로 실행하고 결과 확인
- 파라미터 변화에 따른 즉시 피드백
- 시각적으로 이해하기 쉬운 텍스트 생성 결과

### 4. **확장 가능한 학습 구조**
- 다른 언어 모델로 확장 가능 (KoGPT-Trinity, Polyglot-ko 등)
- 다양한 도메인 데이터로 추가 실험 가능
- 최신 생성 모델 연구 트렌드 반영 용이

## 🔧 GitHub 배포 전략

### Repository 구조:
```
task3-gen/
├── step1_basic_generation.py      # 1단계: 기본 생성 개념
├── step2_generation_strategies.py # 2단계: 생성 전략 비교
├── step3_parameter_tuning.py      # 3단계: 파라미터 조정
├── step4_model_training.py        # 4단계: 모델 Fine-tuning
├── step5_web_service.py           # 5단계: 웹 서비스 구현
├── solutions/                     # 단계별 해답
│   ├── step1_solution.py
│   ├── step2_solution.py
│   ├── step3_solution.py
│   ├── step4_solution.py
│   └── step5_solution.py
├── templates/                     # 웹 UI 템플릿
└── README_stepwise_learning.md    # 이 파일
```

### 수업 진행 방식:
1. **이론 + 실습**: 각 단계별 개념 설명 후 바로 실습
2. **점진적 공개**: 이전 단계 완료 후 다음 단계 접근 허용
3. **피어 리뷰**: 학생들 간 코드 리뷰와 결과 공유
4. **창의적 프로젝트**: 최종 단계에서 개인별 애플리케이션 개발

## 📖 핵심 개념 비교

| 개념 | 분류/NER | 텍스트 생성 |
|------|----------|-------------|
| **모델 타입** | Encoder (BERT) | Decoder (GPT) |
| **학습 목표** | 라벨 예측 | 다음 토큰 예측 |
| **출력** | 고정된 클래스 | 가변 길이 시퀀스 |
| **평가 기준** | Accuracy, F1 | Perplexity, 인간 평가 |
| **주요 도전** | 라벨 불균형 | 반복, 일관성 |
| **응용 분야** | 문서 분류, 정보 추출 | 창작, 대화, 요약 |

## 🌟 확장 학습 주제

### 고급 생성 기법:
- **Contrastive Search**: 최신 디코딩 알고리즘
- **FUDGE**: 제어 가능한 텍스트 생성
- **Prompt Engineering**: 효과적인 프롬프트 설계

### 평가 및 최적화:
- **BLEU, ROUGE**: 자동 평가 메트릭
- **Human Evaluation**: 인간 평가 실험 설계
- **A/B Testing**: 서비스 품질 최적화

### 실무 고려사항:
- **Content Filtering**: 부적절한 내용 필터링
- **Bias Detection**: 편향성 탐지 및 완화
- **Scalability**: 대규모 서비스 아키텍처

이 단계별 학습을 통해 수강생들은 텍스트 생성 모델의 이론적 이해부터 실무 적용까지 완전한 역량을 갖추게 됩니다! 🚀✨
