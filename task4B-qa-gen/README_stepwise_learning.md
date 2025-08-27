# 생성형 질의응답(Generative QA) 단계별 학습 가이드

## 📋 개요

이 프로젝트는 **Generative Question Answering (Seq2Seq QA)** 시스템을 단계별로 학습할 수 있도록 설계된 교육용 자료입니다. T5 모델을 기반으로 한 텍스트 생성 방식의 QA를 통해 Extractive QA와는 완전히 다른 접근법을 체험할 수 있습니다.

## 🎯 학습 목표

- **Generative QA**의 원리와 T5 모델의 텍스트 생성 방식 이해
- **단순하면서도 강력한** Seq2Seq QA 전후처리 과정 체험  
- **창의적 답변 생성**과 다양한 생성 파라미터 활용법 학습
- **T5 fine-tuning**과 Seq2SeqTrainer의 특성 파악
- **생성형 QA 웹 서비스** 구축과 창의적 API 설계 능력 습득

## 🌟 **Extractive QA vs Generative QA 핵심 차이점**

| 특징 | **Extractive QA** | **Generative QA** |
|------|-------------------|-------------------|
| **답변 방식** | 컨텍스트에서 기존 텍스트 **추출** | 새로운 텍스트를 **생성** |
| **모델 구조** | BERT (Encoder만) | T5 (Encoder-Decoder) |
| **전처리 복잡도** | ⭐⭐⭐⭐⭐ (매우 복잡) | ⭐⭐ (매우 간단) |
| **후처리 복잡도** | ⭐⭐⭐⭐⭐ (매우 복잡) | ⭐ (매우 간단) |
| **창의적 답변** | ❌ 불가능 | ✅ **가능 (핵심 장점!)** |
| **요약/설명** | ❌ 불가능 | ✅ **가능** |
| **처리 속도** | ✅ 빠름 (50-100ms) | ⚠️ 느림 (500-2000ms) |
| **정확성 보장** | ✅ 높음 | ⚠️ 검증 필요 |

## 📚 단계별 구성

### 🔷 Step 1: Generative QA 기본 개념
**파일**: `step1_basic_generative_qa.py`

**학습 내용**:
- Generative QA의 기본 원리와 Extractive QA와의 근본적 차이
- T5 모델의 Text-to-Text 변환 방식 이해
- Beam Search와 생성 파라미터의 효과 체험

**핵심 TODO**:
```python
# T5 모델 로드 (BERT가 아닌!)
pretrained = # TODO: "paust/pko-t5-base-finetuned-korquad"
model = # TODO: AutoModelForSeq2SeqLM.from_pretrained()

# T5 입력 형식 구성 (중요!)
input_text = # TODO: f"question: {question} context: {context}"

# 텍스트 생성 (추출이 아닌!)
output_ids = model.generate(
    # TODO: input_ids, max_length, num_beams
)
answer = # TODO: tokenizer.decode(output_ids[0])

# 창의적 질문 실험 (Generative QA의 장점!)
creative_questions = [
    # TODO: "설명해주세요", "요약해주세요" 등
]
```

**학습 포인트**:
- **Text-to-Text 패러다임**: 모든 태스크를 텍스트 생성으로 통일
- **T5 입력 형식**: "question: Q context: C" → 답변 생성
- **생성 vs 추출**: 완전히 다른 접근 방식
- **창의적 가능성**: 요약, 설명, 추론 등 복합 답변

---

### 🔷 Step 2: Seq2Seq QA 데이터 전처리 
**파일**: `step2_seq2seq_preprocessing.py`

**학습 내용**:
- **극도로 단순한** Seq2Seq QA 전처리 과정 체험
- T5 입력 형식의 중요성과 프롬프트 설계
- Extractive QA와의 복잡도 차이 극명 체감

**핵심 TODO**:
```python
# 간단한 입력 형식 변환 (복잡한 위치 매핑 불필요!)
def generate_input(question: str, context: str) -> str:
    return # TODO: f"question: {question.lstrip()} context: {context.lstrip()}"

# 단순한 토크나이제이션 (offset mapping 불필요!)
model_inputs = tokenizer(
    # TODO: inputs, max_length, padding, truncation
)

# 타겟 텍스트 토크나이제이션 (새로운 개념!)
labels = tokenizer(
    # TODO: text_target=targets, max_length, padding, truncation
)

# 패딩 토큰 처리 (간단!)
labels["input_ids"] = [
    # TODO: [(l if l != tokenizer.pad_token_id else -100) for l in label]
]
```

**학습 포인트**:
- **단순함의 미학**: Extractive QA 대비 1/10 수준의 복잡도
- **text_target 파라미터**: T5 타겟 텍스트 처리 방식
- **프롬프트 표준화**: 일관된 입력 형식의 중요성
- **확장성**: 새로운 태스크 쉽게 추가 가능

---

### 🔷 Step 3: 생성 및 후처리
**파일**: `step3_generation_postprocessing.py`

**학습 내용**:
- **극도로 간단한** 후처리 과정 (Extractive QA와 극명한 대조!)
- Beam Search vs Greedy Search 실전 비교
- 토큰 확률 기반 신뢰도 점수 계산

**핵심 TODO**:
```python
# 점수와 함께 생성 (상세 정보 포함)
outputs = model.generate(
    # TODO: return_dict_in_generate=True, output_scores=True
)

# 직접 텍스트 디코딩 (복잡한 위치 변환 불필요!)
answer = # TODO: tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

# 간단한 점수 계산
token_prob = # TODO: F.softmax(outputs.scores[i-1], dim=-1)[0, token_id].item()
overall_score = # TODO: torch.prod(torch.tensor(token_probs)).item()

# 매우 간단한 후처리 (3단계로 끝!)
# 1. 텍스트 디코딩
# 2. 텍스트 정제  
# 3. 평가 메트릭 계산
```

**학습 포인트**:
- **후처리 혁신**: 10단계 → 3단계로 대폭 단순화
- **직접 디코딩**: 복잡한 위치 기반 변환 불필요
- **생성 품질 제어**: Beam Search와 파라미터 조정
- **실시간 적용**: 간단함으로 인한 높은 실용성

---

### 🔷 Step 4: Seq2Seq 모델 학습
**파일**: `step4_seq2seq_training.py`

**학습 내용**:
- T5 기반 Seq2Seq QA 모델의 fine-tuning 과정
- Seq2SeqTrainer와 생성 기반 평가 시스템
- Extractive QA와의 학습 방식 차이점 이해

**핵심 TODO**:
```python
# Seq2Seq 전용 인수 (TrainingArguments 아님!)
parser = # TODO: HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

# T5 모델 로드 (BERT 아님!)
model = # TODO: AutoModelForSeq2SeqLM.from_pretrained()

# Seq2Seq 전용 데이터 콜레이터
data_collator = # TODO: DataCollatorForSeq2Seq(tokenizer, model=model)

# 생성 기반 평가 (위치 기반 아님!)
# 실제 텍스트 생성 후 SQuAD 메트릭 계산
trainer = QuestionAnsweringSeq2SeqTrainer(
    # TODO: 모든 구성 요소 연결
)

# 생성 파라미터와 함께 평가
metrics = # TODO: trainer.evaluate(max_length=max_length, num_beams=num_beams)
```

**학습 포인트**:
- **Seq2SeqTrainer**: 생성 기반 학습과 평가
- **Language Modeling Loss**: 텍스트 생성 손실 함수
- **DataCollatorForSeq2Seq**: 인코더-디코더 데이터 처리
- **실제 성능 평가**: 생성 후 메트릭 계산

---

### 🔷 Step 5: 생성형 웹 서비스 
**파일**: `step5_generative_web_service.py`

**학습 내용**:
- **창의적 생성 기능**을 포함한 고급 웹 서비스 구현
- 표준 vs 창의적 생성 모드 지원
- Generative QA만의 독특한 API 설계

**핵심 TODO**:
```python
# 생성형 QA 모델 클래스
class GenerativeQAModel:
    def infer_one(self, question, context):
        # TODO: 표준 생성 (안정적, 일관된 답변)
        
    def infer_creative(self, question, context, creative_params):
        # TODO: 창의적 생성 (다양하고 창의적인 답변)
        # do_sample=True, temperature, top_p 활용

# 다양한 API 엔드포인트
class WebAPI(FlaskView):
    @route('/api')  # 표준 생성
    @route('/api/creative')  # 창의적 생성 (Generative QA 고유!)
    @route('/api/compare')  # 두 방식 비교 (교육용)

# 창의적 생성 파라미터
creative_params = {
    # TODO: do_sample, temperature, top_p, num_beams
}
```

**학습 포인트**:
- **창의적 생성**: Generative QA만의 독특한 장점
- **다중 모드 서비스**: 표준/창의적/비교 모드
- **실시간 파라미터 조정**: 사용자 맞춤형 생성
- **교육적 도구**: 두 방식의 직접 비교

## 🚀 실행 방법

### 1단계부터 순차적 학습:
```bash
# Step 1: Generative QA 기본 개념 (vs Extractive QA)
python step1_basic_generative_qa.py

# Step 2: 간단한 전처리 체험 (TODO 완성 후)
python step2_seq2seq_preprocessing.py

# Step 3: 간단한 후처리 구현 (TODO 완성 후)
python step3_generation_postprocessing.py

# Step 4: T5 모델 학습 (TODO 완성 후)
python step4_seq2seq_training.py \
  --model_name_or_path paust/pko-t5-base \
  --train_file data/KorQuAD_v1.0_train.json \
  --validation_file data/KorQuAD_v1.0_dev.json \
  --output_dir output/korquad-seq2seq \
  --do_train --do_eval \
  --predict_with_generate

# Step 5: 창의적 웹 서비스 배포 (TODO 완성 후)
python step5_generative_web_service.py serve \
  --pretrained output/korquad-seq2seq/checkpoint-* \
  --server_port 9165
# 브라우저에서 http://localhost:9165 접속
```

## 🎓 교육적 장점

### 1. **Extractive vs Generative의 극명한 대조**
- **복잡도 역전**: Extractive(복잡) ↔ Generative(단순)
- **능력 차이**: Extractive(제한적) ↔ Generative(창의적)
- **패러다임 차이**: 추출 ↔ 생성

### 2. **단순함을 통한 본질 이해**
- 복잡한 전후처리에 가려진 핵심 개념 명확화
- T5의 Text-to-Text 패러다임 직관적 체험
- 생성 AI의 근본 원리 학습

### 3. **창의적 AI 체험**
- 요약, 설명, 추론 등 고차원적 답변 생성
- 사용자 맞춤형 생성 파라미터 조정
- 창작과 교육에서의 AI 활용 가능성 탐색

### 4. **실무 즉시 적용**
- 간단한 구현으로 빠른 프로토타이핑
- 다양한 도메인으로 쉬운 확장
- 실제 서비스에 바로 적용 가능

## 🔧 **태스크별 복잡도 비교**

| 단계 | **분류** | **NER** | **추출형 QA** | **생성형 QA** | **텍스트 생성** |
|------|----------|---------|---------------|---------------|-----------------|
| **전처리** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **⭐⭐** | ⭐⭐ |
| **후처리** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | **⭐** | ⭐ |
| **창의성** | ❌ | ❌ | ❌ | **✅** | ✅ |
| **구현 난이도** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **⭐⭐⭐** | ⭐⭐⭐ |
| **교육 가치** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** | ⭐⭐⭐⭐ |

### **Generative QA가 교육적으로 특별한 이유:**

1. **복잡도의 역설**: 가장 강력하면서도 구현은 단순
2. **창의성 체험**: AI의 창작 가능성을 직접 경험
3. **패러다임 전환**: 기존 NLP 관점을 완전히 바꾸는 경험
4. **실무 직결**: 현재 가장 주목받는 생성 AI 기술

## 📈 **수업 진행 전략**

### **강조 포인트: "단순함 속의 강력함"**

#### **1단계: 패러다임 충격 (60분)**
- "추출 vs 생성"의 근본적 차이 강조
- 창의적 질문으로 Generative QA 장점 체험
- Extractive QA로는 불가능한 답변들 시연

#### **2단계: 단순함의 충격 (90분)**  
- "이렇게 간단해도 되나?" 체험
- Extractive QA 전처리와 직접 비교
- 10분의 1 수준의 복잡도 실감

#### **3단계: 후처리 혁신 (60분)**
- "후처리가 3단계로 끝?" 감탄
- N-best 생성 vs 직접 디코딩 비교
- 단순함의 미학과 실용성

#### **4단계: 실제 학습 체험 (120분)**
- T5 모델의 실제 fine-tuning
- Seq2SeqTrainer의 강력함 체험
- 생성 기반 평가의 직관성

#### **5단계: 창의적 서비스 구축 (90분)**
- 표준/창의적 생성 모드 비교
- 실시간 파라미터 조정 체험
- AI 창작 도구의 가능성 탐색

## 💡 **실무 연결 포인트**

### **Generative QA의 독특한 활용 분야:**

1. **교육 혁신**: 
   - "설명해주세요" → 맞춤형 설명 생성
   - "예시를 들어주세요" → 창의적 예시 제공
   - "차이점을 알려주세요" → 비교 분석 생성

2. **고객 지원 고도화**:
   - 복합 문제에 대한 단계별 해결 방안
   - 상황별 맞춤형 안내 생성
   - 창의적 대안 제시

3. **연구 도구**:
   - 논문 자동 요약 및 해석
   - 핵심 개념 설명 생성
   - 연구 아이디어 제안

4. **창작 지원**:
   - 스토리텔링 지원
   - 아이디어 브레인스토밍
   - 다양한 관점 제시

5. **언어 학습**:
   - 문법 설명 생성
   - 문화적 맥락 설명
   - 학습자 수준별 맞춤 설명

## 🌟 **차별화된 학습 경험**

### **기존 QA 학습의 한계**:
- 복잡한 전후처리에 매몰되어 본질 놓침
- 추출만 가능하여 AI의 창의적 가능성 체험 부족
- 실무 적용 시 높은 구현 난이도

### **Generative QA 학습의 혁신**:
- **단순함을 통한 본질 집중**: 복잡한 기술에 가려진 핵심 이해
- **창의성 체험**: AI가 진짜 할 수 있는 일의 경험
- **즉시 적용**: 학습 즉시 실무에 활용 가능

## 🎯 **기대 학습 효과**

1. **패러다임 이해**: 생성 AI 시대의 새로운 사고방식
2. **기술 깊이**: T5와 Transformer 구조의 본질적 이해  
3. **창의적 활용**: AI를 도구가 아닌 창작 파트너로 인식
4. **실무 역량**: 바로 적용 가능한 생성형 QA 시스템 구축 능력
5. **미래 대비**: 생성 AI 시대를 선도할 수 있는 전문성

## 🔮 **미래 확장 방향**

### **고급 생성 기법**:
- **RAG (Retrieval-Augmented Generation)**: 검색 + 생성 결합
- **Few-shot Learning**: 적은 예제로 새로운 태스크 적응  
- **Chain-of-Thought**: 단계별 추론 과정 생성

### **다양한 생성 모델**:
- **GPT 계열**: 더 강력한 생성 능력
- **PaLM, LLaMA**: 최신 대규모 언어 모델
- **다국어 모델**: 글로벌 서비스 확장

### **실무 고도화**:
- **사실성 검증**: Hallucination 탐지 및 방지
- **개인화**: 사용자별 맞춤형 생성 스타일
- **실시간 학습**: 사용자 피드백 기반 지속 개선

이 단계별 학습을 통해 수강생들은 **"단순하지만 강력한"** Generative QA의 매력을 완전히 체험하고, 생성 AI 시대를 선도할 수 있는 전문성을 갖추게 됩니다! 🚀✨🎓
