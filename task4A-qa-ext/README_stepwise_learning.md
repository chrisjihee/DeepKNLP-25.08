# 질의응답(QA) 모델 단계별 학습 가이드

## 📋 개요

이 프로젝트는 **Extractive Question Answering** 시스템을 단계별로 학습할 수 있도록 설계된 교육용 자료입니다. 수강생들은 5단계에 걸쳐 QA의 핵심 개념부터 실무 배포까지 체계적으로 학습할 수 있습니다.

## 🎯 학습 목표

- **Extractive QA**의 원리와 BERT 기반 모델 구조 이해
- **복잡한 데이터 전처리** 과정과 토큰-문자 매핑 체험
- **정교한 후처리** 알고리즘과 N-best 답변 생성 학습
- **KorQuAD 데이터**로 한국어 QA 모델 fine-tuning 경험
- **실시간 QA 웹 서비스** 구축과 API 설계 능력 습득

## 📚 단계별 구성

### 🔷 Step 1: QA 기본 개념
**파일**: `step1_basic_qa.py`

**학습 내용**:
- Extractive QA의 기본 원리와 BERT 구조 이해
- Start/End Logits의 개념과 활용 방법
- 단순한 argmax 기반 답변 추출 체험

**핵심 TODO**:
```python
# 한국어 QA 모델 로드
pretrained = # TODO: "monologg/koelectra-base-v3-finetuned-korquad"

# 모델과 토크나이저 초기화
tokenizer = # TODO: AutoTokenizer.from_pretrained()
model = # TODO: AutoModelForQuestionAnswering.from_pretrained()

# 기본 QA 추론
outputs = # TODO: model(**inputs)
start_index = # TODO: torch.argmax(start_logits)
answer = # TODO: tokenizer.decode(answer_tokens)
```

**학습 포인트**:
- `AutoModelForQuestionAnswering`: BERT + QA Head 구조
- Start/End Logits: 각 토큰 위치의 답변 시작/끝 확률
- 토큰 인덱스를 실제 답변 텍스트로 변환하는 과정
- Reading Comprehension의 핵심 개념

---

### 🔷 Step 2: 데이터 전처리
**파일**: `step2_data_preprocessing.py`

**학습 내용**:
- QA 데이터의 복잡한 전처리 과정 완전 이해
- Doc Stride를 활용한 긴 컨텍스트 처리 방법
- Offset Mapping을 통한 정밀한 문자-토큰 위치 매핑

**핵심 TODO**:
```python
# 복잡한 토크나이제이션
tokenized_examples = tokenizer(
    questions, contexts,
    # TODO: truncation, stride, return_overflowing_tokens, return_offsets_mapping
)

# 문자 위치를 토큰 위치로 변환
start_char = # TODO: answers["answer_start"][0]
end_char = # TODO: start_char + len(answers["text"][0])

# 정확한 토큰 위치 계산
while offsets[token_start_index][0] <= start_char:
    token_start_index += 1
tokenized_examples["start_positions"].append(token_start_index - 1)
```

**학습 포인트**:
- **Doc Stride**: 긴 문서를 겹치는 청크로 분할 처리
- **Offset Mapping**: 토큰별 원본 문자 위치 정보 (start_char, end_char)
- **위치 라벨링**: 문자 기반 정답을 토큰 기반 라벨로 정확히 변환
- **다중 청크**: 하나의 예제가 여러 feature로 분할되는 복잡성

---

### 🔷 Step 3: 후처리 (가장 복잡!)
**파일**: `step3_postprocessing.py`

**학습 내용**:
- QA에서 가장 복잡하고 중요한 후처리 과정 마스터
- N-best 답변 후보 생성과 점수 기반 선택 방법
- SQuAD v2.0의 "답변 없음" 처리 방식 이해

**핵심 TODO**:
```python
# N-best 후보 생성
start_indexes = # TODO: np.argsort()로 상위 n_best_size개 선택
end_indexes = # TODO: np.argsort()로 상위 n_best_size개 선택

# 모든 시작-끝 조합 검증
for start_index in start_indexes:
    for end_index in end_indexes:
        # TODO: 유효성 검사 및 점수 계산
        if valid_conditions:
            score = # TODO: start_logit + end_logit
            valid_answers.append({"score": score, "text": answer_text})

# 최고 점수 답변 선택
valid_answers = # TODO: sorted(by score, reverse=True)
best_answer = # TODO: valid_answers[0]["text"]
```

**학습 포인트**:
- **N-best 생성**: 단순 argmax의 한계 극복, 모든 조합 고려
- **유효성 검사**: start ≤ end, 길이 제한, 컨텍스트 범위 등
- **점수 계산**: Log 확률 덧셈 = 확률 곱셈의 이해
- **SQuAD v2.0**: CLS 토큰 점수와 일반 답변 점수 비교

---

### 🔷 Step 4: 모델 학습
**파일**: `step4_model_training.py`

**학습 내용**:
- KorQuAD 데이터셋을 활용한 BERT 기반 QA 모델 fine-tuning
- Hugging Face Transformers 생태계의 효율적 활용
- QA 특화 학습 설정과 평가 메트릭 이해

**핵심 TODO**:
```python
# 학습 설정 구성
model_args = ModelArguments(
    model_name_or_path=# TODO: "monologg/koelectra-base-v3",
)
data_args = DataTrainingArguments(
    train_file=# TODO: "data/KorQuAD_v1.0_train.json",
    max_seq_length=# TODO: 384,
    doc_stride=# TODO: 128,
)

# 모델과 토크나이저 로드
config = # TODO: AutoConfig.from_pretrained()
model = # TODO: AutoModelForQuestionAnswering.from_pretrained()

# 커스텀 Trainer로 학습
trainer = QuestionAnsweringTrainer(
    # TODO: 모든 구성 요소 연결
)
train_result = # TODO: trainer.train()
```

**학습 포인트**:
- **KorQuAD**: 한국어 Reading Comprehension 데이터셋 특성
- **QA Head**: 기존 BERT에 start/end 분류 헤드 추가
- **평가 메트릭**: Exact Match, F1 Score의 의미와 계산
- **하이퍼파라미터**: QA 특화 설정 (learning_rate, max_seq_length 등)

---

### 🔷 Step 5: 웹 서비스
**파일**: `step5_web_service.py`

**학습 내용**:
- 학습된 QA 모델을 실제 웹 서비스로 배포
- Flask-Classful을 활용한 REST API 설계
- 실시간 질의응답 시스템의 완전한 구현

**핵심 TODO**:
```python
# QA 모델 서비스 클래스
class QAModel(LightningModule):
    def __init__(self, pretrained, server_page, normalized):
        self.tokenizer = # TODO: AutoTokenizer.from_pretrained()
        self.model = # TODO: AutoModelForQuestionAnswering.from_pretrained()
        
    def infer_one(self, question, context):
        inputs = # TODO: tokenizer.encode_plus()
        outputs = # TODO: model(**inputs)
        answer = # TODO: 토큰 디코딩
        return {"answer": answer, "score": score}

# Flask API 엔드포인트
class WebAPI(FlaskView):
    @route('/api', methods=['POST'])
    def api(self):
        data = # TODO: request.json
        result = # TODO: self.model.infer_one()
        return # TODO: jsonify(result)
```

**학습 포인트**:
- **프로덕션 아키텍처**: Model Loading, Inference Engine, Web Framework
- **REST API**: 단일/배치 처리, 상태 확인 엔드포인트
- **신뢰도 점수**: Softmax 정규화 vs Raw Logit 점수
- **실무 고려사항**: 확장성, 보안, 모니터링

## 🚀 실행 방법

### 1단계부터 순차적 학습:
```bash
# Step 1: 기본 QA 개념 이해
python step1_basic_qa.py

# Step 2: 복잡한 전처리 체험 (TODO 완성 후)
python step2_data_preprocessing.py

# Step 3: 정교한 후처리 구현 (TODO 완성 후)
python step3_postprocessing.py

# Step 4: 모델 학습 (TODO 완성 후)
python step4_model_training.py \
  --model_name_or_path monologg/koelectra-base-v3 \
  --train_file data/KorQuAD_v1.0_train.json \
  --validation_file data/KorQuAD_v1.0_dev.json \
  --output_dir output/korquad \
  --do_train --do_eval

# Step 5: 웹 서비스 배포 (TODO 완성 후)
python step5_web_service.py serve \
  --pretrained output/korquad/checkpoint-* \
  --server_port 9164
# 브라우저에서 http://localhost:9164 접속
```

## 🎓 교육적 장점

### 1. **점진적 복잡성 증가**
- Step 1: 기본 개념 → Step 5: 프로덕션 서비스
- 각 단계별 명확한 학습 목표와 실무 연결점

### 2. **QA 특화 심화 학습**
- 다른 NLP 태스크와 차별화된 QA만의 복잡성 체험
- 전처리-후처리의 극단적 복잡성을 통한 엔지니어링 역량 향상
- 실제 프로덕션에서 마주하는 문제들의 사전 경험

### 3. **실무 중심 커리큘럼**
- 학술적 개념을 넘어선 실제 서비스 배포 경험
- 대용량 데이터 처리와 실시간 추론 최적화
- API 설계와 웹 서비스 아키텍처 구축

### 4. **한국어 NLP 특화**
- KorQuAD 데이터셋을 통한 한국어 특성 이해
- 한국어 토크나이제이션의 복잡성 체험
- 한국어 QA 시스템의 실무 구현 능력

## 🔧 복잡도 비교

| 단계 | 텍스트 생성 | 분류/NER | **QA (Extractive)** |
|------|-------------|-----------|---------------------|
| **전처리** | ⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐⭐⭐** |
| **후처리** | ⭐ | ⭐⭐ | **⭐⭐⭐⭐⭐** |
| **평가** | ⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐⭐** |
| **전체** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |

### **QA가 가장 복잡한 이유:**

1. **전처리 복잡성**:
   - Doc Stride: 긴 문서를 겹치는 청크로 분할
   - Offset Mapping: 정밀한 문자-토큰 위치 매핑
   - 다중 청크: 하나의 예제가 여러 feature로 분할

2. **후처리 복잡성**:
   - N-best 생성: 모든 start×end 조합 고려
   - 복잡한 유효성 검사: 논리적, 길이, 범위 제약
   - SQuAD v2.0: "답변 없음" 판단 로직

3. **평가 복잡성**:
   - Exact Match: 완전 일치 평가
   - F1 Score: 토큰 단위 겹침 계산
   - 정규화: 언어별 특수 처리

## 📖 핵심 개념 정리

### **QA vs 다른 태스크 비교**

| 특징 | 분류 | NER | **QA** |
|------|------|-----|--------|
| **출력** | 고정 라벨 | BIO 태그 | **가변 길이 텍스트** |
| **위치** | 문장 전체 | 토큰 단위 | **문자 정밀도** |
| **복잡성** | 단순 | 중간 | **매우 복잡** |
| **후처리** | 거의 없음 | 간단 | **극도로 복잡** |
| **응용** | 감정분석 | 개체인식 | **검색, 챗봇** |

### **실무 핵심 포인트**

1. **전처리가 성능의 50%**:
   - 정확한 위치 매핑이 최종 성능 결정
   - Doc Stride 설정에 따른 성능 차이 극명
   - 언어별 특성 고려 필수

2. **후처리가 나머지 30%**:
   - 단순 argmax는 실무에서 사용 불가
   - N-best 크기와 임계값이 핵심 하이퍼파라미터
   - 도메인별 최적화 필요

3. **모델 자체는 20%**:
   - BERT vs ELECTRA vs RoBERTa 차이는 상대적으로 작음
   - 데이터 품질과 전후처리가 더 중요
   - 하이퍼파라미터 튜닝보다 엔지니어링이 핵심

## 🌟 확장 학습 주제

### **고급 QA 기법**:
- **Dense Passage Retrieval**: 대규모 문서에서 관련 구간 검색
- **Multi-hop QA**: 여러 문서를 연결한 추론
- **Conversational QA**: 대화 맥락을 고려한 질의응답

### **한국어 특화 최적화**:
- **형태소 기반 토크나이제이션**: 의미 단위 분절
- **한국어 평가 메트릭**: 언어적 특성 반영
- **도메인 적응**: 법률, 의료, 기술 문서 특화

### **프로덕션 최적화**:
- **모델 압축**: 양자화, 프루닝, 지식 증류
- **추론 가속**: ONNX, TensorRT, 배치 처리
- **분산 서비스**: 로드 밸런싱, 오토 스케일링

## 📈 수업 진행 전략

### **1단계: 개념 이해 (60분)**
- QA의 독특함 강조 (다른 태스크와의 차이)
- Start/End Logits 직관적 설명
- 간단한 예제로 전체 플로우 체험

### **2단계: 전처리 깊이 체험 (90분)**
- Doc Stride의 필요성과 효과 시연
- Offset Mapping 실습으로 정밀성 체감
- 실제 데이터의 복잡성 경험

### **3단계: 후처리 완전 정복 (120분)**
- N-best가 왜 필요한지 직접 확인
- 다양한 케이스에서 후처리 결과 비교
- SQuAD v2.0 "답변 없음" 로직 실습

### **4단계: 학습과 평가 (90분)**
- KorQuAD 데이터로 실제 학습 경험
- 한국어 QA의 특성과 도전 과제
- 평가 메트릭의 의미와 해석

### **5단계: 서비스 배포 (90분)**
- 실제 사용 가능한 웹 서비스 구축
- API 설계와 성능 최적화
- 실무 배포 시 고려사항 논의

## 💡 실무 연결 포인트

1. **검색 엔진**: 질의에 대한 정확한 답변 추출
2. **고객 지원**: FAQ 자동 응답 시스템
3. **교육 플랫폼**: 학습 자료 기반 질의응답
4. **법률/의료**: 전문 문서에서 정보 추출
5. **챗봇**: 대화형 AI의 핵심 구성 요소

이 단계별 학습을 통해 수강생들은 QA 시스템의 복잡성을 완전히 이해하고, 실무에서 바로 활용할 수 있는 전문성을 갖추게 됩니다! 🚀🎯
