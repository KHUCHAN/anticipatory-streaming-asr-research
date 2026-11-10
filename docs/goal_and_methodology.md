# 연구 목표와 방법론

작성·근거 확인일: 2026-11-10. 상태: **데이터·문헌 준비 단계. 모델 아키텍처 미확정, 학습 미시작.**

## 1. 사용자가 제안한 목표

> 현재까지 들은 음성과 문맥으로 다음에 말할 단어 경로를 미리 계산한다. 실제 음성이 들어오면 후보를 검증하고, 맞으면 준비된 경로를 재사용하며 진행한다. 다르면 새 음향 증거에 맞는 후보로 전환하거나 다시 계산한다.

목표는 **동일한 정확도 수준에서 실제 음성으로 확인된 전사의 확정 지연과 불필요한 순차 연산을 줄이는 streaming ASR**이다. 음성이 도착하기를 기다리는 시간에 다음 후보를 준비한다. 긴 과거 전체를 매번 참조하기보다 제한된 상태나 문맥으로 처리할 가능성도 검토한다.

연구 질문:

**Can anticipatory candidate generation and acoustic verification reduce committed-transcript latency and decoder work at matched recognition accuracy and a controlled memory budget?**

아직 들리지 않은 단어를 예측하는 능력, 이미 받은 음성의 디코딩을 빠르게 하는 능력, 과거 상태의 메모리를 줄이는 능력은 별도로 검증한다. 이 세 효과를 하나의 “속도 향상” 수치로 합치지 않는다.

## 2. 현재 확정한 범위

| 항목 | 준비 기준 |
|---|---|
| 작업 | 영어 streaming speech-to-text의 연구 준비 |
| 우선 데이터 | LibriSpeech train-clean-100 + dev-clean/dev-other + test-clean/test-other |
| 작업 환경 | Apple M5 Pro, 통합 메모리 24 GiB |
| 평가 중심 | 최종 WER, 확정 지연 P50/P95, 실제 연산량·메모리 |
| 데이터 준비 수준 | 원음, 전사, 발화 manifest, 체크섬, split·화자 중복 검사 |
| 미확정 | encoder, decoder/정렬 방식, state 구조, tokenizer, 후보 폭·길이, 학습 recipe |
| 이후 선택지 | CTC / RNN-T / TDT, 제한 문맥 attention / recurrent SSM, graph / FGN-style sampling |

FGN은 Google 기상 모델에서 얻은 **복수 미래 경로 생성의 발상**이다. 핵심 predict–verify 방법이 FGN을 반드시 요구하는 것은 아니다. GNN·Mamba·Conformer 등의 채택은 문헌과 실제 연산 측정 후 결정한다.

## 3. 방법의 구성 — 모델 구조와 독립적인 동작 정의

시각 `t`까지 실제로 도착한 음성을 `a[0:t]`, 확인된 전사 prefix를 `C_t`, 잠정 후보 집합을 `D_t`라 한다.

1. **관측 정리:** 실제로 받은 음향 정보와 확인된 텍스트만 현재 문맥에 반영한다.
2. **선행 예측:** 다음 단어 또는 subword 경로를 소수 생성한다. 후보마다 생성 시각, prefix, 확률, 필요한 임시 상태와 예상 유효 범위를 기록한다.
3. **준비된 계산 보관:** 공통 prefix를 공유하고 재사용 가능한 decoder 상태를 보관한다. 상태·후보 예산을 명시한다.
4. **음향 검증:** 새 청크가 들어오면 인코딩·정렬을 거쳐 후보를 실제 음성 아래 재평가한다. 언어적으로 자연스럽다는 이유만으로 채택하지 않는다.
5. **채택 또는 수정:** 검증된 prefix를 확정하고 재사용한다. 불일치 부분은 다른 후보로 전환하거나 정상 decoding으로 복구한다.
6. **다음 후보 갱신:** 수정된 prefix에 맞추어 오래된 후보와 상태를 무효화하고 다음 예측을 준비한다.

이는 절차 정의이며 encoder block, layer 수, 차원, 학습 코드를 확정한 설계가 아니다.

```text
도착한 음성 + 확정 prefix
        ↓
다음 경로 선행 계산 → 후보·재사용 상태 보관
        ↑                         ↓
   문맥 갱신 ← 채택 / 분기 / 복구 ← 새 음성으로 검증
```

## 4. 잠정 출력과 확정 출력

- **잠정 후보:** 미래 발화를 예측한 결과다. 실제 발화와 달라질 수 있다.
- **확정 전사:** 정해진 음향 검증 및 commit 정책을 통과한 결과다.
- **재사용:** 후보를 미리 만들었다는 사실만으로 성립하지 않는다. 실제로 절약한 target decoder 호출, 상태 갱신 또는 탐색량을 측정한다.

음성이 도착하기 전 결과를 보여주는 것과, 실제로 들은 말을 수정 없이 확정하는 것은 다르다. 전자는 prediction lead time으로, 후자는 commit latency와 false commit으로 평가한다. 최종 WER만 좋아도 중간에 잘못 확정했다면 그 비용을 별도로 보고한다.

## 5. 검증할 가설

| 가설 | 확인할 결과 | 반증 또는 실패 신호 |
|---|---|---|
| H1 선행 계산 | 같은 WER 수준에서 commit latency 감소 | 예측 표시만 빨라지고 확정 시점은 그대로 |
| H2 계산 재사용 | target 호출·실제 decoder 시간이 감소 | predictor·분기 관리 비용이 절감량보다 큼 |
| H3 음성 조건부 예측 | text-only보다 유용한 후보·승인 길이 증가 | 단순 LM 결합과 차이가 없음 |
| H4 제한 상태 | 입력 시간이 늘어도 상태·후보 메모리 예산 유지 | 전체 이력 cache나 분기 저장량이 계속 증가 |
| H5 복구 가능성 | 오예측 후 수정 지연·false commit 통제 | 고유명사·숫자·낮은 예측 가능성에서 오류 고착 |

FGN을 추가할 경우의 가설은 “후보의 유용한 다양성이 늘어 총 지연을 줄이는가”이다. noise를 넣으면 자동으로 더 정확해진다는 가정은 하지 않는다.

## 6. 비교 방법

아래 B0–B3는 기능 비교군의 정의이며 구체적인 모델 아키텍처 선택은 남겨 둔다.

| 비교군 | 기능 | 분리하려는 효과 |
|---|---|---|
| B0 | 같은 acoustic model의 기본 streaming decoding | 음성 인식 기준 |
| B1 | 같은 모델 + LM 또는 prefix search | 언어 문맥만 추가한 효과 |
| B2 | 같은 모델 + text-only 선행 후보·검증 | 사전 계산·재사용 효과 |
| B3 | 같은 모델 + audio-aware 선행 후보·검증 | 현재 음향 정보를 예측에 쓰는 추가 효과 |

encoder, tokenizer, 학습 데이터, 증강, 학습량, 청크·lookahead, endpointing, commit 정책을 통제한다. predictor의 추가 파라미터·학습·전력·메모리도 실험 예산에 포함한다. 단순한 모델 크기 증가나 더 큰 beam의 효과와도 비교한다.

CTC greedy의 디코딩은 원래 가벼울 수 있다. 이 경우 후보 모델의 추가 비용이 순수 연산 가속을 상쇄할 수 있으므로 결과가 유리하다고 가정하지 않는다. 순차 decoding 비용이 큰 transducer 등과의 비교 여부는 아키텍처 결정 단계에서 판단한다.

## 7. 데이터와 라벨 생성 원칙

원본 utterance의 train/dev/test를 먼저 유지하고, 그 안에서 시간 prefix·청크를 만든다. 한 발화에서 생성한 다른 prefix를 서로 다른 split에 섞지 않는다.

향후 한 학습 사례는 `관측 cutoff`, `cutoff 이전 음성`, `이전 전사 가설`, `다음 정답 구간`, `필요한 alignment 정보`를 가질 수 있다. 미래 텍스트를 **정답 라벨**로 쓰는 것과 미래 음향을 **모델 입력**으로 넣는 것을 구분한다. 전체 발화 forced alignment는 오프라인 라벨·평가 시각 생성에만 쓰고, 그 결과가 미래 정보를 입력 상태에 흘리지 않도록 한다.

모든 입력 특징과 정규화의 인과성을 검사해야 한다. 전 발화 정규화, centered convolution, 양방향 encoder, 미래 frame cache는 prefix 실험에서 특히 확인할 대상이다. LM 학습과 tokenizer 선택에도 test 전사를 사용하지 않는다. **현재 단계에서는 tokenizer, acoustic feature, word alignment를 생성하지 않았다.**

## 8. 시간·비용 측정

| 지표 | 정의 또는 기록 기준 |
|---|---|
| WER/CER | 고정한 정규화 규칙 아래 최종 전사 오류 |
| Commit latency | 기준 단어 종료 시각부터 시스템의 실제 commit 이벤트까지; P50/P95와 false commit 동시 보고 |
| Prediction lead time | 후보를 만든 시각이 실제 단어 종료보다 얼마나 앞섰는지 |
| Hit@K / prefix acceptance | 고정한 후보 수·미래 horizon에서 정답 포함 여부 및 실제 채택 길이 |
| Revision / false commit | 잠정 텍스트 수정과 이미 확정한 텍스트 오류를 구분 |
| RTF | 처리 시간 / 음성 시간; 오프라인 처리량과 실시간 paced replay를 따로 측정 |
| 연산 분해 | encoder, predictor, verifier, fallback, branch 관리 각각의 시간과 호출 수 |
| 자원 | peak memory, cache/state 크기, 후보 수, 총 CPU/GPU 작업량 |

MPS 연산은 비동기 실행을 고려해 동기화한 시간도 측정한다. 미래 음성은 실제 도착 시각 전에는 넘기지 않는다. 통계 비교는 동일 발화에 대해 짝을 맞추고, WER 허용 차이와 반복 횟수·신뢰구간 방식은 실험 전에 정한다.

단어 종료 시각을 forced alignment로 만들면 시각 오차가 존재하므로 평가 도구·버전과 허용 오차를 기록한다. 종료 전 commit에서 나온 음수 지연도 임의로 0으로 잘라 없애지 않고, 그 조건과 false commit을 함께 보고한다.

## 9. 선행연구와 신규성의 경계

[SSR 2024](../notes/anticipatory/P01-ssr.md)는 미래 발화 예측을 이미 다룬다. [Speculative Listening 2026](../notes/anticipatory/P02-speculative-listening.md)은 음성 대기 시간의 사전 계산을 다루며, [SpecASR](../notes/decoding/P03-specasr.md)와 [ParaASR](../notes/decoding/P04-paraasr.md)는 디코딩 경로 재사용·다중 토큰 검증의 선행연구다.

따라서 “미리 예측하고 검증한다”는 개념 자체를 새롭다고 주장할 수 없다. 이 프로젝트의 잠재 기여는 **실제 streaming ASR에서 제한된 상태·분기 예산으로 재사용을 수행하고, 정확도·확정 지연·총 연산을 함께 검증하는 구체적 방법**에 있다. 무엇이 새로운지는 구현 선택과 선행연구 재현 이후에 판단한다.

## 10. 다음 단계로 넘길 결정 항목

모델 설계 전에는 데이터 inventory와 원문 근거, 입력의 인과성, 후보·확정 이벤트 정의, 비교군과 측정 규칙을 검토한다. 그다음 사용자가 설계를 요청하면 정렬 방식, 상태 관리, 후보 검증 규칙 및 Mac 실행 가능성을 함께 고려해 아키텍처를 결정한다.

[논문 지도](literature_map.md) · [사실 검토](claim_audit.md) · [데이터·Mac 가이드](datasets_and_mac.md)
