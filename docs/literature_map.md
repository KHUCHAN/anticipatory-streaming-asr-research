# 논문 지도와 읽는 순서

## 핵심 구분

| 문제 | 실제로 아직 도착하지 않은 음성을 예측하나? | 우선 논문 |
|---|---|---|
| 미래 발화 텍스트 예측 | 예 | [P01 SSR](../notes/anticipatory/P01-ssr.md) |
| 입력 대기 중 출력 사전 계산 | 다음 청크를 가정해 계산; full-duplex SLM | [P02 Speculative Listening](../notes/anticipatory/P02-speculative-listening.md) |
| 주어진 음성의 decoder 가속 | 해당 사실만으로는 아님 | [P03 SpecASR](../notes/decoding/P03-specasr.md), [P04 ParaASR](../notes/decoding/P04-paraasr.md) |
| 확률적 draft 검증 | 보통 고정 입력의 target 분포 보존 | [P05](../notes/decoding/P05-speculative-decoding.md), [P06](../notes/decoding/P06-speculative-sampling.md) |
| 복수 후보 tree 검증 | 그 자체는 음성 예측이 아님 | [P07 SpecInfer](../notes/decoding/P07-specinfer.md), [P08 Blockwise](../notes/decoding/P08-blockwise-parallel.md) |
| 제한 상태·문맥 인코딩 | 과거 처리의 효율 문제 | [P15 Emformer](../notes/streaming/P15-emformer.md), [P16 Mamba](../notes/state-space/P16-mamba.md), [P20 LC-BiMamba](../notes/state-space/P20-lc-bimamba.md) |
| 함수적 확률 경로 생성 | 기상 상태의 확률 예측 | [P21 FGN](../notes/weather/P21-fgn.md) |

## 권장 읽기 순서

1. **목표가 이미 어디까지 연구됐는지:** P01 → P02 → P03 → P04.
2. **채택·거절·복구의 원리:** P05 → P06 → P07 → P08.
3. **음성과 텍스트 정렬:** P10 CTC → P09 RNN-T → P11 TDT.
4. **streaming과 메모리 제약:** P15 → P18 Speech Slytherin → P20 → P30 Zipformer.
5. **FGN의 실제 역할:** P21 → P22 GraphCast → P23 GenCast → P29 WeatherNext Cyclones.
6. **후보 graph 및 약어 혼동:** P25 → P24 → P26; P27·P28은 별도 동명 연구.
7. **데이터 출처:** P31 LibriSpeech → P32 GigaSpeech → P33 Common Voice.

## 방법을 비교할 때 질문할 것

각 논문에서 입력 음성의 범위, 미래 문맥, 출력 단위, 채택 규칙, fallback, 실제 하드웨어, 모델 크기, 유휴 시간 활용 여부를 확인한다. “accuracy unchanged”가 target 모델 출력 보존인지, corpus WER의 실험적 유사성인지도 구분한다.

논문끼리 속도 배수를 순위로 나열하면 비교 기준이 달라진다. 특히 기상 diffusion 대비 속도, LLM decoder 가속, streaming word commit 지연은 서로 다른 수치다.

[33편 전체 목록](../papers/INDEX.md) · [연구 목표와 방법론](goal_and_methodology.md)
