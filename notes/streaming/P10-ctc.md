# P10. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks

| 항목 | 내용 |
|---|---|
| 저자 | Alex Graves; Santiago Fernández; Faustino Gomez; Jürgen Schmidhuber |
| 확인한 원문 날짜 | 2006 |
| 발표 정보 | ICML 2006 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 1 — 정렬 이론 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 8쪽; `papers/pdf/P10_ctc.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://www.cs.toronto.edu/~graves/icml_2006.pdf) · [PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

## 핵심과 방법

프레임별 정답 정렬 없이 음성과 라벨 시퀀스를 학습하는 Connectionist Temporal Classification의 원 논문이다.

blank, 반복 라벨 collapse, forward-backward를 통해 한 전사에 해당하는 모든 프레임 경로의 확률을 합한다.

## 논문이 보고한 근거

라벨 정렬이 주어지지 않은 음소 인식 문제에서 CTC를 평가한다.

확인 위치: Sections 2–4, CTC objective와 forward-backward.

## 연구 아이디어와의 연결 — 정리자의 해석

모델 구조를 정하기 전 음향 증거와 텍스트 prefix를 어떻게 연결할지 이해하는 기본 자료다.

## 적용할 때의 제한

CTC 자체가 streaming encoder를 보장하지 않는다. greedy 디코더는 이미 매우 저렴하여 후보 예측의 연산 절감 여지가 작을 수 있다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `ed7955f5d4ca3b068ef18855632e14195d63fd4ca7ea77d2c6ba02e94346d157`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
