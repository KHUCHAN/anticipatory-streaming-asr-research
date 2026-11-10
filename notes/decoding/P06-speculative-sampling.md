# P06. Accelerating Large Language Model Decoding with Speculative Sampling

| 항목 | 내용 |
|---|---|
| 저자 | Chen, Charlie; Borgeaud, Sebastian; Irving, Geoffrey; Lespiau, Jean-Baptiste; Sifre, Laurent; Jumper, John |
| 확인한 원문 날짜 | 2023-02-02 |
| 발표 정보 | arXiv technical report, 2023 |
| 분류 | 추측 디코딩과 경로 재사용 |
| 읽기 우선순위 | 2 — 이론 확인 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 11쪽; `papers/pdf/P06_speculative-sampling.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2302.01318) · [PDF](https://arxiv.org/pdf/2302.01318v1)

## 핵심과 방법

짧은 draft를 한 번에 scoring하고 수정된 rejection sampling으로 target 언어 모델의 분포를 보존한다.

target이 각 draft 위치를 병렬 평가하고 거절 위치에서는 잔여 분포로 보정한다.

## 논문이 보고한 근거

70B Chinchilla의 분산 환경에서 2–2.5배 디코딩 가속을 보고한다.

확인 위치: sampling algorithm 및 Chinchilla experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

분기 확률이 높다는 이유만으로 확정하지 않고 검증 규칙을 별도로 정의해야 한다는 근거다.

## 적용할 때의 제한

언어 모델의 고정 입력에서 성립하는 결과다. 스트리밍 음성의 시간 정렬과 새 입력 도착은 별도 문제다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `ffa03c6ae46f3122570bacd7da358cae8659b6421162bbc25088622fd4889c37`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
