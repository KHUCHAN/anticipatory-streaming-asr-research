# P05. Fast Inference from Transformers via Speculative Decoding

| 항목 | 내용 |
|---|---|
| 저자 | Leviathan, Yaniv; Kalman, Matan; Matias, Yossi |
| 확인한 원문 날짜 | 2022-11-30 |
| 발표 정보 | ICML 2023; arXiv 최초 공개 2022 |
| 분류 | 추측 디코딩과 경로 재사용 |
| 읽기 우선순위 | 1 — 먼저 읽기 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 13쪽; `papers/pdf/P05_speculative-decoding.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2211.17192) · [PDF](https://arxiv.org/pdf/2211.17192v2) · [추가 1차 출처 1](https://proceedings.mlr.press/v202/leviathan23a.html)

## 핵심과 방법

작은 draft 모델의 후보를 큰 target 모델이 병렬 확인하는 추측 디코딩의 기반 논문이다.

draft 분포와 target 분포를 이용한 채택 및 보정 샘플링으로 target 분포를 유지한다.

## 논문이 보고한 근거

T5-XXL을 대상으로 표준 구현 대비 2–3배 가속을 보고한다. 보존 대상은 target 모델의 생성 분포다.

확인 위치: Algorithm 1, correctness analysis, experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

“맞으면 채택하고 아니면 수정”을 구현할 때 필요한 확률적 채택 규칙의 출발점이다.

## 적용할 때의 제한

ASR 정답이나 아직 들리지 않은 음성 내용이 보장되는 것은 아니다. 입력이 바뀌면 그 입력 아래 target 분포로 다시 검증해야 한다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `8967e2daab74178b8a73a6f0b0780c5cbb40f9f6dcbd5c456b24a89c128c52ed`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
