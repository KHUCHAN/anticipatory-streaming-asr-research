# P26. Deep Graph Random Process for Relational-Thinking-Based Speech Recognition

| 항목 | 내용 |
|---|---|
| 저자 | Hengguan Huang; Fuzhao Xue; Hao Wang; Ye Wang |
| 확인한 원문 날짜 | 2020-11-21 |
| 발표 정보 | ICML 2020, PMLR 119 |
| 분류 | 그래프 기반 음성인식 |
| 읽기 우선순위 | 3 — relational ASR 배경 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 11쪽; `papers/pdf/P26_deep-graph-random-process.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://proceedings.mlr.press/v119/huang20k.html) · [PDF](https://proceedings.mlr.press/v119/huang20k/huang20k.pdf)

## 핵심과 방법

발화 간 관계를 잠재적인 확률 graph로 모델링하는 Deep Graph Random Process다.

Bayesian nonparametric 방법으로 여러 percept graph를 추론해 acoustic modeling에 반영한다.

## 논문이 보고한 근거

CHiME-2와 CHiME-5 ASR에서 관계 모델링의 효과를 평가한다.

확인 위치: DGP formulation 및 CHiME experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

graph 기반 음성인식의 범위를 파악하는 배경이다. 후보 전사 tree와 발화 관계 graph를 혼동하지 않도록 한다.

## 적용할 때의 제한

Google의 FGN이나 Factor Graph NN을 사용한 ASR이 아니다. 저지연 후보 선행 계산이 핵심 목적도 아니다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

PMLR 원문이 연결한 https://github.com/GlenHGHUANG/Deep_graph_random_process 는 2026-11-10 HTTP 404였다. 현재 이용 가능한 공식 구현은 확인하지 못했다.

## 원문 버전

PDF SHA-256: `15111002f2b1dc815af017f564e830b9fa62d1bd652c48471e11677289154828`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
