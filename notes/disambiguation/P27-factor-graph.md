# P27. Factor Graph Neural Networks

| 항목 | 내용 |
|---|---|
| 저자 | Zhen Zhang; Mohammed Haroon Dupty; Fan Wu; Javen Qinfeng Shi; Wee Sun Lee |
| 확인한 원문 날짜 | 2023 |
| 발표 정보 | JMLR 24(181):1–54, 2023 |
| 분류 | FGN/FGNN 동명 아키텍처 구분 |
| 읽기 우선순위 | 3 — 약어 혼동 방지 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 54쪽; `papers/pdf/P27_factor-graph.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://jmlr.org/papers/v24/21-0434.html) · [PDF](https://jmlr.org/papers/volume24/21-0434/21-0434.pdf)

## 핵심과 방법

FGNN이 Factor Graph Neural Network를 뜻하는 고차 관계 학습 연구다.

factor와 variable 사이의 message passing을 신경망화하여 sum-product 및 max-product 추론을 표현한다.

## 논문이 보고한 근거

합성 graphical model, LDPC decoding, graph matching 등을 평가한다.

확인 위치: JMLR abstract 및 Section 5 experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

사용자가 말한 Google FGN과 이름이 비슷하지만 별도 계열이라는 점을 기록한다.

## 적용할 때의 제한

논문에서 ASR 학습 모델을 제시한 것으로 확인하지 않았다. 이 논문만으로 FGNN-ASR 공개 체크포인트의 존재를 주장할 수 없다.

## 코드와 모델

- [https://github.com/zzhang1987/Factor-Graph-Neural-Network](https://github.com/zzhang1987/Factor-Graph-Neural-Network)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `6b198dfceb67a11bf2b807231c67f8c34f8f8f98a5bb5585d671dd077d330cae`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
