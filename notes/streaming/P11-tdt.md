# P11. Efficient Sequence Transduction by Jointly Predicting Tokens and Durations

| 항목 | 내용 |
|---|---|
| 저자 | Hainan Xu; Fei Jia; Somshubra Majumdar; He Huang; Shinji Watanabe; Boris Ginsburg |
| 확인한 원문 날짜 | 2023-07-03 |
| 발표 정보 | ICML 2023, PMLR 202 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 2 — 시간축 건너뛰기 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 23쪽; `papers/pdf/P11_tdt.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://proceedings.mlr.press/v202/xu23g.html) · [PDF](https://proceedings.mlr.press/v202/xu23g/xu23g.pdf)

## 핵심과 방법

토큰과 함께 입력 프레임의 진행량을 예측하는 Token-and-Duration Transducer다.

독립적으로 정규화한 token·duration 출력으로 transducer의 디코딩 시간축을 여러 프레임씩 진행한다.

## 논문이 보고한 근거

음성인식에서 기존 transducer 대비 최대 2.82배 추론 가속을 보고한다.

확인 위치: TDT formulation 및 ASR experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

후보 선행 예측과 별개로, 검증·출력 단계의 joiner 호출을 줄이는 비교 방법이다.

## 적용할 때의 제한

모든 음향 인코딩을 생략한다는 의미가 아니다. TDT 계열 체크포인트의 native streaming 지원은 해당 모델 설정별로 확인해야 한다.

## 코드와 모델

- [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

원문 NVIDIA/NeMo 주소는 현재 NVIDIA-NeMo/Speech로 redirect된다. 모델별 recipe·가중치·Mac 연산 지원은 실행 검증하지 않았다.

## 원문 버전

PDF SHA-256: `b637cea35e648e9f3469dac7f76d67b09308947b53d1ada81f5991100fd45519`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
