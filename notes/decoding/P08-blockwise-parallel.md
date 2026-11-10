# P08. Blockwise Parallel Decoding for Deep Autoregressive Models

| 항목 | 내용 |
|---|---|
| 저자 | Stern, Mitchell; Shazeer, Noam; Uszkoreit, Jakob |
| 확인한 원문 날짜 | 2018-11-07 |
| 발표 정보 | arXiv preprint, 2018 |
| 분류 | 추측 디코딩과 경로 재사용 |
| 읽기 우선순위 | 2 — 다중 토큰 기반 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 10쪽; `papers/pdf/P08_blockwise-parallel.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/1811.03115) · [PDF](https://arxiv.org/pdf/1811.03115v1)

## 핵심과 방법

여러 미래 시점의 출력을 병렬로 제안하고 scoring 모델이 인정하는 가장 긴 prefix로 진행한다.

다음 여러 위치를 예측하는 보조 모델과 검증·back-off를 이용해 순차 디코딩 반복을 줄인다.

## 논문이 보고한 근거

기계번역과 이미지 super-resolution에서 반복 수 및 실제 실행시간의 개선을 평가한다.

확인 위치: blockwise decoding algorithm 및 experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

연속 후보 제안과 verified-prefix 채택의 오래된 선행연구로, 방법의 신규성 범위를 정리하는 데 필요하다.

## 적용할 때의 제한

음성 도착 시간이나 단어 정렬을 다루지 않는다. 논문의 품질 보존 설정과 품질·속도 절충 설정을 구분해야 한다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `e34c3c9ddd1066b7dc2c11406580a586a6d483a23e6362e1414f760b655c52e1`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
