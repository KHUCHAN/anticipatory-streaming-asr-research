# P09. Sequence Transduction with Recurrent Neural Networks

| 항목 | 내용 |
|---|---|
| 저자 | Graves, Alex |
| 확인한 원문 날짜 | 2012-11-14 |
| 발표 정보 | arXiv preprint, 2012 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 1 — 정렬 이론 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 9쪽; `papers/pdf/P09_rnnt.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/1211.3711) · [PDF](https://arxiv.org/pdf/1211.3711v1)

## 핵심과 방법

음성과 출력 라벨의 길이가 다른 문제를 확률적으로 다루는 RNN Transducer의 원 논문이다.

입력 인코더와 출력 이력의 prediction network를 joint distribution으로 결합하고 가능한 정렬들을 합산한다.

## 논문이 보고한 근거

TIMIT의 음소 인식으로 sequence transduction을 평가한다.

확인 위치: transducer formulation 및 TIMIT experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

음성 프레임과 단어가 일대일로 대응하지 않는 상황에서 후보를 음향 증거로 검증할 정렬 기반을 제공한다.

## 적용할 때의 제한

원 논문의 설정이 곧바로 현대적 causal streaming 구현임을 뜻하지 않는다. encoder의 미래 문맥과 decoding 정책을 따로 정해야 한다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `d28fddcfc95a646ad60d3007a7f5fbd0fbba35eec418c018a595da81089f06cf`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
