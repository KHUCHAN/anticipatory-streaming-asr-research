# P33. Common Voice: A Massively-Multilingual Speech Corpus

| 항목 | 내용 |
|---|---|
| 저자 | Ardila, Rosana; Branson, Megan; Davis, Kelly; Henretty, Michael; Kohler, Michael; Meyer, Josh; Morais, Reuben; Saunders, Lindsay; Tyers, Francis M.; Weber, Gregor |
| 확인한 원문 날짜 | 2019-12-13 |
| 발표 정보 | Common Voice 원 논문, arXiv 2019 |
| 분류 | 학습 데이터셋 원 논문 |
| 읽기 우선순위 | 2 — 다국어 데이터 이해 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P33_common-voice.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/1912.06670) · [PDF](https://arxiv.org/pdf/1912.06670v2) · [추가 1차 출처 1](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490)

## 핵심과 방법

커뮤니티의 녹음·검증 참여로 구축한 다국어 음성 corpus Common Voice의 원 논문이다.

문장 녹음, 다수 참여자 검증, 공개 corpus 구축 과정을 설명한다.

## 논문이 보고한 근거

원 논문의 시점·언어 수·시간과 2026년 v27 릴리스 통계는 다른 버전의 정보다.

확인 위치: 원 논문 corpus design; 2026 v27 공식 릴리스 공지.

## 연구 아이디어와의 연결 — 정리자의 해석

억양과 언어 다양성에 대한 확장 경로를 제공한다. 먼저 언어·validated subset·speaker split을 명확히 해야 한다.

## 적용할 때의 제한

42,593시간은 v27 scripted 전체 언어의 녹음 총량이다. 영어 validated 학습 시간이나 자발 대화 시간으로 쓰지 않는다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `abda109167628b721e63a61a76f6701dd52a12b19a70cc1f8b3d66108b2dc03c`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
