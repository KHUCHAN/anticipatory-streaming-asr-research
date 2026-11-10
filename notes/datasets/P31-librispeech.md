# P31. LibriSpeech: An ASR Corpus Based on Public Domain Audio Books

| 항목 | 내용 |
|---|---|
| 저자 | Vassil Panayotov; Guoguo Chen; Daniel Povey; Sanjeev Khudanpur |
| 확인한 원문 날짜 | 2015 |
| 발표 정보 | ICASSP 2015 |
| 분류 | 학습 데이터셋 원 논문 |
| 읽기 우선순위 | 1 — 기본 데이터 이해 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P31_librispeech.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://www.openslr.org/resources/12/about.html) · [PDF](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf) · [추가 1차 출처 1](https://www.openslr.org/12)

## 핵심과 방법

LibriVox 공개 오디오북에서 수집한 16 kHz 영어 낭독 ASR corpus의 원 논문이다.

전사 정제와 발화 분할, train/dev/test를 제공하고 언어 모델 자원을 별도로 배포한다.

## 논문이 보고한 근거

약 1,000시간 corpus 및 baseline ASR 결과를 제시한다. 이 프로젝트는 train-clean-100과 네 dev/test split만 준비한다.

확인 위치: corpus construction 및 official OpenSLR split list.

## 연구 아이디어와의 연결 — 정리자의 해석

반복 가능한 작은 규모의 첫 비교와, 동일 utterance에서 여러 시간 prefix를 만드는 기반 데이터다.

## 적용할 때의 제한

낭독 음성이고 정제·분할되어 있어 자연 대화나 연속 마이크 환경과 차이가 있다. 단어 종료 시각은 별도 alignment가 필요하다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `0e6ee6ace1a55631b54e8671c25203418f8af00dcec3273ca06b1945b7a8a953`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
