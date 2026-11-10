# P01. Speculative Speech Recognition by Audio-Prefixed Low-Rank Adaptation of Language Models

| 항목 | 내용 |
|---|---|
| 저자 | Yusuf, Bolaji; Baskar, Murali Karthick; Rosenberg, Andrew; Ramabhadran, Bhuvana |
| 확인한 원문 날짜 | 2024-07-05 |
| 발표 정보 | Interspeech 2024 |
| 분류 | 음성 도착 전 예측 |
| 읽기 우선순위 | 1 — 먼저 읽기 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P01_ssr.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2407.04641) · [PDF](https://arxiv.org/pdf/2407.04641v1) · [추가 1차 출처 1](https://www.isca-archive.org/interspeech_2024/yusuf24_interspeech.html)

## 핵심과 방법

현재까지 들은 음성과 전사를 이용해 아직 들리지 않은 발화의 뒷부분을 추측하는 Speculative Speech Recognition 연구다.

Conformer-Transducer 출력에 음성 조건부 soft prefix와 LoRA를 적용한 언어 모델을 연결한다. ASR 오류를 고려하는 학습 라벨 정렬도 제안한다.

## 논문이 보고한 근거

LibriSpeech 및 다중 도메인 실험에서 음성 조건부 예측을 평가한다. 기본 SOWER 설정은 미래 1초, 후보 8개이며 가장 좋은 후보를 사후 선택하는 oracle 평가다.

확인 위치: PDF 1–4쪽, Figure 1, Section 2.2, Eq. (8), Tables 1–3.

## 연구 아이디어와의 연결 — 정리자의 해석

“다음 말을 미리 준비하고 새 음성으로 확인한다”는 목표의 가장 직접적인 선행연구다. text-only 대비 audio-aware 예측 비교를 설계할 근거가 된다.

## 적용할 때의 제한

SOWER 개선을 실제 시스템의 단어 확정 지연 감소로 바꾸어 말할 수 없다. 이 논문은 작은 모델로 큰 모델의 계산을 줄이는 일반 speculative decoding과 문제를 구분한다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `8c556bc0943cc66a658cf9f2048743d5f548714cc683e5c20dc3a83fa9cca7ec`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
