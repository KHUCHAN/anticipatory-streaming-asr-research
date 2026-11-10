# P20. Attention-Free Dual-Mode ASR with Latency-Controlled Selective State Spaces

| 항목 | 내용 |
|---|---|
| 저자 | Moriya, Takafumi; Mimura, Masato; Matsui, Kiyoaki; Sato, Hiroshi; Matsuura, Kohei |
| 확인한 원문 날짜 | 2025 |
| 발표 정보 | Interspeech 2025 |
| 분류 | 상태공간 모델과 ConMamba |
| 읽기 우선순위 | 1 — streaming 상태공간 근거 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P20_lc-bimamba.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.pdf)

## 핵심과 방법

Mamba 기반 인코더를 streaming과 offline에 함께 사용하는 attention-free dual-mode ASR 연구다.

LC-BiMamba로 청크 안의 제한된 미래 문맥을 사용하고 AdaALSD로 transducer 탐색을 개선한다.

## 논문이 보고한 근거

TLv2와 LibriSpeech에서 Conformer와 WER·RTF를 비교한다. LibriSpeech 표는 RTF 0.58 대 0.50/0.51 등 특정 설정 결과를 보여준다.

확인 위치: PDF 1–4쪽, Table 5 및 latency 설정.

## 연구 아이디어와의 연결 — 정리자의 해석

상태공간 모델을 실제 streaming 조건에서 비교할 때 필요한 청크·미래 문맥 통제의 직접 근거다.

## 적용할 때의 제한

latency-controlled는 미래 문맥이 0이라는 뜻이 아니다. constant-time이라는 표현을 전체 음성을 O(1) 시간에 처리한다는 뜻으로 읽으면 안 된다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `de00b9bd79b0da7b7fb421e167df1deb6398628ae5fc2e7bf2f91b47cb3cf155`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
