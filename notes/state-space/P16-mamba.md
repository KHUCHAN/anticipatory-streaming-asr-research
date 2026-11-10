# P16. Mamba: Linear-Time Sequence Modeling with Selective State Spaces

| 항목 | 내용 |
|---|---|
| 저자 | Gu, Albert; Dao, Tri |
| 확인한 원문 날짜 | 2023-12-01 |
| 발표 정보 | Mamba 원 논문, arXiv 2023 |
| 분류 | 상태공간 모델과 ConMamba |
| 읽기 우선순위 | 2 — 상태 압축 후보 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 36쪽; `papers/pdf/P16_mamba.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2312.00752) · [PDF](https://arxiv.org/pdf/2312.00752v2)

## 핵심과 방법

입력에 따라 정보를 선택하는 selective state space model로 긴 시퀀스를 효율적으로 처리한다.

입력 의존 SSM 파라미터와 hardware-aware scan을 사용하며, 추론 시 recurrent state를 갱신한다.

## 논문이 보고한 근거

언어·오디오 등 sequence modeling에서 성능과 처리량을 평가한다.

확인 위치: selective SSM, hardware-aware scan 및 evaluation.

## 연구 아이디어와의 연결 — 정리자의 해석

전체 이력 KV cache 대신 제한된 상태를 유지하려는 목표의 핵심 배경이다.

## 적용할 때의 제한

Mamba가 ASR 정렬이나 미래 단어 검증을 자동 해결하지 않는다. 원 구현의 CUDA 최적화 결과를 Apple MPS 성능으로 일반화하지 않는다.

## 코드와 모델

- [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `adf70ed1803c85b1899dec3e21f3af0b124411439e8654b840ea65f7b9f52b2e`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
