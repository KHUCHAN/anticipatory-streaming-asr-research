# P18. Speech Slytherin: Examining the Performance and Efficiency of Mamba for Speech Separation, Recognition, and Synthesis

| 항목 | 내용 |
|---|---|
| 저자 | Jiang, Xilin; Li, Yinghao Aaron; Florea, Adrian Nicolas; Han, Cong; Mesgarani, Nima |
| 확인한 원문 날짜 | 2024-07-13 |
| 발표 정보 | arXiv preprint, 2024 |
| 분류 | 상태공간 모델과 ConMamba |
| 읽기 우선순위 | 1 — 상태공간 적용 근거 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 9쪽; `papers/pdf/P18_speech-slytherin.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2407.09732) · [PDF](https://arxiv.org/pdf/2407.09732v1)

## 핵심과 방법

Mamba를 음원 분리·음성인식·합성에 적용하며 ConMamba를 제시한 연구다.

음성인식에는 Conformer의 attention을 Mamba 계열 모듈로 바꾼 구조를 평가한다.

## 논문이 보고한 근거

Mamba의 효율상 이점은 입력 길이와 작업에 의존하며, 짧은 음성에서는 항상 유리하지 않다고 보고한다.

확인 위치: ConMamba 및 speed/memory evaluation.

## 연구 아이디어와의 연결 — 정리자의 해석

긴 상태를 줄이는 접근의 가능성과, 짧은 실시간 청크에서 이득이 사라질 위험을 함께 검토할 수 있다.

## 적용할 때의 제한

bidirectional 구성과 전체 발화 실험을 strict streaming 결과로 취급하지 않는다. 청크 경계의 상태 처리와 학습 예산을 따로 검증해야 한다.

## 코드와 모델

- [https://github.com/xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `5cc3c3aeebac17170e5880843ca760aad202424aa4a232101086bb1add8eaa76`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
