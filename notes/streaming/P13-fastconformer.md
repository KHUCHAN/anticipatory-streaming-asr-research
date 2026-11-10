# P13. Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition

| 항목 | 내용 |
|---|---|
| 저자 | Rekesh, Dima; Koluguri, Nithin Rao; Kriman, Samuel; Majumdar, Somshubra; Noroozi, Vahid; Huang, He; Hrinchuk, Oleksii; Puvvada, Krishna; Kumar, Ankur; Balam, Jagadeesh; Ginsburg, Boris |
| 확인한 원문 날짜 | 2023-05-08 |
| 발표 정보 | ASRU 2023 accepted; 확인 원문 arXiv |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 2 — 효율 기준선 후보 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 8쪽; `papers/pdf/P13_fastconformer.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2305.05084) · [PDF](https://arxiv.org/pdf/2305.05084v6)

## 핵심과 방법

Conformer의 subsampling과 attention 구성을 바꿔 학습·추론 효율을 높인다.

강한 시간축 downsampling과 제한 문맥 attention을 이용한다.

## 논문이 보고한 근거

원문은 기본 Conformer 대비 2.8배 속도와 긴 음성 처리 설정을 보고한다.

확인 위치: architecture redesign 및 efficiency experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

speculation의 이득이 단순한 인코더 최적화보다 나은지 비교할 때 참고한다.

## 적용할 때의 제한

다른 하드웨어나 streaming 청크에서 동일한 가속이 보장되지 않는다. 11시간 처리 가능성이 그만큼의 native streaming 지원을 뜻하지 않는다.

## 코드와 모델

- [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

원문 NVIDIA/NeMo 주소는 현재 NVIDIA-NeMo/Speech로 redirect된다. 모델별 recipe·가중치·Mac 연산 지원은 실행 검증하지 않았다.

## 원문 버전

PDF SHA-256: `eb8c7706a567ec5795333139f8cd41ff9c127ba99b6dc85d378eb3fa3213bf8d`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
