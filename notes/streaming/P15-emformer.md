# P15. Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition

| 항목 | 내용 |
|---|---|
| 저자 | Shi, Yangyang; Wang, Yongqiang; Wu, Chunyang; Yeh, Ching-Feng; Chan, Julian; Zhang, Frank; Le, Duc; Seltzer, Mike |
| 확인한 원문 날짜 | 2020-10-21 |
| 발표 정보 | ICASSP 2021; arXiv 최초 공개 2020 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 2 — 제한 문맥 비교 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P15_emformer.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2010.10759) · [PDF](https://arxiv.org/pdf/2010.10759v4)

## 핵심과 방법

과거 문맥을 memory bank로 요약하고 청크 단위로 동작하는 저지연 Transformer acoustic model이다.

왼쪽 문맥 KV cache, 요약 메모리, 병렬 block training을 조합한다.

## 논문이 보고한 근거

LibriSpeech에서 서로 다른 latency 설정의 WER와 RTF를 비교한다.

확인 위치: Emformer memory 구조 및 latency별 평가.

## 연구 아이디어와의 연결 — 정리자의 해석

“과거 전체를 저장하지 않고 상태로 처리”하는 목표는 attention을 완전히 없애지 않고도 연구할 수 있음을 보여준다.

## 적용할 때의 제한

Mamba와 같은 recurrent state와 동일한 구조는 아니다. lookahead, chunk, memory bank의 예산을 맞춰 비교해야 한다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `ec1141086fdd61862a9baf64e6128bd4d62d3f9f50351d23fe739b92a94dc369`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
