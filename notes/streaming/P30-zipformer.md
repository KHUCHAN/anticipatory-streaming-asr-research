# P30. Zipformer: A faster and better encoder for automatic speech recognition

| 항목 | 내용 |
|---|---|
| 저자 | Yao, Zengwei; Guo, Liyong; Yang, Xiaoyu; Kang, Wei; Kuang, Fangjun; Yang, Yifan; Jin, Zengrui; Lin, Long; Povey, Daniel |
| 확인한 원문 날짜 | 2023-10-17 |
| 발표 정보 | Zipformer 원 논문, arXiv 2023 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 1 — 외부 streaming 기준선 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 16쪽; `papers/pdf/P30_zipformer.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2310.11230) · [PDF](https://arxiv.org/pdf/2310.11230v4) · [추가 1차 출처 1](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)

## 핵심과 방법

여러 시간 해상도의 인코더를 사용해 효율과 인식 성능을 높이는 Zipformer 논문이다.

U-Net 유사한 다중 frame rate 구조, attention weight 재사용, BiasNorm, Swoosh activation 및 ScaledAdam을 제안한다.

## 논문이 보고한 근거

LibriSpeech, AISHELL-1, WenetSpeech에서 비교하고 icefall 구현을 공개한다.

확인 위치: encoder design 및 ASR experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

sherpa-onnx의 20M streaming 모델을 외부 참조로 사용할 때 구조와 학습 출처를 이해할 근거다.

## 적용할 때의 제한

공개 20M 모델이 이 프로젝트의 100시간 학습과 동일한 조건인 것은 아니다. ONNX CPU와 PyTorch MPS의 속도 차이를 아키텍처 효과로 단정하지 않는다.

## 코드와 모델

- [https://github.com/k2-fsa/icefall](https://github.com/k2-fsa/icefall)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `94deedaae51eb7ad2016845c84c8e27c070cc0c0538162851ca41e18778b269b`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
