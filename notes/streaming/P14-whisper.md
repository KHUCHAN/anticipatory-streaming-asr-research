# P14. Robust Speech Recognition via Large-Scale Weak Supervision

| 항목 | 내용 |
|---|---|
| 저자 | Radford, Alec; Kim, Jong Wook; Xu, Tao; Brockman, Greg; McLeavey, Christine; Sutskever, Ilya |
| 확인한 원문 날짜 | 2022-12-06 |
| 발표 정보 | Whisper 원 논문, arXiv 2022 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 2 — 외부 정확도 참고 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 28쪽; `papers/pdf/P14_whisper.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2212.04356) · [PDF](https://arxiv.org/pdf/2212.04356v1)

## 핵심과 방법

대규모 약지도 음성·전사 학습으로 여러 언어와 도메인의 음성 처리를 수행하는 Whisper 논문이다.

audio encoder와 autoregressive text decoder로 학습하며 다국어·다중 작업 supervision을 사용한다.

## 논문이 보고한 근거

680,000시간 규모 학습과 다양한 zero-shot 인식 평가를 보고한다.

확인 위치: training data 및 evaluation sections.

## 연구 아이디어와의 연결 — 정리자의 해석

모든 모델을 처음부터 학습하는 실험과, 대규모 사전학습 외부 모델의 성능을 구분하는 기준이다.

## 적용할 때의 제한

기본 모델은 실시간 streaming 전용 구조가 아니다. 외부 pretrained Whisper의 WER과 100시간 자체 학습 모델의 WER을 공정한 동일 조건 비교로 제시하지 않는다.

## 코드와 모델

- [https://github.com/openai/whisper](https://github.com/openai/whisper)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `6337bde031b2f237547a977b022f831169a7e05b4d9047f29501166d83594566`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
