# P03. SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding

| 항목 | 내용 |
|---|---|
| 저자 | Wei, Linye; Zhong, Shuzhang; Xu, Songqiang; Wang, Runsheng; Huang, Ru; Li, Meng |
| 확인한 원문 날짜 | 2025-07-24 |
| 발표 정보 | arXiv 2025; 원문에 DAC 2025 accepted 기재 |
| 분류 | 추측 디코딩과 경로 재사용 |
| 읽기 우선순위 | 1 — 먼저 읽기 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 7쪽; `papers/pdf/P03_specasr.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2507.18181) · [PDF](https://arxiv.org/pdf/2507.18181v2)

## 핵심과 방법

음성 조건부 작은 ASR과 큰 ASR의 디코딩 경로 유사성을 이용해 추측 디코딩을 가속한다.

adaptive draft length, draft recycling, two-pass sparse token tree를 결합하고 여러 후보 경로를 병렬 검증한다.

## 논문이 보고한 근거

3.04–3.79배는 Vicuna-13B를 simulated target으로 사용한 결과다. 실험은 RTX A6000에서 Whisper tiny.en/medium.en 경로를 기록한 뒤 TinyLlama/Llama/Vicuna로 지연을 시뮬레이션한다.

확인 위치: PDF 5쪽 Section V-A/V-B 및 Figure 11; 6쪽 ablation.

## 연구 아이디어와의 연결 — 정리자의 해석

후보의 재계산을 줄이는 recycling과 분기 예산 제어가 이 프로젝트의 후보 재사용 정책에 관련된다.

## 적용할 때의 제한

실제 음성이 오기 전 예측을 증명한 연구가 아니다. 보고된 가속을 완성된 Llama-ASR 배포나 Mac 실시간 성능으로 인용하지 않는다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `5bede7b1c67a5c37559cd584c92c0d25ce188c597985d0352caad3021bafd5ea`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
