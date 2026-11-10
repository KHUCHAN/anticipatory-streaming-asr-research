# P02. Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models

| 항목 | 내용 |
|---|---|
| 저자 | Bian, Yutong; Zhang, Yanglin; Wang, Yuxiang; Zhang, Luoyuan; He, Rongshen; Wang, Zihao; Qiu, Junwen; Li, Jiaqi; Wu, Zhizheng |
| 확인한 원문 날짜 | 2026-11-04 |
| 발표 정보 | arXiv preprint, 2026-11-04 |
| 분류 | 음성 도착 전 예측 |
| 읽기 우선순위 | 1 — 먼저 읽기 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 13쪽; `papers/pdf/P02_speculative-listening.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2611.06982) · [PDF](https://arxiv.org/pdf/2611.06982v1)

## 핵심과 방법

실제 다음 음성 청크가 도착하기 전에 유휴 연산으로 출력을 초안 생성하고, 음성 도착 후 검증하는 full-duplex 음성 언어 모델 연구다.

silence hypothesis, 첫 토큰의 차순위 분기, 청크 간 hypothesis pool을 이용한다. 새 음성 아래 target 모델이 확인한 prefix만 채택한다.

## 논문이 보고한 근거

TickTock-Bench의 네 모델에서 speak chunk 2.6–2.8배 가속과 응답 시작 전 디코딩 지연 감소를 보고한다. 본문은 1초 청크와 저동시성 환경을 사용한다.

확인 위치: PDF 3–7쪽, Sections 3–5, Tables 1–2.

## 연구 아이디어와의 연결 — 정리자의 해석

음성을 기다리는 시간을 사전 계산에 쓰는 스케줄링과, 틀린 분기의 재사용·수정 절차에 직접 참고할 수 있다.

## 적용할 때의 제한

순수 ASR의 다음 단어 예측 실험이 아니다. silence 가정에서의 높은 출력 일치율을 미래 단어 예측 정확도로 해석하면 안 된다. lossless는 target 출력에 대한 보존이다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `31dbfc0ef28ccd92add078b8e04e230660e6a5ddb4330b3ef2b64a3c1d767f5c`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
