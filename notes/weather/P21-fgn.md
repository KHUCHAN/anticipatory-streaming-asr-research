# P21. Skillful joint probabilistic weather forecasting from marginals

| 항목 | 내용 |
|---|---|
| 저자 | Alet, Ferran; Price, Ilan; El-Kadi, Andrew; Masters, Dominic; Markou, Stratis; Andersson, Tom R.; Stott, Jacklynn; Lam, Remi; Willson, Matthew; Sanchez-Gonzalez, Alvaro; Battaglia, Peter |
| 확인한 원문 날짜 | 2025-06-12 |
| 발표 정보 | arXiv technical report, 2025 |
| 분류 | FGN·GraphCast·WeatherNext |
| 읽기 우선순위 | 1 — FGN 원리 확인 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 53쪽; `papers/pdf/P21_fgn.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2506.10772) · [PDF](https://arxiv.org/pdf/2506.10772v1)

## 핵심과 방법

WeatherNext 2의 Functional Generative Network를 설명하는 확률적 기상 예측 논문이다.

최근 두 대기 상태에 조건을 두고, 32차원 공통 noise를 conditional normalization에 전달해 일관된 ensemble sample을 생성한다.

## 논문이 보고한 근거

2차 Markov 모형은 Eq. (1), noise 경로는 본문·부록에 명시된다. GenCast 대비 8배 가속은 반복 diffusion sampling과 비교한 기상 예측 결과다.

확인 위치: PDF 3쪽 Eq. (1), 7쪽 Section 3.3, 13쪽 및 22–23쪽.

## 연구 아이디어와의 연결 — 정리자의 해석

여러 미래 후보를 일관되게 샘플링하는 선택적 확장 아이디어의 출처다. 우선 핵심 predict–verify 효과를 분리한 뒤 검토할 수 있다.

## 적용할 때의 제한

32차원은 전체 기억 상태 크기가 아니라 확률적 변동을 제어하는 noise 차원이다. 날씨 결과는 ASR 가속이나 다음 단어 정확도를 증명하지 않는다.

## 코드와 모델

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `2141e6fe9b314143a2fe3655f6d41e74e804c573de4987e7e94b16faf313e16e`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
