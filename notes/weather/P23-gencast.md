# P23. GenCast: Diffusion-based ensemble forecasting for medium-range weather

| 항목 | 내용 |
|---|---|
| 저자 | Price, Ilan; Sanchez-Gonzalez, Alvaro; Alet, Ferran; Andersson, Tom R.; El-Kadi, Andrew; Masters, Dominic; Ewalds, Timo; Stott, Jacklynn; Mohamed, Shakir; Battaglia, Peter; Lam, Remi; Willson, Matthew |
| 확인한 원문 날짜 | 2023-12-25 |
| 발표 정보 | arXiv 2023; 관련 정식 논문 Nature, 2024 online |
| 분류 | FGN·GraphCast·WeatherNext |
| 읽기 우선순위 | 3 — FGN 비교 대상 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 96쪽; `papers/pdf/P23_gencast.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2312.15796) · [PDF](https://arxiv.org/pdf/2312.15796v2) · [추가 1차 출처 1](https://doi.org/10.1038/s41586-024-08252-9)

## 핵심과 방법

여러 가능한 미래 기상 경로를 생성하는 diffusion 기반 ensemble 예측 모델 GenCast다.

graph 기반 네트워크와 반복 denoising sampling으로 다음 상태의 조건부 분포를 모델링한다.

## 논문이 보고한 근거

기상 ensemble skill, 극한현상·태풍·풍력 예측을 평가한다. FGN 논문이 계산량을 비교하는 주요 대상이다.

확인 위치: probabilistic forecasting method 및 sampling procedure.

## 연구 아이디어와의 연결 — 정리자의 해석

단일 단계 생성과 반복 sampling의 비용 차이를 이해하고, ASR에서 실제 비교 대상을 명확히 하는 자료다.

## 적용할 때의 제한

이 카탈로그 PDF 제목은 arXiv의 GenCast 제목이다. Nature 정식 논문 제목은 “Probabilistic weather forecasting with machine learning”으로 다르다.

## 코드와 모델

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `4beaff56fa98f836a77808f9979397e59ad0dcc5f31fec169b77d05e7a1e67dd`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
