# P29. Operational tropical cyclone forecasting with AI

| 항목 | 내용 |
|---|---|
| 저자 | Alet, Ferran; Andersson, Tom R.; Price, Ilan; Markou, Stratis; El-Kadi, Andrew; Masters, Dominic; Li, Amy; Merchant, Samier; Williams, Natalie; Thornton, Gregory; MacKay, Ken; Graham, Olivia; Uddin, Akib; Gaiarin, Ben; Shah, Devaja; Kruse, Elinor; Hogsett, Wallace; Zelinsky, David; Cangialosi, John; Martinez, Jonathan; Franklin, James; DeMaria, Mark; Musgrave, Kate; Bain, Caroline L.; Titley, Helen; Stott, Jacklynn; Lam, Remi; Bell, Aaron; Komarek, Paul; Willson, Matthew; Sanchez-Gonzalez, Alvaro; Battaglia, Peter |
| 확인한 원문 날짜 | 2026-08-06 |
| 발표 정보 | Nature 657, 2026; online 2026-08-06 |
| 분류 | FGN·GraphCast·WeatherNext |
| 읽기 우선순위 | 3 — FGN 후속 응용 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 26쪽; `papers/pdf/P29_weathernext-cyclones.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://www.nature.com/articles/s41586-026-10953-2) · [PDF](https://repository.library.noaa.gov/view/noaa/76404/noaa_76404_DS1.pdf) · [추가 1차 출처 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13581599/)

## 핵심과 방법

기상 상태와 태풍 자료를 함께 학습하는 WeatherNext Cyclones의 운영 예측 연구다.

대규모 ensemble과 태풍 관련 목표를 결합해 경로·강도·크기를 예측한다.

## 논문이 보고한 근거

2023–2025년 태풍 평가와 기존 운영 모델 비교를 보고한다. 공식 WeatherNext 저장소가 이 논문과 가중치 안내를 연결한다.

확인 위치: Nature 본문, Figure 1, Methods; NOAA 보관 PDF.

## 연구 아이디어와의 연결 — 정리자의 해석

FGN이 실제 복수 미래 경로를 생성하는 응용 맥락과 공개 구현의 출처를 확인하는 보완 자료다.

## 적용할 때의 제한

기상 응용의 성공이 ASR에 대한 효과의 근거는 아니다. WeatherNext 3의 아키텍처 논문과도 동일시하지 않는다.

## 코드와 모델

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `419b33da61eeee115d139518b25df34552d9768331cf9e0922c3f8d0af4abcbf`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
