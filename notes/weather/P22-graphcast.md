# P22. GraphCast: Learning skillful medium-range global weather forecasting

| 항목 | 내용 |
|---|---|
| 저자 | Lam, Remi; Sanchez-Gonzalez, Alvaro; Willson, Matthew; Wirnsberger, Peter; Fortunato, Meire; Alet, Ferran; Ravuri, Suman; Ewalds, Timo; Eaton-Rosen, Zach; Hu, Weihua; Merose, Alexander; Hoyer, Stephan; Holland, George; Vinyals, Oriol; Stott, Jacklynn; Pritzel, Alexander; Mohamed, Shakir; Battaglia, Peter |
| 확인한 원문 날짜 | 2022-12-24 |
| 발표 정보 | Science 2023; 확인 PDF는 arXiv 2022 최초 공개본의 개정 버전 |
| 분류 | FGN·GraphCast·WeatherNext |
| 읽기 우선순위 | 3 — 그래프 기상 배경 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 102쪽; `papers/pdf/P22_graphcast.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2212.12794) · [PDF](https://arxiv.org/pdf/2212.12794v2) · [추가 1차 출처 1](https://doi.org/10.1126/science.adi2336)

## 핵심과 방법

구면 mesh의 GNN으로 전 지구 중기 기상 상태를 예측하는 GraphCast 연구다.

grid–mesh encoder, mesh message passing, mesh–grid decoder를 사용해 다음 기상 상태를 autoregressive하게 계산한다.

## 논문이 보고한 근거

기상 재분석 자료와 운영 예보에 대한 정확도 및 효율을 평가한다.

확인 위치: graph encoder–processor–decoder architecture.

## 연구 아이디어와의 연결 — 정리자의 해석

Google 날씨 모델에서 graph 구조와 FGN의 함수적 확률 모델링이 서로 다른 구성요소라는 점을 정리한다.

## 적용할 때의 제한

날씨의 구면 mesh가 음성에 적절한 graph를 자동 제공하지 않는다. 음성 그래프는 노드·엣지·시간 인과성을 새로 정의해야 한다.

## 코드와 모델

- [https://github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `71ce5f63fb0b67955d38df35b766d9cd4ef27b4cd6188c7cbe4f9307b0e758c3`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
