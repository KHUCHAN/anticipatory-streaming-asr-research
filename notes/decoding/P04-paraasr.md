# P04. ParaASR: Multi-Token Prediction for Fast and Long-Context LLM-Based Speech Recognition

| 항목 | 내용 |
|---|---|
| 저자 | Lin, Qingjian; Li, Yuxin; Zhang, Haoyang; Chen, Jun; Huang, Yechang; Tian, Feng; Li, Xie; Zhang, Xiangyu Tony; Liu, Daijiao; Zhang, Yuxin; Gong, Jinglan; Zhao, Bo; Tian, Fei; Yang, Xuerui; Yu, Gang; Zhang, Xiangyu; Jiang, Daxin |
| 확인한 원문 날짜 | 2026-07-31 |
| 발표 정보 | arXiv preprint, 2026-07-31 |
| 분류 | 추측 디코딩과 경로 재사용 |
| 읽기 우선순위 | 1 — 먼저 읽기 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 14쪽; `papers/pdf/P04_paraasr.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2607.29279) · [PDF](https://arxiv.org/pdf/2607.29279v1)

## 핵심과 방법

LLM 기반 ASR에서 미래 토큰 여러 개를 동시에 제안하고 확인된 prefix를 채택하는 Multi-Token Prediction 시스템이다.

4B 디코더의 다섯 미래 토큰 branch로 한 번에 6토큰을 제안한다. 단계적 학습으로 기본 ASR과 MTP를 맞춘다.

## 논문이 보고한 근거

WenetSpeech meeting 평가의 MTP-5는 평균 5.0/6토큰을 채택한다. RTF 0.0053은 단일 H800, 단일 동시성, 30초 음성 100개 평가 조건이다.

확인 위치: PDF 8–11쪽, Tables 1–4, Section 4.

## 연구 아이디어와의 연결 — 정리자의 해석

채택 길이·후보 폭·rollback 비용의 균형을 정하는 자료다. acoustic verifier와 언어적으로 그럴듯한 후보를 구분할 필요를 보여준다.

## 적용할 때의 제한

음성 입력을 미리 받은 다중 토큰 디코딩과 미래 음성 도착 전 예측은 다르다. Table 1과 Table 4의 일부 데이터셋 수치가 일치하지 않아 정확도 비교는 표와 설정을 함께 기록해야 한다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `b6b1148d08ea3cc48d28d64d7c469646ca71bfc1f18fa129a27998004481decc`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
