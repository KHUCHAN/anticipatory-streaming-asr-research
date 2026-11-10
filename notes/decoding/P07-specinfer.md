# P07. SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification

| 항목 | 내용 |
|---|---|
| 저자 | Miao, Xupeng; Oliaro, Gabriele; Zhang, Zhihao; Cheng, Xinhao; Wang, Zeyu; Zhang, Zhengxin; Wong, Rae Ying Yee; Zhu, Alan; Yang, Lijie; Shi, Xiaoxiang; Shi, Chunan; Chen, Zhuoming; Arfeen, Daiyaan; Abhyankar, Reyna; Jia, Zhihao |
| 확인한 원문 날짜 | 2023-05-16 |
| 발표 정보 | arXiv 2023; SpecInfer 원문 버전 |
| 분류 | 추측 디코딩과 경로 재사용 |
| 읽기 우선순위 | 2 — 분기 구현 참고 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 18쪽; `papers/pdf/P07_specinfer.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2305.09781) · [PDF](https://arxiv.org/pdf/2305.09781v4)

## 핵심과 방법

여러 후보 토큰 경로를 tree로 묶고 target LLM이 한꺼번에 검증하는 시스템이다.

token tree와 tree attention mask로 서로 다른 prefix의 검증을 병렬화한다.

## 논문이 보고한 근거

분산 추론과 offloading 환경에서 기존 serving 시스템보다 낮은 지연을 보고한다.

확인 위치: token-tree verification 설계 및 serving evaluation.

## 연구 아이디어와의 연결 — 정리자의 해석

사용자가 말한 “가능성 높은 다음 단어 경로로 이동”을 tree·분기·공통 prefix 재사용 문제로 구체화할 때 참고한다.

## 적용할 때의 제한

LLM serving 결과이며 음향 검증이나 스트리밍 ASR의 효율을 직접 입증하지 않는다. 분기 수가 늘면 메모리와 검증 비용도 늘어난다.

## 코드와 모델

- [https://github.com/flexflow/FlexFlow](https://github.com/flexflow/FlexFlow)

원문 링크는 현재 flexflow/flexflow-train으로 redirect된다. 논문의 serving artifact가 현재 같은 저장소에 보존되는지와 실행 가능성은 확인하지 않았다.

## 원문 버전

PDF SHA-256: `38764bfe741d39c9a309d1b715e8dc2cea6ea6c965337752b3562b1785afd9f1`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
