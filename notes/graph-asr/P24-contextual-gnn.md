# P24. Graph Neural Networks for Contextual ASR with the Tree-Constrained Pointer Generator

| 항목 | 내용 |
|---|---|
| 저자 | Sun, Guangzhi; Zhang, Chao; Woodland, Phil |
| 확인한 원문 날짜 | 2023-05-30 |
| 발표 정보 | 확인 원문: arXiv preprint, 2023 |
| 분류 | 그래프 기반 음성인식 |
| 읽기 우선순위 | 2 — 후보 graph 참고 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 10쪽; `papers/pdf/P24_contextual-gnn.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2305.18824) · [PDF](https://arxiv.org/pdf/2305.18824v1)

## 핵심과 방법

contextual biasing 단어의 prefix tree를 GNN으로 인코딩하여 드문 단어·미등록 단어의 인식을 개선한다.

TCPGen에 tree-RNN, GCN, GraphSAGE 등의 노드 인코딩을 적용하여 하위 wordpiece 정보를 lookahead에 사용한다.

## 논문이 보고한 근거

LibriSpeech와 AMI에서 contextual ASR을 평가한다. rare-word WER와 전체 WER를 구분한다.

확인 위치: GNN prefix-tree encoding 및 contextual ASR results.

## 연구 아이디어와의 연결 — 정리자의 해석

복수 후보의 공통 prefix와 다음 branch 정보를 graph로 관리하는 방향에 참고한다.

## 적용할 때의 제한

그래프는 주로 biasing vocabulary의 구조이며 acoustic frame graph가 아니다. 발화가 오기 전 단어를 검증한 결과도 아니다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `f55d77c0478930645875d6bf0d4ef7848e8d0d8dd4ad5387a3f8782bfa5d70c7`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
