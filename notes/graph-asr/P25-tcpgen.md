# P25. Tree-constrained Pointer Generator with Graph Neural Network Encodings for Contextual Speech Recognition

| 항목 | 내용 |
|---|---|
| 저자 | Sun, Guangzhi; Zhang, Chao; Woodland, Phil |
| 확인한 원문 날짜 | 2022 |
| 발표 정보 | Interspeech 2022 |
| 분류 | 그래프 기반 음성인식 |
| 읽기 우선순위 | 2 — contextual graph 원형 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P25_tcpgen.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://www.isca-archive.org/interspeech_2022/sun22_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2022/sun22_interspeech.pdf)

## 핵심과 방법

tree-constrained pointer generator에 GNN encoding을 결합한 contextual ASR 연구다.

biasing word prefix tree의 branch 정보를 노드 표현에 반영한다.

## 논문이 보고한 근거

LibriSpeech의 biasing 과제와 AMI에서 문맥 단어 인식을 평가한다.

확인 위치: method 및 contextual biasing experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

후보 tree의 구조적 표현에 대한 선행연구이며 P24의 앞선 연구로 함께 읽을 수 있다.

## 적용할 때의 제한

문맥 단어 목록을 안다는 조건을 미래 발화 정답을 안다는 조건으로 바꿀 수 없다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `0aa25d8719f22aeb91c726dbf9bc5d8dafd0feec69dcd5b46ad114cbb68215ec`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
