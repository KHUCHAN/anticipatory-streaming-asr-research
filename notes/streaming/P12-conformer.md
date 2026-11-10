# P12. Conformer: Convolution-augmented Transformer for Speech Recognition

| 항목 | 내용 |
|---|---|
| 저자 | Gulati, Anmol; Qin, James; Chiu, Chung-Cheng; Parmar, Niki; Zhang, Yu; Yu, Jiahui; Han, Wei; Wang, Shibo; Zhang, Zhengdong; Wu, Yonghui; Pang, Ruoming |
| 확인한 원문 날짜 | 2020-05-16 |
| 발표 정보 | Conformer 원 논문, arXiv 2020 |
| 분류 | ASR 정렬·인코더·스트리밍 기준선 |
| 읽기 우선순위 | 2 — 인코더 비교 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P12_conformer.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2005.08100) · [PDF](https://arxiv.org/pdf/2005.08100v1)

## 핵심과 방법

Convolution과 self-attention을 결합해 음성의 국소·전역 패턴을 모델링한다.

Conformer block에서 feed-forward, attention, convolution을 조합한다.

## 논문이 보고한 근거

LibriSpeech에서 기존 인코더 대비 높은 인식 성능을 보고한다.

확인 위치: Conformer block 및 LibriSpeech results.

## 연구 아이디어와의 연결 — 정리자의 해석

후보 생성 기법의 효과를 분리할 때 유지할 acoustic encoder의 비교 후보를 이해하는 자료다.

## 적용할 때의 제한

원형은 전체 문맥을 사용하는 설정을 포함한다. causal mask만 붙이면 모든 convolution·subsampling까지 인과적이 되는 것은 아니다.

## 코드와 모델

공식 구현 링크를 이번 검토에서 확정하지 않았다. 코드가 없다는 뜻은 아니다.

공개 checkpoint의 존재를 추정하지 않는다.

## 원문 버전

PDF SHA-256: `4d53a2d87f7b68b3b5a86d8e4301c5092ae61e260f9c76dab48af34faac9a81f`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
