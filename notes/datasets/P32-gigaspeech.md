# P32. GigaSpeech: An Evolving, Multi-domain ASR Corpus with 10,000 Hours of Transcribed Audio

| 항목 | 내용 |
|---|---|
| 저자 | Chen, Guoguo; Chai, Shuzhou; Wang, Guanbo; Du, Jiayu; Zhang, Wei-Qiang; Weng, Chao; Su, Dan; Povey, Daniel; Trmal, Jan; Zhang, Junbo; Jin, Mingjie; Khudanpur, Sanjeev; Watanabe, Shinji; Zhao, Shuaijiang; Zou, Wei; Li, Xiangang; Yao, Xuchen; Wang, Yongqing; Wang, Yujun; You, Zhao; Yan, Zhiyong |
| 확인한 원문 날짜 | 2021-06-13 |
| 발표 정보 | GigaSpeech 원 논문, arXiv 2021 |
| 분류 | 학습 데이터셋 원 논문 |
| 읽기 우선순위 | 2 — 확장 데이터 이해 |
| 원문 확인 범위 | 서지·초록 확인; PDF 전문 수집·파싱 검증 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P32_gigaspeech.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2106.06909) · [PDF](https://arxiv.org/pdf/2106.06909v1)

## 핵심과 방법

오디오북·팟캐스트·YouTube를 포함하는 10,000시간 다중 도메인 영어 ASR corpus 연구다.

수집·전사·정렬·품질 필터링과 10/250/1,000/2,500/10,000시간의 하위 집합을 제공한다.

## 논문이 보고한 근거

다양한 ASR baseline을 제공하며 여러 도메인 확장을 평가한다.

확인 위치: corpus description 및 공식 repository Download 안내.

## 연구 아이디어와의 연결 — 정리자의 해석

LibriSpeech에서 확인한 방법이 더 자연스럽고 다양한 음성에서도 유지되는지 확인할 후속 후보 데이터다.

## 적용할 때의 제한

공식 배포는 신청 절차를 안내한다. 이 프로젝트에서는 신청·계약 동의·원음 다운로드를 수행하지 않았고 접근 조건을 registry에 기록한다.

## 코드와 모델

- [https://github.com/SpeechColab/GigaSpeech](https://github.com/SpeechColab/GigaSpeech)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `cd21790d69372497ce28b01e85d16d41889105f80d3832ed3c7a08d8c06a9541`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
