# P28. Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"

| 항목 | 내용 |
|---|---|
| 저자 | Grezes, Felix |
| 확인한 원문 날짜 | 2023-06-13 |
| 발표 정보 | CUNY PhD dissertation / arXiv, 2023 |
| 분류 | FGN/FGNN 동명 아키텍처 구분 |
| 읽기 우선순위 | 3 — 약어 혼동 방지 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 90쪽; `papers/pdf/P28_finite-gaussian.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://arxiv.org/abs/2306.07796) · [PDF](https://arxiv.org/pdf/2306.07796v1)

## 핵심과 방법

Finite Gaussian Neuron으로 입력에서 멀어진 영역의 과도한 confidence를 줄이는 박사학위 논문이다.

기존 신경망을 FGN으로 변환·재학습하며 adversarial robustness를 평가한다.

## 논문이 보고한 근거

MNIST와 SpeechCommands에서 실험하며, 더 복잡한 음성 명령 및 강한 공격에서 한계도 보고한다.

확인 위치: 학위논문 abstract 및 SpeechCommands experiments.

## 연구 아이디어와의 연결 — 정리자의 해석

처음 등장한 FGNN 설명의 출처를 보존하되 이 연구의 핵심 경로에서는 분리한다.

## 적용할 때의 제한

SpeechCommands 분류는 연속 발화를 문장으로 전사하는 ASR과 다르다. Functional Generative Network와 같은 모델이 아니다.

## 코드와 모델

- [https://github.com/grezesf/FGN---Research](https://github.com/grezesf/FGN---Research)

링크의 존재와 원문 연결을 확인했다. 설치·실행·가중치 다운로드·성능 재현은 수행하지 않았다.

## 원문 버전

PDF SHA-256: `0249184611054039ba37e6a5129175db141f75fb607a7bd04484f914aa25dd62`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
