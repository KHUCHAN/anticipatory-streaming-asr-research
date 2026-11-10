# 구현·모델·데이터의 공식 출처

확인일: 2026-11-10. 링크를 읽고 연결 관계를 확인했으며 설치·학습·모델 성능 재현은 하지 않았다.

| 출처 | 용도 | 확인한 상태 |
|---|---|---|
| [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext) | WN2/FGN 및 이전 GraphCast·GenCast | README가 모델 코드·가중치 bucket·demo를 안내 |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | Mamba | 공식 구현; Mac 커널 성능은 별도 확인 |
| [state-spaces/s4](https://github.com/state-spaces/s4) | S4 | 상태공간 구현 |
| [xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR) | Speech Slytherin / ConMamba | 원문이 직접 연결한 구현 |
| [k2-fsa/icefall](https://github.com/k2-fsa/icefall) | Zipformer·ASR recipe | 논문이 공개한 구현 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | streaming 모델 배포·추론 | 20M ONNX 모델 안내 확인 |
| [NVIDIA NeMo의 현재 Speech 저장소](https://github.com/NVIDIA-NeMo/Speech) | FastConformer·transducer 계열 | 원 논문의 NVIDIA/NeMo 링크는 현재 이 주소로 이동 |
| [openai/whisper](https://github.com/openai/whisper) | 외부 정확도·디코딩 참조 | 공개 코드·모델 안내 |
| [Factor Graph NN](https://github.com/zzhang1987/Factor-Graph-Neural-Network) | 동명 FGNN 구분 | JMLR 연결 구현 |
| [Finite Gaussian FGN](https://github.com/grezesf/FGN---Research) | 동명 FGN 구분 | 학위논문 연결 구현 |

SpecInfer의 원문 링크 `flexflow/FlexFlow`는 현재 `flexflow/flexflow-train`으로 이동한다. 따라서 원래 serving artifact가 같은 상태로 보존되었다고 가정하지 않는다. DGP 원문이 연결한 `GlenHGHUANG/Deep_graph_random_process`는 이번 검사에서 HTTP 404였다. 해당 논문의 원문은 PMLR에서 확보했다.

SSR·SpecASR·ParaASR·Speculative Listening·LC-BiMamba에 대해 공식 구현·체크포인트를 확인하지 못한 항목은 논문 노트에 미확인으로 남겼다. 논문의 존재와 코드·사전학습 가중치의 공개는 서로 다른 상태다.

데이터 원음과 모델 가중치를 다른 공개 저장소에서 임의로 찾아 대체하지 않았다. 재수집 가능한 논문 PDF 주소는 [카탈로그](../references/catalog.json), 데이터 원본·접근 상태는 [registry](../data/datasets.json)를 사용한다.
