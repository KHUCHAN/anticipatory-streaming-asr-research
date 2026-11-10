# 데이터 준비와 Mac 실행 조건

## 준비 범위

사용자가 선택한 범위는 **LibriSpeech train-clean-100과 dev/test 4개 split**이다. 원본 FLAC과 전사, 발화별 manifest까지 준비한다. 데이터 증강, tokenizer, feature cache, word alignment와 학습 recipe는 아키텍처 결정 이후 단계다.

| 데이터 | 공식 규모·특성 | 이 작업에서의 역할·상태 |
|---|---|---|
| LibriSpeech | train 100/360/500시간 분할, 16 kHz 영어 낭독 | 100시간 train + dev-clean/dev-other/test-clean/test-other 실제 준비; 최종 결과는 [inventory](../data/reports/librispeech_inventory.md) |
| GigaSpeech | 10/250/1,000/2,500/10,000시간 subset, 영어 다중 도메인 | 확장용 문헌·배포 경로 준비; 신청·접근 동의·원음 다운로드 미수행 |
| Common Voice | v27 scripted 전체 언어 42,593시간, 295언어 | 언어·validated subset를 정한 뒤 검토할 확장용; 원음 다운로드 미수행 |
| AMI | 약 100시간 회의, 여러 마이크·화자와 시간 annotation | 대화·겹침·억양 평가 후보; 이번 100시간 선택 범위에는 포함하지 않음 |

출처: [OpenSLR](https://www.openslr.org/12), [GigaSpeech 공식 배포](https://github.com/SpeechColab/GigaSpeech), [Mozilla v27 발표](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490), [AMI 개요](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml).

LibriSpeech 공식 표의 train-clean-100 압축 크기는 6.3G다. 본 작업의 총 다운로드·발화 수·길이는 inventory의 실측값을 사용한다. 압축 크기와 압축 해제 후 디스크 사용량을 구분한다. 전체 960시간 corpus의 원본 MP3·도서 텍스트까지 자동으로 받지 않는다.

## 준비 스크립트

Python 3.12+와 `curl`을 사용한다. 파이썬 학습 라이브러리를 설치하지 않는다.

```bash
python3 scripts/prepare_librispeech.py
python3 scripts/prepare_librispeech.py --inventory-only
```

다운로드는 HTTPS byte range를 확인하고 이어받을 수 있게 구성했다. 공식 `md5sum.txt`와 검토한 체크섬을 모두 비교한 뒤 압축을 푼다. FLAC 헤더에서 sample rate, channel, sample count, bit depth를 읽고 전사–음성 ID의 완전 대응 및 split 간 화자 중복을 검사한다. 전체 waveform을 디코딩하는 검사는 아니다.

발화별 manifest의 필드는 `dataset`, `split`, `utterance_id`, `speaker_id`, `chapter_id`, `audio`, `text`, `sample_rate`, `channels`, `num_samples`, `duration_seconds`다. 상대 경로를 쓰며 원본 uppercase 전사는 그대로 보관한다. raw audio와 manifest는 Git에서 제외하고 집계 보고서와 재현 스크립트만 push한다.

## M5 Pro 24GB에서 확인한 것

2026-11-10 로컬 읽기 검사에서 Apple M5 Pro, 24 GiB 통합 메모리, arm64, PyTorch 2.12.0, MPS built/available을 확인했다. 하드웨어 식별번호나 계정 정보는 기록하지 않았다. 패키지 설치·업그레이드 및 모델 실행은 하지 않았다.

[PyTorch 2.14 공식 발표](https://pytorch.org/blog/pytorch-2-14-release-blog/)는 MPS의 CTC Loss forward/backward 지원을 설명한다. 실제 채택 전에는 별도 환경에서 작은 연산 검증과 메모리·시간 측정이 필요하다. 설치된 2.12.0에서 같은 지원을 가정하지 않는다. CUDA 전용 fused scan, transducer loss, decoding extension의 지원도 별도 확인 대상이다.

24 GiB는 OS와 CPU/GPU가 공유한다. 파라미터만으로 학습 가능성이나 소요 시간을 확정하지 않는다. 차후 선택한 구조에서 입력 길이, activation, optimizer state, 정렬 loss와 fallback의 비용을 측정해야 한다.

## 외부 비교 모델

[Zipformer 20M streaming ONNX 공식 안내](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)와 [모델 카드](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17)는 공개되어 있다. 이는 선택 가능한 외부 참조이며 본 작업에서는 가중치를 받거나 inference를 실행하지 않았다.

같은 100시간으로 학습한 모델 간 비교와, 외부 pretrained 모델 비교를 별도 표로 나눈다. ONNX CPU와 PyTorch MPS의 처리시간 차이는 실행 backend까지 포함하므로 알고리즘 효과만의 수치로 주장하지 않는다.

[연구 목표와 방법론](goal_and_methodology.md) · [데이터 registry](../data/datasets.json)
