# LibriSpeech 데이터 준비 검증 결과

검증 실행 시각: 2026-11-10T11:01:56+00:00 (UTC). 원본: [OpenSLR SLR12](https://www.openslr.org/12), CC BY 4.0.

| Split | 발화 수 | 화자 수 | 음성 길이 (시간) |
|---|---:|---:|---:|
| train-clean-100 | 28,539 | 251 | 100.591 |
| dev-clean | 2,703 | 40 | 5.388 |
| dev-other | 2,864 | 33 | 5.121 |
| test-clean | 2,620 | 40 | 5.403 |
| test-other | 2,939 | 33 | 5.342 |

총 **39,665발화 / 121.845시간**. 압축 원본은 7.715 GB다.

- 선택한 5개 압축 파일의 공식 MD5와 수집 파일이 일치한다.
- 모든 FLAC의 STREAMINFO를 읽고 16 kHz·mono·16-bit, sample 수와 길이를 확인했다.
- 모든 음성과 전사 ID가 일대일 대응하며 전사 중복·빈 전사가 없다.
- split 간 utterance ID 중복과 화자 중복이 없다.
- 전체 waveform의 디코딩 검사를 수행한 것은 아니다. 원본 payload는 압축 파일 체크섬으로 확인했다.

원본: `data/raw/librispeech/LibriSpeech/`. 발화 manifest: `data/processed/manifests/`.

현재 단계는 원음·전사·발화 메타데이터 준비다. 모델 구조, tokenizer, acoustic feature, alignment 및 학습은 수행하지 않았다.

[기계 판독용 전체 보고서](librispeech_inventory.json) · [데이터 가이드](../../docs/datasets_and_mac.md)
