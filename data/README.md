# 실험 데이터 위치

사용자가 선택한 LibriSpeech 100시간 학습 세트와 네 dev/test 세트를 원음·전사·발화 manifest 수준까지 준비했다. 최종 실측 상태는 [검증 보고서](reports/librispeech_inventory.md), 데이터 선정 근거와 접근 조건은 [데이터·Mac 가이드](../docs/datasets_and_mac.md)에 정리했다.

```text
data/
├── README.md
├── datasets.json
├── reports/librispeech_inventory.json
├── raw/librispeech/                 # 압축 파일과 원본 FLAC·전사; Git 제외
└── processed/manifests/             # split별 발화 JSONL; Git 제외
```

데이터를 준비할 때는 corpus 버전, split, speaker/session ID, 원본 utterance ID, 정렬 모델 및 버전, 정규화 규칙, 다운로드 날짜를 별도 manifest에 남긴다. 먼저 원본 발화 단위로 train/dev/test를 나눈 뒤 청크를 생성한다.

준비한 manifest는 원본 발화의 상대 경로·전사·sample 수·길이·화자·chapter를 기록한다. 청크화, 토큰화, acoustic feature, alignment와 모델 학습은 아직 수행하지 않았다.
