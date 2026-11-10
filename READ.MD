# Anticipatory Streaming ASR — 학습 전 연구 준비

현재 음성·텍스트 문맥으로 다음 발화 경로를 미리 계산하고, 실제 음성이 들어오면 검증해 채택·수정·재사용하는 음성인식을 연구하기 위한 자료 모음이다.

**기준일: 2026-11-10 · 모델 아키텍처 미확정 · 모델 학습 미시작**

## 가장 먼저 볼 문서

1. [연구 목표와 방법론](docs/goal_and_methodology.md): 사용자의 의도, 검증 가설, 후보·확정 출력의 정의, 비교군, 측정 원칙.
2. [논문 지도](docs/literature_map.md): 미래 발화 예측 / decoder 가속 / 상태 압축 / FGN의 차이와 읽기 순서.
3. [전체 논문 목록 — 33편](papers/INDEX.md): 개별 한국어 요약, 원문·PDF·구현 링크.
4. [데이터·Mac 가이드](docs/datasets_and_mac.md): 선택한 100시간 데이터 범위, 준비 절차와 M5 Pro 24GB 조건.
5. [데이터 실측 검증 결과](data/reports/librispeech_inventory.md): 실제 다운로드·체크섬·발화 수·길이·split 검사.
6. [첨부 내용 사실 검토](docs/claim_audit.md): 성능 수치의 조건, 약어 혼동, 확인되지 않은 주장.

## 준비한 범위

- 논문 33편: 원문 PDF, 개별 노트, 주제별 분류, CSV·JSON 카탈로그, BibTeX.
- 데이터: LibriSpeech train-clean-100 및 dev-clean/dev-other/test-clean/test-other. **39,665발화 / 121.845시간**의 원음·전사·발화 manifest와 집계 검증 보고서.
- 확장 자료: GigaSpeech, Common Voice, AMI의 출처·규모·접근 상태와 관련 논문.
- 연구 정의: 목표, 방법의 단계, B0–B3 기능 비교군, 정확도·확정 지연·총 연산·메모리 측정 기준.

구체적인 encoder/decoder, layer 수, tokenizer, state 구조, alignment, 후보 폭과 학습 recipe는 이후 결정할 항목으로 남겼다. 원문에서 보고한 결과와 이 프로젝트의 가설은 문서에서 구분한다.

## 파일 구조

```text
.
├── README.md                          # GitHub 첫 화면
├── READ.MD                            # 전체 안내 및 파일 구조 — 이 문서
├── .gitignore                         # 원문·원음·개인 작업 파일·모델 파일 제외
├── docs/
│   ├── goal_and_methodology.md         # 연구 목표와 방법론의 기준 문서
│   ├── literature_map.md               # 선행연구 관계와 읽기 순서
│   ├── claim_audit.md                  # 첨부 대화의 주장·성능 조건 검토
│   ├── datasets_and_mac.md             # 데이터 및 M5 Pro 준비 조건
│   ├── resources.md                    # 공식 구현·모델·배포 출처
│   └── scope_and_provenance.md         # 수집 범위·서지·공개 기준
├── notes/                             # 논문별 한국어 노트 33개
│   ├── anticipatory/                   # P01–P02: 음성 도착 전 예측
│   ├── decoding/                       # P03–P08: draft·검증·후보 tree
│   ├── streaming/                      # P09–P15, P30: 정렬·인코더·Zipformer
│   ├── state-space/                    # P16–P20: S4·Mamba·ConMamba
│   ├── weather/                        # P21–P23, P29: FGN·기상 모델
│   ├── graph-asr/                      # P24–P26: contextual·relational ASR
│   ├── disambiguation/                 # P27–P28: 동명 FGN/FGNN
│   └── datasets/                       # P31–P33: corpus 원 논문
├── papers/
│   ├── INDEX.md                        # 주제별 전체 논문 표
│   └── pdf/
│       ├── README.md                   # PDF 수집·검증 안내
│       └── P01_ssr.pdf …               # 원문 33개; 로컬 전용, Git 제외
├── references/
│   ├── catalog.json                    # 논문 정리의 기준 데이터
│   ├── catalog.csv                     # 필터·정렬용 서지 표
│   └── references.bib                  # 인용용 BibTeX 33개
├── data/
│   ├── README.md                       # 원음·manifest 위치와 사용 원칙
│   ├── datasets.json                   # 4개 corpus 출처·접근 상태
│   ├── reports/
│   │   ├── librispeech_inventory.md    # 사람이 읽는 집계 검증 결과
│   │   └── librispeech_inventory.json  # 기계 판독용 집계 검증 결과
│   ├── raw/librispeech/                # 원본 데이터; Git 제외
│   │   ├── archives/                   # 공식 압축 파일·체크섬 검증
│   │   └── LibriSpeech/                # 선택한 5개 split의 FLAC·전사
│   └── processed/manifests/            # split별 utterance JSONL; Git 제외
└── scripts/
    ├── build_index.py                  # JSON → 논문 노트·목록·CSV·BibTeX
    ├── download_papers.py              # 원문 PDF 재수집 및 SHA-256 확인
    ├── validate_catalog.py             # 서지·로컬 링크·PDF 일관성 검사
    └── prepare_librispeech.py           # 100h+dev/test 수집·manifest·집계
```

## 다시 준비하거나 검증하기

논문 정리 스크립트는 Python 3.9+, 데이터 준비는 Python 3.12+, 다운로드는 `curl`을 사용한다. 학습 라이브러리는 설치하지 않는다.

```bash
# 공개 문서·서지 확인
python3 scripts/validate_catalog.py

# 논문 원문 받기: 전부 또는 일부
python3 scripts/download_papers.py
python3 scripts/download_papers.py --ids P01 P02 P03 P04 P21
python3 scripts/validate_catalog.py --pdfs

# 선택한 LibriSpeech 100h + dev/test 준비
python3 scripts/prepare_librispeech.py

# 이미 받은 데이터의 체크섬·manifest·집계 다시 만들기
python3 scripts/prepare_librispeech.py --inventory-only

# 논문 JSON을 수정한 뒤 파생 문서 재생성
python3 scripts/build_index.py
```

## 공개 저장소와 로컬 파일

Git에는 정리 문서, 원문 링크, 서지, 집계 보고서, 재현 스크립트를 올린다. PDF 원문, 음성, 전사 manifest, 모델 가중치와 원본 대화는 로컬에 보관한다. Git clone 후 위 스크립트로 공개 원문과 선택한 데이터를 다시 받을 수 있다.

전체 분야의 완전한 문헌조사나 학습 성능을 보장하는 자료집은 아니다. 두 첨부의 연구 방향을 모델 설계 전에 검토할 수 있도록 준비한 문헌·데이터·방법론 묶음이다. [출처 관리 기준](docs/scope_and_provenance.md)에 검토 범위와 한계를 기록했다.
