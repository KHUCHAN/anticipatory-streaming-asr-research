# 수집 범위와 출처 관리

기준일은 2026-11-10이다. 두 첨부 연구 대화의 명시적 논문과 직접 연결되는 기반 연구를 33개 레코드로 정리했다. 전 세계 관련 논문을 빠짐없이 찾았다는 systematic review 주장은 하지 않는다.

수집 묶음은 음성 도착 전 예측 2편, 추측 디코딩 6편, 정렬·인코더·streaming 8편, 상태공간 5편, 기상 FGN 4편, graph ASR 3편, 동명 약어 2편, dataset 원 논문 3편이다.

arXiv, ISCA, PMLR, JMLR, 저자 페이지, Nature/NOAA 원문 및 프로젝트 공식 저장소를 사용했다. 검색은 SSR, speculative listening/ASR, token tree verification, ConMamba/LC-BiMamba, FGN/WeatherNext, contextual graph ASR, Zipformer 및 첨부의 데이터셋 이름을 중심으로 수행했다.

## 레코드 구성

각 논문에 저자·원문 날짜·발표 정보·분류·읽기 우선순위·방법·근거·해석·제한·원문 위치·구현 확인 상태를 기록한다. 모든 PDF는 다운로드 후 파서로 읽을 수 있는지 확인하고 쪽수와 SHA-256을 저장했다. 핵심 성능 주장에는 원문 실험 설정과 표·절 위치를 확인했다. 모든 논문에 대한 실험 재현이나 모든 페이지의 정밀 독해를 수행한 것은 아니다.

arXiv PDF는 검토한 버전으로 고정했다. 정식 출판본과 preprint의 제목·연도·조건이 다를 수 있어 카탈로그에 구분했다. BibTeX는 확인한 원문 또는 검증한 학회 서지를 기준으로 만들었다. arXiv DOI는 동료심사 완료의 증거가 아니다.

## 공개 저장소의 내용

공개 대상은 작성한 연구 정리, bibliographic metadata, 원문 링크, 집계 데이터 검증 보고서와 재현 스크립트다. 원본 대화, 개인 절대 경로, 계정 정보, raw 음성, 발화별 전사 manifest, 제3자 PDF, 모델 가중치는 Git에서 제외한다.

제3자 원문의 저작권·이용 조건은 각 원 출처를 따른다. 공개 링크가 있다는 사실만으로 모든 PDF의 재배포 조건이 같다고 간주하지 않았다. 원문은 로컬에 수집하고, 공개 저장소에서는 직접 링크와 체크섬으로 연결했다.

## 수정·재생성

```bash
python3 scripts/build_index.py
python3 scripts/validate_catalog.py
python3 scripts/validate_catalog.py --pdfs
```

`references/catalog.json`이 논문 정리의 기준 파일이다. 개별 노트·CSV·BibTeX·전체 목록은 이 파일에서 재생성한다. 목표·방법론과 데이터 준비 보고서는 각각의 작성 문서 및 실행 결과로 관리한다.
