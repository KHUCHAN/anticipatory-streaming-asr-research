# P19. Assessing the Performance and Efficiency of Mamba ASR in Low-Resource Scenarios

| 항목 | 내용 |
|---|---|
| 저자 | Zevallos, Rodolfo; Cortada Garcia, Martí; Solito, Sarah; Mena, Carlos; Peiró-Lilja, Alex; Hernando, Javier |
| 확인한 원문 날짜 | 2025 |
| 발표 정보 | Interspeech 2025 |
| 분류 | 상태공간 모델과 ConMamba |
| 읽기 우선순위 | 2 — 보고 수치의 출처 |
| 원문 확인 범위 | 서지·초록·핵심 본문/실험 설정 확인 (전체 정독·재현 미수행) |
| PDF | 5쪽; `papers/pdf/P19_mamba-low-resource.pdf` |
| 검증일 | 2026-11-10 |

[원문](https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.html) · [PDF](https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.pdf)

## 핵심과 방법

9개 언어의 저자원 조건에서 Conformer와 ConMamba의 성능·효율을 비교한다.

학습 데이터 크기와 입력 문맥 길이를 바꾸어 두 구조를 비교한다.

## 논문이 보고한 근거

해당 설정에서 학습 시간 40–45%, 메모리 50% 절감과 추론 63–70% 가속을 보고한다.

확인 위치: PDF 1쪽 abstract, 본문 실험 설정·결과.

## 연구 아이디어와의 연결 — 정리자의 해석

첨부 대화의 “ConMamba 63–70%” 수치를 올바른 논문과 조건에 연결하는 근거다.

## 적용할 때의 제한

이 수치는 ConMamba 전체의 보편적 속도가 아니다. 언어·학습량·발화 길이·하드웨어가 달라지면 다시 측정해야 한다.

## 코드와 모델

- [https://github.com/xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)

논문에서 참고한 ConMamba 구현이다. 이 저자들이 새 학습 가중치를 공개했다는 뜻은 아니다.

## 원문 버전

PDF SHA-256: `7e075d3fb67015a3cdc6c803275194591f2356fa80d276aa5b7f902f348f2b3a`. 링크의 원문 버전과 이 해시를 기준으로 수집했다. 학습 및 저자 실험의 재현은 수행하지 않았다.

[전체 논문 목록](../../papers/INDEX.md) · [연구 지도](../../docs/literature_map.md)
