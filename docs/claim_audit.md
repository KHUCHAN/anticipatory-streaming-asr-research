# 첨부 대화의 주장 검토

검토일: 2026-11-10. 최초 설명을 그대로 연구 사실로 옮기지 않고 1차 출처를 확인했다.

| 원래 설명의 쟁점 | 확인 결과와 문서화 방식 | 근거 |
|---|---|---|
| FGNN은 Google 날씨 모델인가? | Google의 FGN은 Functional Generative Network다. Factor Graph NN 및 Finite Gaussian NN은 별도 연구다. | [P21](../notes/weather/P21-fgn.md), [P27](../notes/disambiguation/P27-factor-graph.md), [P28](../notes/disambiguation/P28-finite-gaussian.md) |
| Chain rule이라서 Transformer보다 빠른가? | chain rule 자체와 과거 문맥을 제한하는 모델링 가정을 구분한다. Transformer도 autoregressive 분해를 사용한다. | [P21 Eq. (1)](https://arxiv.org/abs/2506.10772), [P05](https://proceedings.mlr.press/v202/leviathan23a.html) |
| FGN의 상태는 32차원인가? | 32차원은 주입하는 확률 noise다. 모델 입력인 두 대기 상태는 고차원 배열이다. | [P21 원문](https://arxiv.org/abs/2506.10772) |
| 하나의 forward로 모든 미래 후보·시간을 생성하나? | 해당 기상 예측 단계·ensemble member의 생성 비용을 뜻한다. rollout과 ensemble 규모의 추가 계산은 남는다. | [P21](../notes/weather/P21-fgn.md) |
| FGN이 8배 빠르니 ASR도 빠른가? | 기상 GenCast의 diffusion sampling과 비교한 결과다. Whisper·ASR 속도의 근거가 아니다. | [P21 Section 3.3](https://arxiv.org/abs/2506.10772) |
| ConMamba 2025의 속도 수치 | 63–70% 수치의 정확한 출처는 저자원 ASR 비교 논문이다. 원형 ConMamba 연구와 구분했다. | [P19](https://www.isca-archive.org/interspeech_2025/zevallos25_interspeech.html), [P18](https://arxiv.org/abs/2407.09732) |
| SpecASR 3.04–3.79배 | Whisper 경로와 Llama/Vicuna simulated target을 쓰는 실험 설정을 반드시 함께 적는다. | [P03 PDF 5쪽](https://arxiv.org/pdf/2507.18181v2) |
| ParaASR의 평균 5개 채택 | WenetSpeech meeting의 특정 MTP 설정이며 H800 RTF와 함께 조건을 기록했다. 본문 표 사이 일부 수치 불일치도 남겼다. | [P04 Tables 1–4](https://arxiv.org/abs/2607.29279) |
| Speculative Listening은 미래 ASR 단어를 잘 맞히나? | full-duplex 출력 사전 계산 연구다. silence와 실제 입력에서 출력이 같은 비율은 다음 단어의 정확도가 아니다. | [P02](https://arxiv.org/abs/2611.06982) |
| TDT는 음성 처리를 통째로 건너뛰나? | duration에 따라 transducer decoding 시간축을 진행한다. acoustic encoder가 항상 불필요해지는 것은 아니다. | [P11](https://proceedings.mlr.press/v202/xu23g.html) |
| Mamba는 무조건 빠른 streaming ASR인가? | 길이·연산 커널·bidirectional/causal 설정에 따라 달라진다. CPU/MPS 성능은 별도 측정 대상이다. | [P18](https://arxiv.org/abs/2407.09732), [P20](https://www.isca-archive.org/interspeech_2025/moriya25_interspeech.html) |
| 단순 Bayesian 곱셈으로 LM+ASR을 합칠 수 있나? | likelihood와 posterior를 구분해야 하며 언어 prior를 중복 반영할 수 있다. 결합·검증 규칙은 미확정이다. | [RNN-T formulation](https://arxiv.org/abs/1211.3711); 본 프로젝트의 방법론 검토 |
| PyTorch 2.14 MPS CTC 지원 | 공식 발표에서 forward/backward 지원을 확인했다. 현재 설치된 버전은 2.12.0이며 업그레이드는 하지 않았다. | [공식 2.14 발표](https://pytorch.org/blog/pytorch-2-14-release-blog/) |
| Common Voice 42,593시간·295언어 | v27 scripted 전체 녹음 통계다. 영어 validated 학습 시간이나 자발 대화 데이터의 규모가 아니다. | [Mozilla v27 공지](https://discourse.mozilla.org/t/common-voice-scripted-speech-v27-0-and-spontaneous-speech-v5-0-released/149490) |
| 100시간·10–20M이면 정확도/학습 시간이 보장되나? | 초기 규모 제안일 뿐이다. 모델 크기와 학습 시간은 실제 구조·연산·메모리 측정 전에는 확정하지 않는다. | [준비 범위와 결정 항목](goal_and_methodology.md) |
| WeatherNext 3의 세부 구조가 FGN 논문에 있나? | 공식 제품 페이지의 존재는 확인했다. P21은 WN2 기술보고서이며 WN3의 공개 구조를 동일하다고 단정하지 않는다. | [공식 제품 페이지](https://deepmind.google/science/weathernext/), [WN2 공식 저장소](https://github.com/google-deepmind/weathernext) |

고정 크기 상태, memory bank, KV cache, 후보 tree cache는 구분해서 측정한다. encoder의 상태가 제한되어도 다른 구성요소의 메모리가 무한히 증가하면 시스템 전체의 제한 메모리 목표는 충족되지 않는다.
