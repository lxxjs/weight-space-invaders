# MinAtar 가중치 직접 작성과 Space Invaders 확장

이 작업은 **Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab**의 [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) 실험 5와 [저자 공개 로그](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch)를 출발점으로 한다. 원문은 Breakout·Seaquest·Asterix에서 게임을 플레이하거나 학습하지 않은 Astra가 규칙과 소스 코드를 읽고 `600 → 32 → 32 → 6` 정책망의 가중치를 한 번에 작성할 수 있는지 확인했다. 이 저장소는 그 검증 경로를 독립적으로 구현하고 **원문에 없는 Space Invaders**로 확장한다. 이 폴더의 실행 코드는 새로 작성했으며 저자의 평가 하네스를 그대로 복사한 것은 아니다.

**MinAtar는 Atari 게임의 규칙을 10×10 이진 격자로 단순화한 강화학습 실험 환경이다.**
*MinAtar is a reinforcement learning testbed that simplifies Atari-inspired games to 10×10 binary grids.* [원 논문](https://arxiv.org/abs/1903.03176)

## 필요한 자료

| 자료 | 출처 | 용도 |
| --- | --- | --- |
| MinAtar **1.0.15** | `requirements.txt`로 설치 | Breakout, Seaquest, Asterix 게임 환경. 평가 중 화면 상태와 보상이 생성된다. |
| 게임 규칙·소스 코드·입출력 명세가 들어 있는 프롬프트 | [저자 공개 로그](https://github.com/RikiyaT/lm_understand-logs), 커밋 `4d09ba6` | Astra에게 주는 입력. 게임마다 `runs/<game>/message.txt` 하나. |
| 저자 공개 답변과 `hand.npz` | 같은 공개 로그 | 편집문 적용기와 평가기의 검증용. 새 실험에서는 사용하지 않는다. |
| PPO 비교 점수 | 같은 공개 로그의 `eval/minatar_eval.json` | 발표 결과와 비교할 때만 쓴다. PPO 체크포인트는 공개되지 않았다. |

**별도의 학습 데이터셋은 필요 없다.** 이 실험에서 Astra는 게임 장면이나 정답 행동의 예시를 받지 않는다. 게임을 실행하며 나오는 관측·보상은 가중치 작성이 끝난 **뒤의 평가**에만 사용한다. 새 정책을 평가하기 전에는 그 정책으로 게임을 플레이하거나 결과를 보고 수정하지 않는다.

## 설치 및 공개 자료 준비

저장소 루트에서 실행한다. Python 3.12와 `uv`가 설치되어 있다고 가정한다.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r experiments/weight_space_exp5/requirements.txt
sh experiments/weight_space_exp5/get_reference.sh
codex login status
```

`get_reference.sh`는 저자 저장소의 지정 커밋에서 **5번 실험 폴더만** `data/upstream/`에 받는다. `.venv/`, `data/`, `results/`는 Git에서 제외한다. Codex CLI는 ChatGPT 계정 로그인 상태를 재사용하므로 API 키는 필요하지 않다. Astra 사용량은 계정의 Codex 한도에 반영된다.

## 1단계: 평가 경로 검증

저자 답변의 마지막 `edits` 블록을 0으로 초기화된 망에 적용한다.

```sh
.venv/bin/python experiments/weight_space_exp5/exp5.py apply \
  --answer experiments/weight_space_exp5/data/upstream/exp5_write_from_scratch/runs/breakout/reply/answer.md \
  --output experiments/weight_space_exp5/results/breakout/rebuilt.npz

.venv/bin/python experiments/weight_space_exp5/exp5.py evaluate \
  --game breakout \
  --weights experiments/weight_space_exp5/results/breakout/rebuilt.npz \
  --episodes 10 \
  --output experiments/weight_space_exp5/results/breakout/check.json
```

검증이 끝나면 `--episodes 1000`으로 원문과 같은 게임 수를 사용한다. 첫 실행은 라이브러리의 글꼴 캐시 때문에 더 오래 걸릴 수 있다. 이 평가기는 에피소드마다 고정 시드를 사용하지만 저자가 사용한 난수열은 공개되지 않아 **정확히 같은 게임별 점수는 보장하지 않는다**. 표본 수가 10처럼 작으면 평균 점수의 변동도 크다.

## 2단계: 새 정책 작성

게임마다 별개의 Codex 실행을 사용한다. `--dry-run`은 명령만 보여 주며 모델을 호출하지 않는다.

```sh
.venv/bin/python experiments/weight_space_exp5/run_agent.py breakout --dry-run
.venv/bin/python experiments/weight_space_exp5/run_agent.py breakout

.venv/bin/python experiments/weight_space_exp5/exp5.py apply \
  --answer experiments/weight_space_exp5/results/breakout/answer.md \
  --output experiments/weight_space_exp5/results/breakout/new.npz

.venv/bin/python experiments/weight_space_exp5/exp5.py evaluate \
  --game breakout \
  --weights experiments/weight_space_exp5/results/breakout/new.npz \
  --episodes 1000 \
  --workers 4 \
  --output experiments/weight_space_exp5/results/breakout/new_eval.json
```

`run_agent.py`는 저자의 원본 프롬프트, `gpt-6-astra`와 `xhigh` 추론 수준, 도구 비활성화 설정을 사용한다. 한 번의 `codex exec`만 호출하며 답변·이벤트·오류 로그를 보존한다. `exp5.py apply`는 마지막 `edits` 블록의 숫자 대입만 허용하고 해석할 수 없는 문장이 있으면 파일을 만들지 않고 실패한다. **새 정책 실행은 아직 하지 않았다.**

Seaquest와 Asterix는 위 명령의 게임 이름을 각각 `seaquest`, `asterix`로 바꿔 독립 실행한다. 세 게임 모두 행동 6개를 사용하지만 원래 MinAtar 화면 채널에서 공통 6채널 입력으로 바꾸는 매핑은 게임별로 다르며, `exp5.py`에 명시되어 있다.

## 평가 규칙과 한계

- 정책망: `600 → 32 → 32 → 6`, 두 은닉층에 `tanh`, 모든 가중치와 편향은 0에서 시작한다.
- 행동: 출력 로짓의 softmax 분포에서 표본 추출한다. MinAtar의 sticky action 확률은 0.1이다.
- 평가: 게임당 1,000회, 에피소드당 최대 10,000스텝, 평균 보상을 보고한다.
- `--workers 4`는 독립 에피소드를 CPU 프로세스 네 개에 나눠 실행한다. 에피소드별 시드를 고정하므로 작업자 수를 바꿔도 같은 정책의 점수열은 같다.
- 비교: 공개 PPO 점수는 참고값이다. PPO를 직접 학습시키지 않아도 가중치 직접 작성 실험을 수행할 수 있다. PPO 기준선까지 독립 재현하려면 별도의 1억·5억 환경 스텝 학습이 필요하다.
- 저자 공개 정책의 평가 점수와 이번 로컬 점수는 난수열 및 하네스 차이로 달라질 수 있다. 결과에는 시드, 게임 수, 정책 파일 해시, 개별 점수를 JSON으로 남긴다.

세 저자 답변의 편집문을 이 적용기로 다시 읽었을 때 92/122/55개 문장이 모두 적용되었고, 생성된 여섯 가중치 텐서는 공개 `hand.npz`와 원소 단위로 일치했다. 세 게임 모두 소수 에피소드 실행까지 확인했다. 이것은 **설정과 코드 경로의 확인**이며, 1,000회 평가나 새 Astra 정책의 성능 결과는 아니다.

## 새 게임 확장: Space Invaders

MinAtar에는 원문 실험 5의 세 게임 외에 Freeway와 Space Invaders도 있다. Space Invaders는 **원래 관측 자체가 10×10×6**이라 같은 `600 → 32 → 32 → 6` 망을 정보 손실 없이 쓸 수 있다. Freeway는 원래 7채널이라 동일한 600입력 구조를 유지하려면 채널을 합치거나 구조를 바꿔야 한다. 여기서는 Space Invaders만 확장했다. **저자 논문에 없는 새 실험**이며, 동일 조건의 공개 Astra 점수나 PPO 비교값은 없다.

`prepare_space_invaders.py`는 설치된 MinAtar 1.0.15의 정확한 게임 소스를 읽어 새 프롬프트에 삽입하고 해시를 남긴다. 입력 6채널은 `cannon`, `alien`, `alien_left`, `alien_right`, `friendly_bullet`, `enemy_bullet` 순서로 사용한다. 원문 세 게임의 공통 의미 채널로 바꾸지 않으며, 이것은 명시적인 프로토콜 차이다. 행동 6개, sticky action 0.1, 1,000회 평가, 최대 10,000스텝은 유지한다.

```sh
.venv/bin/python experiments/weight_space_exp5/prepare_space_invaders.py
.venv/bin/python experiments/weight_space_exp5/run_agent.py space_invaders --dry-run

# 모델 호출 없이 평가 경로와 균등 무작위 행동 기준선을 확인한다.
.venv/bin/python experiments/weight_space_exp5/exp5.py init-zero \
  --output experiments/weight_space_exp5/results/space_invaders/zero.npz
.venv/bin/python experiments/weight_space_exp5/exp5.py evaluate \
  --game space_invaders \
  --weights experiments/weight_space_exp5/results/space_invaders/zero.npz \
  --episodes 1000 \
  --workers 4 \
  --output experiments/weight_space_exp5/results/space_invaders/zero_eval_1000.json
```

새 가중치를 작성할 때는 `--dry-run`을 빼고 `run_agent.py space_invaders`를 실행한다. 이후 `exp5.py apply --answer .../results/space_invaders/answer.md --output .../new.npz`로 정책을 만들고 `exp5.py evaluate`로 평가한다. 현재 로컬의 **균등 무작위 행동 점수**는 시드 `20261125`, 1,000회 평가에서 평균 **2.702점**, 표준오차 **0.071점**이다. 이는 학습된 강화학습 정책과의 비교를 대신하지 않는다. 새 Astra 호출은 아직 실행하지 않았다.

### PPO 비교 기준

[Anokin 등, ICLR 2025, 표 10](https://proceedings.iclr.cc/paper_files/paper/2025/file/770cabd044c4eacb6dc5924d9a686dce-Paper-Conference.pdf)은 MinAtar SpaceInv-v0 PPO의 1,000만 샘플 학습 후 평균 점수를 **297.49 ± 69.86**(3개 시드의 표준오차)으로 보고한다. 같은 논문의 표 4는 행동 반복 확률 0.25에서 **91.38 ± 12.33**을 보고한다. 두 값 모두 참고용이다. 현재 실험은 행동 반복 확률 0.1, `600 → 32 → 32 → 6` tanh 망, 1,000회 평가, 1억·5억 스텝 기준이므로 숫자를 같은 표에서 우열 비교하면 안 된다. 원문 저자의 [실험 5 공개 로그](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/README.md)에도 Space Invaders의 동일 조건 PPO 값이나 체크포인트는 없다. 직접 비교하려면 같은 관측·망·학습 예산·환경·평가 방식으로 Space Invaders PPO를 학습하고 이 평가기로 측정해야 한다.

### 로컬 PPO 학습과 속도 측정

`ppo_baseline.py`는 위와 같은 600입력, 32·32 은닉층, tanh, 6출력 정책을 PPO로 학습하고 `exp5.py`가 읽는 NPZ 형식으로 저장한다. 저자가 공개한 학습 설정(64개 환경 × 128스텝, 업데이트당 4회 반복·8개 미니배치, 학습률 5e-4의 선형 감소, 할인율 0.99, GAE 0.95, 클립 0.2, 엔트로피 계수 0.01)을 적용했다. 공개 로그에는 PPO 학습 코드·가치망 구조·초기화의 모든 세부 사항이 없으므로 **저자와 완전히 동일한 PPO 구현이라고 주장할 수는 없다**.

학습 중인 64개 게임은 `ppo_live_training.py --live-frame-every 32`로 관찰 상태를 주기적으로 저장할 수 있고, `watch_ppo.py`의 `http://127.0.0.1:8765/training`에서 볼 수 있다. 화면은 학습 환경의 실제 상태를 샘플링한 것으로, 저장 간격 사이의 모든 프레임을 녹화한 영상은 아니다. 시범 플레이는 뷰어의 `/`에서 별도로 볼 수 있다. **이 저장소에 기록한 세 시드의 학습은 모두 완료됐다.**

재설치할 때는 Python 3.14 가상환경에 `requirements-ppo.txt`를 설치한다. 아래 예비 속도 측정은 기존 로컬 워크스페이스에서 수행했다. 완성된 세 시드의 정책, 평가 원자료, 학습 로그, 압축된 최종 체크포인트는 [결과 아카이브](../../artifacts/ppo_500m_3seeds/README.md)에 포함했다.

```sh
uv venv --python 3.14 .venv-ppo
uv pip install --python .venv-ppo/bin/python -r experiments/weight_space_exp5/requirements-ppo.txt

# 100회 업데이트 = 819,200 환경 스텝. 결과는 Git에서 제외된다.
.venv-ppo/bin/python experiments/weight_space_exp5/ppo_baseline.py \
  --game space_invaders --updates 100 --seed 0 \
  --output-dir experiments/weight_space_exp5/results/space_invaders/ppo_bench_100

.venv/bin/python experiments/weight_space_exp5/exp5.py evaluate \
  --game space_invaders \
  --weights experiments/weight_space_exp5/results/space_invaders/ppo_bench_100/actor.npz \
  --episodes 1000 --workers 4 \
  --output experiments/weight_space_exp5/results/space_invaders/ppo_bench_100/eval_1000.json
```

이 기기의 Apple M1 Pro(10코어), 메모리 32GB에서 CPU 스레드 1개로 100회 업데이트를 실행한 결과 **819,200스텝에 33.12초**, 안정 구간 중앙값 **8,192스텝당 0.3294초**였다. 앞 25회 평균 0.331초, 뒤 25회 평균 0.333초였다. 이를 단순 선형 환산하면 **1억 스텝 약 1.12시간, 5억 스텝 약 5.58시간/시드**다. 긴 학습에서 발열, 백그라운드 작업, 정책 변화에 따른 게임 상태 분포 때문에 실제 시간은 달라질 수 있다. 81.9만 스텝 학습 가중치를 1,000회 평가한 결과는 **26.479 ± 0.429점**(에피소드 표준오차)이다. 이는 속도 측정 과정의 중간 정책 점수이며 장기 학습 기준선과 별개다. 예비 측정 파일은 Git 제외 폴더 `results/space_invaders/ppo_bench_100/`에 있다.

2026-11-25에 같은 PPO 코드와 시드 0, 30회 업데이트(245,760스텝)로 CPU와 Apple GPU(MPS)를 별도 측정했다. 첫 업데이트를 제외한 **8,192스텝당 중앙값**은 CPU **0.335초**, MPS **0.803초**였다. GPU 실행이 약 **2.39배 느렸다**. 전체 실행 시간도 CPU 10.16초, MPS 24.77초였다. 이 작은 정책망은 매 128스텝에 64개 게임의 CPU 상태를 GPU로 옮기고, 행동을 다시 CPU로 가져와야 해서 GPU 호출·전송 비용이 크다. 최종 5억 스텝 학습은 CPU에서 실행했다. 예비 원자료는 Git 제외 폴더 `results/space_invaders/device_bench_30/`에 있다.

### 5억 스텝 × 3개 시드 장시간 실행

`run_ppo_overnight.sh`는 시드 0·1·2를 순서대로 실행한다. 각 시드는 **500회 업데이트마다** `latest.pt`에 정책망, 가치망, Adam 상태, 64개 게임 상태, 난수 상태를 원자적으로 저장한다. 다음 실행에서 `--resume`이 있으면 해당 지점부터 학습을 계속한다. 업데이트 경계를 넘어설 때 `actor_100m.npz`와 `actor_500m.npz`를 따로 저장한다. 실제 저장 스텝은 각각 **100,007,936**과 **500,006,912**다. `progress.json`에 마지막 저장 스텝, `train.log`에 진행 기록을 남긴다. 재시작 검증에서 중간에 멈춘 6회 업데이트와 처음부터 연속 실행한 6회 업데이트의 모든 정책 텐서가 정확히 일치했다.

```sh
# 이 맥에 등록된 macOS 작업 상태
launchctl print gui/501/com.codex.weightspace.spaceinvaders.ppo.seed01
launchctl print gui/501/com.codex.weightspace.spaceinvaders.ppo.seed2
# 시드 0의 마지막 저장 지점
cat /private/tmp/weightspace-ppo/results/space_invaders/ppo_500m_3seeds/seed_0/progress.json
```

시드별 최종 파일은 `results/space_invaders/ppo_500m_3seeds/seed_<n>/`에 생긴다. **맥을 전원에 연결하고 덮개를 열어 둬야 한다.** `caffeinate`는 덮개를 닫아 발생하는 잠자기를 막을 수 없다. 이 실행의 1억 스텝 파일은 **5억 스텝까지 학습률을 낮추는 일정의 중간 체크포인트**다. 원문의 독립적인 1억 스텝 PPO 실험은 학습률을 1억 스텝에 맞춰 감소시켰을 가능성이 있으므로, 그 결과와 완전히 같은 학습 예산 설정이라고 보기는 어렵다.

2026-11-25 실제 장시간 실행은 macOS `launchd` 작업으로 등록했다. `launchd`가 사용자 `Documents` 경로의 스크립트를 읽지 못해, 동일한 코드와 가상환경을 `/private/tmp/weightspace-ppo/`에 복사해 실행했다. 완료 후 최종 결과를 이 저장소의 [결과 아카이브](../../artifacts/ppo_500m_3seeds/README.md)에 보존했다.

같은 날 09:54 KST에 시드 0이 완료된 뒤 시드 1과 시드 2를 **두 개의 독립된 CPU 작업**으로 전환했다. M1 Pro는 물리·논리 코어가 각각 10개(성능 8, 효율 2)이며, 각 PPO 프로세스는 약 1코어를 사용했다. 별도 권한으로 실행한 병렬 측정(작업당 819,200스텝)에서는 두 작업이 각각 35.56초와 35.57초 걸렸다. 단독 실행 약 33초보다 작업당 약 8% 느리지만, 합산 처리량은 약 1.86배였다. 시드 1은 155,648,000스텝 체크포인트에서 재개했다. 예비 원자료는 Git 제외 폴더 `results/space_invaders/parallel_bench/`에 있다.

한 시드에 PyTorch CPU 스레드를 2개 배정하는 방식도 같은 코드·시드의 50회 업데이트(409,600스텝)로 측정했다. 다른 두 학습 작업이 실행 중인 상태에서 1스레드 **19.29초**, 2스레드 **20.33초**로 2스레드가 약 5% 느렸다. 게임 환경 64개를 순차 실행하는 코드여서 스레드 수만 늘려서는 그 부분이 병렬화되지 않는다. 최종 세 시드 모두 `--threads 1`로 학습했다. 예비 원자료는 Git 제외 폴더 `results/space_invaders/thread_bench/`에 있다.

### 학습 중 게임 관전

`watch_ppo.py`의 [학습 화면](http://127.0.0.1:8765/training)은 실행 중 시드 1·2의 **실제 학습 환경 64개**를 주기적으로 샘플링해 보여 주었다. 세 시드 모두 완료된 현재는 마지막 화면만 남아 있다. 저장 간격 사이의 모든 프레임을 녹화한 영상은 아니다. [시범 플레이](http://127.0.0.1:8765/)는 최신 체크포인트 정책으로 **별도 게임**을 실행한다. 뷰어는 로컬 주소 `127.0.0.1`에만 열린다.
