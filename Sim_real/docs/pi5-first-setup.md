# Raspberry Pi 5 최초 준비와 모방학습 실험

2026-09-14 현재 Pi의 RAM·OS·접속 주소와 시험할 팔은 미확정이고 학습 모델은 없다. 원격 설치나 실물 모터 구동은 아직 수행하지 않았다. PC는 RTX 4070 SUPER 12GB이며 기존 Isaac 환경을 보존한다.

## 첫 구성

`팔 + 손목 카메라 → Pi 5 → 유선 LAN/이후 Wi-Fi → GPU PC에서 모델 추론 → Pi의 제어기 → 팔`

우선 한 대만 연결한다. Pi 단독 CPU 추론은 같은 모델로 별도 비교한다. CPU에서 모델을 불러오는 것과 요구 주기로 동작하는 것은 별개의 합격 조건이다. Isaac Sim과 학습은 PC에 둔다. ROS 2 통합은 연결·추론 실험 이후에 추가한다.

## 준비물과 OS 설치

- Raspberry Pi 5, 냉각팬, microSD와 카드 리더. 저장장치는 실험용으로 64GB 이상을 권장하되 이는 공식 최소 요구량이 아니다. 기존 RAM 용량부터 확인하며 업그레이드를 먼저 요구하지 않는다.
- Pi용 27W USB-C 전원과 로봇팔에 맞는 별도 전원. Pi 전원과 팔 전원은 구분한다. 특히 이전 SO-101 저전압 문제를 해결하려고 확인 없이 12V로 바꾸지 않는다.
- 초기 연결용 LAN 케이블. PC와 Pi를 같은 공유기/스위치에 연결한다. 카메라는 우선 한 대만 붙인다.
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)에서 기기를 Raspberry Pi 5로, OS를 **Ubuntu Server 24.04 LTS 64-bit의 Raspberry Pi용 이미지**로 고른다. 목록에 없으면 [Ubuntu 공식 이미지 안내](https://ubuntu.com/hardware/docs/boards/how-to/ubuntu_supported/raspberry-pi/)에서 해당 이미지를 받아 사용자 지정 이미지로 쓴다.
- 기록 대상은 Pi용 microSD를 선택한다. 이미지 기록은 선택한 카드의 기존 내용을 지운다.
- 이미지에서 지원하면 계정·호스트명·SSH를 Imager에서 지정한다. 예시 호스트명은 `arm-pi5-01`이다. 지원하지 않으면 모니터·키보드로 첫 부팅을 완료하고 계정을 설정한다. 암호는 본인이 입력하고 채팅에 보내지 않는다.

Ubuntu 24.04는 Pi 5를 지원하며 기존 PC의 ROS 2 Jazzy와 OS 계열을 맞추기 위한 선택이다. Pi 5는 5V/5A급 전원을 권장한다. 적절한 전원을 인식했을 때 USB 주변장치 전류 한도가 총 1.6A로 늘어난다. 카메라 두 대의 소비전력은 모델 확인 후 검토한다. [Ubuntu 지원표](https://ubuntu.com/hardware/docs/boards/how-to/ubuntu_supported/raspberry-pi/) · [Pi 공식 시작 안내](https://www.raspberrypi.com/documentation/computers/getting-started.html)

## Pi 첫 부팅 후

아래 명령은 **Pi 터미널**에서 실행한다. 첫 부팅 초기 설정이 끝난 뒤 진행한다.

```bash
sudo apt update
sudo apt upgrade
sudo apt install python3-venv python3-pip git openssh-server usbutils v4l-utils
sudo systemctl enable --now ssh
sudo usermod -aG dialout,video "$USER"
hostname -I
uname -m
free -h
```

그룹 변경은 로그아웃·재로그인 후 적용된다. `uname -m`은 `aarch64`여야 한다. 표시된 Pi의 LAN 주소와 로그인 계정으로 PC에서 SSH에 접속한다. 최초 호스트 키는 Pi에서 표시한 지문과 비교한다. 이 문서의 `PI_LOGIN`·`PI_IP`는 실제 계정·주소로 바꿀 자리이며 그대로 실행하지 않는다.

```bash
ssh PI_LOGIN@PI_IP
```

## 설치 상태 점검

PC에서 점검 파일 하나만 복사한다. 기존 PC 가상환경은 ARM64인 Pi로 복사하지 않는다.

```bash
scp /path/to/Final_Arm/Sim_real/scripts/pi_preflight.py PI_LOGIN@PI_IP:~/pi_preflight.py
```

Pi에서 실행한다. 별도 Python 패키지 없이 동작하며 USB 장치를 열거나 모터에 명령하지 않는다.

```bash
python3 ~/pi_preflight.py > ~/pi_preflight.json
cat ~/pi_preflight.json
```

장치 연결 전에는 USB 팔·카메라 항목이 비어 있는 것이 정상이다. 팔을 클램프로 고정하고 해당 모델의 전원 절차에 따라 USB와 전원을 연결한 뒤 다시 실행해 장치 인식 차이를 확인한다. 포트 이름만으로 리더/팔로워를 자동 판단하지 않는다. 깊이 카메라의 정확한 모델도 이 단계에서 확인한다.

## 그다음 실험 순서

1. **연결:** 팔로워 종류·USB 포트·모터 ID와 카메라 모델 확인, 관절 상태 읽기. 기존 `neutral.json`은 시뮬레이션용이라 실물 팔로워 보정으로 재사용하지 않는다.
2. **환경:** 별도 학습용 가상환경에 고정한 LeRobot을 설치한다. 로컬 vendor 소스는 0.6.2, Python 3.12 이상이다. `config/source_revisions.json`의 커밋을 기준으로 PC/Pi를 맞춘다. ARM64용 PyTorch·카메라 드라이버 설치 가능 여부를 Pi에서 확인한다. 현재 vendor의 uv 설정은 Linux CUDA 인덱스를 지정하므로 Pi에 그대로 적용하지 않는다.
3. **모델 없는 추론 검사:** ACT를 첫 후보로 삼아 임의 초기화 모델을 입력 형상·계산 시간 측정에만 사용한다. 출력은 모터로 보내지 않는다. 이는 학습된 집기 능력의 검증이 아니다.
4. **학습:** 리더로 간단한 부품 하나 집어 옮기기를 시연하고, RGB 영상·팔로워 관절 상태·행동을 함께 기록해 PC에서 학습한다. 기존 JSONL 재생 기록에는 학습용 카메라 데이터가 없다.
5. **비교:** 같은 모델·영상 입력으로 PC GPU와 Pi CPU의 추론 지연(p50/p95), 메모리·온도, 통신 지연, 행동 버퍼 부족을 측정한다. 행동 묶음의 길이를 단순히 나누어 실시간 성공으로 판정하지 않는다.
6. **실물:** 읽기/추론 검사 후 관절별 속도·가속도·목표 변화·명령 유효기간 제한을 갖춘 제어기로 작은 범위를 먼저 시험한다. 네트워크가 끊기거나 명령이 오래되면 적재 자세를 고려해 제어된 정지를 수행한다. 이전 데모의 2.4rad/s 설정을 곧바로 실물에 적용하지 않는다.

LeRobot의 공식 비동기 추론은 PC 서버/Pi 클라이언트 분리를 지원하며 현재 소스에 SO-101·OMX와 ACT가 포함되어 있다. 단, 기본 `robot_client`는 초기화 중 로봇에 연결하고 이후 행동을 전송한다. 읽기 전용 점검으로 실행하면 안 된다. 기본 전송은 인증 없는 gRPC와 pickle을 사용하므로 추론 서버는 localhost에 두고 SSH 터널로 연결하는 구성을 우선한다. 공개 포트 개방이나 공유기 포트포워딩은 필요 없다. [공식 비동기 추론 문서](https://huggingface.co/docs/lerobot/en/async)

현재 완료한 것은 준비 안내와 읽기 전용 점검 파일이다. Pi 실측, 학습용 환경 설치, 추론 벤치마크, 실물 구동은 Pi 준비 후 별도로 수행한다.
