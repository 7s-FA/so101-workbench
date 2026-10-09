# Ball transfer 조립 작업대와 시연 기록

확정한 구성: 하우징·볼·캡 용기 **3개**, 각 **2×2 = 4칸**. 총 12부품, 4세트 분량이다.
목재 느낌의 작업대 두 개는 10cm 떨어져 있다. 작업대 1에는 재료 용기 3개와 OMX-AI,
작업대 2에는 SO-101, 1세트용 준비 용기(부품별 1칸), 조립 지그, 완성품 1개용 저장 용기가 있다.
작업대 크기는 각각 55×70cm, 높이 70cm이며 임시 배치다.

## 실행

프로젝트 폴더 `/path/to/Final_Arm`에서 실행한다.

```bash
# 배치 확인 / 리더암 직접 조작 (기록하지 않음)
./scripts/assembly.sh manual

# 직접 조작하며 기록: 새 파일 이름을 사용한다.
./scripts/assembly.sh record outputs/episodes/take_001.jsonl

# 기록을 마치려면 시뮬레이터 터미널에서 Ctrl+C를 한 번 누른다.
# 정리와 저장이 끝난 후, 같은 초기 배치에서 자동 재생한다.
./scripts/assembly.sh replay outputs/episodes/take_001.jsonl
```

리더암 입력은 별도 터미널 두 개에서 기존 `io.sh`를 실행한다.
실제 장치를 연결하고 [README](../README.md)의 기준 자세 등록을 먼저 수행한다.
장치 경로는 해당 리더의 `/dev/serial/by-id/...`로 바꾼다.

```bash
./scripts/io.sh --model omx --port /dev/serial/by-id/OMX_LEADER --calibration config/omx_neutral.json
./scripts/io.sh --model so101 --port /dev/serial/by-id/SO101_LEADER --calibration config/so101_neutral.json
```

두 입력 프로세스는 각자 관절 이름을 포함한 ROS 2 JointState를 발행한다.
기록은 시뮬레이션 시작부터 종료까지 이어지므로, 시작 직후 대기 시간도 포함된다.
재생 중에는 리더 입력을 무시하고 마지막 관절 목표를 유지한다. 재생을 중단하려면 Ctrl+C를 누른다.
재생은 가상 팔에만 적용하며 실물 팔로 명령을 발행하지 않는다.
현재 두 모드는 실행 명령으로 구분하며 화면 내 녹화 버튼은 없다.

## 시연 순서

1. OMX로 재료 용기에서 하우징·볼·캡을 각각 준비 용기의 해당 칸으로 옮긴다.
2. OMX를 물리고 SO-101로 하우징을 지그에 놓는다.
3. SO-101로 볼을 넣고 캡을 정렬한다.
4. 완성품 저장 칸으로 옮기는 과정을 시험한다.

위 순서는 수집할 시연의 작업 순서이며, 이미 성공한 자동 조립 궤적이 아니다.
이번 구현은 두 팔의 동작을 함께 기록하고 재생할 수 있는 환경을 제공한다.
집기와 삽입은 PhysX 접촉으로 수행하며 물체를 그리퍼에 강제로 붙이거나 순간 이동시키지 않는다.
원본 제품은 자석으로 캡을 결합한다. **현재 자석·접착 구속은 없으므로 캡 고정 및 완성품 운반 성공은 보장되지 않는다.**
실물 리더를 통한 시연, 파지 자세·그리퍼 간격·충돌 경로·삽입 공차 검증은 남아 있다.

## 기록과 자동 모드의 의미

자동 모드는 **시연 궤적 재생**이다. 물체 위치를 인식해 보정하는 학습 정책은 아니다.
장면을 매번 같은 초기 배치로 만들고 두 팔에 같은 시간축의 관절 목표를 준다.
접촉 결과는 실행 환경이나 물리 조건이 달라지면 달라질 수 있다.

JSONL 파일에는 다음을 저장한다.

- 헤더: 형식 버전, 1/120초 시간 간격, 관절 이름·범위·기준 자세, 장면/코드/모델 해시, 부품 초기 위치.
- 프레임: 프레임 번호, 두 팔의 속도 제한 후 관절 명령과 실제 관절각, 입력 추종 여부, 12부품의 실제 위치·자세.
- 마지막 행: 프레임 수와 정상 파일 종료 여부. 정상 종료는 조립 성공을 뜻하지 않는다.

시간은 `index × dt`, 각도는 rad, 위치는 m, 부품 quaternion은 wxyz 순서다.
일반 `leader_ros` 표기는 입력 경로를 뜻하며 실제 사람의 조작이라는 인증 정보는 아니다.
`--demo` 합성 입력은 `synthetic_test`로 표시되고 일반 재생에서 거부한다.
관절 데이터와 부품 상태를 저장하며 연속 카메라 영상·힘·접촉 센서 데이터는 아직 기록하지 않는다.
`--capture`는 별도로 개요/손목 카메라 정지 사진을 저장한다.

파일이 미완성이거나, 입력이 전혀 없거나, 모델·배치·코드가 바뀌었거나,
관절 명령이 잘못된 경우 재생을 시작하지 않는다. 기존 기록은 덮어쓰지 않는다.
이 데이터는 향후 모방학습 데이터의 기반으로 쓸 수 있지만 LeRobot 학습 데이터셋 형식으로 변환되지는 않았다.

## 모델과 물리 근사

첨부 ZIP의 STL·STEP·이미지·설명 자료만 복사했으며 첨부 Python은 실행하지 않았다.
원본 단위 mm를 m로 변환했고, 캡 STL은 X축으로 180도 돌린 뒤 밑면을 z=0에 맞췄다.
하우징 50×50×23mm, 볼 지름 30mm, 캡 높이 7.7mm다.
조립 시 하우징 바닥 기준 볼 STL 원점은 z=3mm, 변환된 캡 바닥은 z=20.3mm다.
지그 바닥은 작업대 위이고 하우징은 지그 바닥보다 4mm 위에 놓인다.

볼은 구 충돌체, 움직이는 하우징·캡은 최대 64개의 convex decomposition 충돌체,
정적 지그는 삼각형 메시를 사용한다. 단일 convex hull로 오목한 구멍을 막지 않도록 구성했지만
작은 삽입 공차를 충돌 근사가 정확히 보존하는지는 별도 접촉 시험이 필요하다.
[NVIDIA 충돌체 문서](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html).

질량(하우징 35g·볼 18g·캡 15g), 마찰계수 0.8, 접촉 여유 0.5mm는 임시값이다.
로봇 이득·힘 제한도 기존 시뮬레이션 시험값이며 실물 모터 사양을 재현하지 않는다.
물리 계산은 PC CPU, 렌더링은 GPU를 사용한다. Raspberry Pi 5는 향후 리더/팔 I/O와 ROS 2 통신을 담당한다.

배치와 임시 물성은 `config/assembly.json`, 장면 생성은 `arm_sim/workcell.py`, 기록 형식은
`arm_sim/episodes.py`에 있다. 초기화·제어 코드가 바뀌면 새 시연을 기록해야 한다.

## 검증 명령

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 scripts/check_workcell_reach.py
./scripts/assembly.sh manual --headless --steps 600 --capture

# 합성 입력 기반 기록/재생 검증: 실제 조립 시연 데이터가 아니다.
./scripts/assembly.sh record outputs/test_new.jsonl --headless --demo --steps 600
./scripts/assembly.sh replay outputs/test_new.jsonl --headless --allow-synthetic-replay
```

위치 IK 검사는 TCP 위치만 검사한다. 파지 방향, 접근 경로와 충돌 회피를 검증하지 않는다.
렌더는 `outputs/assembly_overview.png`, 장면은 `outputs/assembly_scene.usda`,
실행 보고서는 `outputs/assembly_report.json`에 생성된다.

2026-09-10 검증 결과: 단위 시험 16개 통과. 외부 ROS 합성 입력을 수신하며 522프레임을
기록했고, 그중 406프레임에서 입력 추종 상태를 확인했다. Ctrl+C 정상 저장 후 재생한
522프레임 전체에서 기록 대비 최대 관절 오차와 부품 위치 오차는 이 실행 환경에서 0이었다.
12개 부품은 용기 안에서 약 1mm 내려앉고 수평 위치를 유지했다.
작업 지점 20개 위치 IK의 최대 잔차는 0.0116mm였다.
근거는 `outputs/assembly_validation.json`, `outputs/reach_report.json`과 해당 실행 로그다.
이 검증은 이송·조립 성공률 측정이 아니다. 실물 리더는 연결되지 않아 실제 시연은 아직 없다.

## 직접 조작 속도 조정 (2026-09-10)

사용자 승인에 따라 기본 리더 발행 주기를 30Hz에서 60Hz로, 관절 목표 변화 제한을
0.8rad/s에서 3rad/s로 높였다. 가상 구동기의 속도 상한은 2rad/s에서 4rad/s로 높였다.
이득·힘 제한 및 실물 모터 설정은 변경하지 않았다. `io.sh --rate 30`으로 입력 주기만 낮출 수 있다.
변경 전 실측: OMX 입력 29.9Hz, ROS 전달 지연 중앙값 약 0.2ms, 시뮬레이션 시간 비율 0.934.
90도 목표의 명령 증가 시간은 약 1.97초에서 0.525초로 줄어든다. 실제 팔 도달 시간은
자세·하중·접촉·시뮬레이션 처리 속도의 영향을 받는다. 이전 속도 설정으로 만든 기록은
장면 해시가 달라 재생이 거부되므로 현재 설정에서 다시 기록해야 한다.
