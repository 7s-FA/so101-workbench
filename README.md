# SO-101 Workbench

현재 OpenCV 티칭 프로그램으로 이어지기 전의 SO-101 시뮬레이션·실물 티칭·모방학습 작업을 모았습니다. 현재 프로그램은 **[opencv-teach](https://github.com/7s-FA/opencv-teach)**에서 관리합니다.

| 프로젝트 | 작업 내용 | 진입 문서 |
|---|---|---|
| `SO101_rl` | Isaac Sim 기반 B 조립 시뮬레이션·정합·강화학습 실험 | [설명](SO101_rl/README.ko.md) |
| `Sim_real` | 실물 배치를 반영하던 전체 공정 시뮬레이션 | [설명](Sim_real/README.md) |
| `Test_teach` | SO-101 집기·배치·실물 연결 시험 | [설명](Test_teach/README.md) |
| `Test_IL` | 리더·팔로워 시연 기록, ACT 학습·실행 도구 | [설명](Test_IL/README.md) |
| `Demo` | 초기 전체 공정 데모 | [설명](Demo/README.md) |
| `OpenCV_teach` | 초기 OpenCV 티칭 GUI와 계산·장치 제어 | [설명](OpenCV_teach/README.ko.md) |

## 보관 범위

실행 코드, 설정 예제, 필요한 CAD·시뮬 모델을 보관합니다. 학습 가중치·대용량 학습 데이터·실행 로그·검증 출력·백업·가상환경은 제외했습니다. OMX 전용 파일도 제외했으며 공유 공정 코드에 남아 있는 과거 OMX 언급을 현재 지원 범위로 해석하지 않습니다. `Sim_real`과 `Demo`는 당시 전체 공정 기록이어서 이 저장소의 SO-101 범위만으로 전체 공정을 즉시 재생할 수 있는 배포 패키지는 아닙니다.

현재 시뮬레이션의 정합·강화학습은 진행 중입니다. 과거 README의 수치는 당시 조건의 기록이며 최신 실물 성공률을 뜻하지 않습니다.

## 환경 복원

각 프로젝트 README의 Python·Isaac Sim·ROS 2 환경을 따릅니다. 공개본의 `/path/to/...`는 자신의 설치 경로로 바꿉니다. 외부 의존 저장소는 중복 복사 대신 `vendor-sources.json`에 URL과 커밋을 기록했습니다.

```bash
python3 tools/restore_vendor.py
```

이미 있는 vendor 폴더는 덮어쓰지 않습니다. 현재 티칭의 설정·모델을 참조하는 `SO101_rl`은 두 저장소를 나란히 받은 뒤 아래처럼 연결할 수 있습니다.

```bash
ln -s ../opencv-teach/SO101_teach SO101_teach
```

Isaac Sim 설치, Python 가상환경, 실물 장치 보정·연결은 별도로 준비해야 합니다. 이 저장소는 이전 작업의 소스 보관본이며 전체 실물·시뮬레이션 실행을 새로 검증한 배포본은 아닙니다. 외부 라이선스는 `third-party-licenses/`에 보존했습니다.
