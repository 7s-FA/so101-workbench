# Test_IL 카메라 구성 · 2026-09-22

**그리퍼: 32×32 UVC 모듈 / 상단: RealSense D435**. 학습 입력은 두 카메라의 RGB입니다. 모터 보정값은 변경하지 않았습니다.

## 적용한 모델

- 상단: 기존 SO-ARM100 상단 마운트의 아래·중간 기둥과 로봇 받침을 유지하고, 맨 위 부품을 D435용 **전방 돌출형**으로 수정했습니다. 뒤로 돌아가는 브래킷은 사용하지 않습니다.
- 그리퍼: [XLeRobot 원본](https://github.com/Vector-Wangel/XLeRobot/tree/main/hardware)의 시뮬레이션 자산 `XLeRobot_camera1.stl`(마운트), `XLeRobot_camera2.stl`(카메라)을 사용합니다. 앞서 만든 별도 측면 5홀 브래킷은 제거했습니다. 카메라 몸체를 임의로 늘리거나 원본 마운트를 변형하지 않았습니다.

원본 카메라는 32×32 모듈의 시각 모델입니다. 원본 메시의 PCB 외곽은 약 34mm로 단순화되어 있으므로 실제 구멍 간격·기판 외곽 치수는 실물을 우선합니다. 이 파일을 정밀 제조 도면으로 간주하지 않습니다.

## STL

| 파일 | 용도 |
|---|---|
| `overhead_top_d435_print.stl` | 새 상단 부품. 기존 상단 결합부 유지 |
| `wrist_xlerobot_mount_print.stl` | XLeRobot 원본 손목 마운트 형상 참고용 |
| `overhead_top_d435.stl`, `wrist_xlerobot_mount.stl` | 조립 좌표를 유지한 같은 형상 |

단위 **mm**, 배율 **100%**. `_print`는 원점·바닥에 맞춰 평행이동만 한 파일입니다. 출력 방향·서포트는 슬라이서에서 정해야 합니다. 카메라·나사 메시들은 표시용입니다.

## 상단 D435 위치와 체결

- 기존 하향 각도 **25°(수직 아래 기준)** 유지.
- 기둥 중심에 D435 몸체를 맞추고 앞쪽으로 25mm 이동했습니다. 새 RGB 명목 중심은 로봇 base_link 기준 **(50.987, −83.548, 545.672)mm**입니다. 기존 RGB 중심 대비 전방 +25mm, 좌측 +32.5mm, 높이는 동일합니다. D435의 RGB 센서는 몸체 중심에서 벗어나 있으므로 몸체 정렬과 광학 중심 정렬을 구분합니다.
- 하단 결합부(원본 STL Y≤280mm)의 표면 표본 최대 차이: **0.000006mm 미만**.
- D435 후면 **M3 2개, 간격 45mm**. 브래킷 통과 구멍 Ø3.4mm, 체결판 5mm, 간격용 와셔 0.5mm.
- M3×8을 사용하면 명목 삽입 길이 2.5mm입니다. 실제 나사·와셔를 확인해 카메라 내부 최대 3mm를 넘지 않아야 합니다.
- 센서 전면과 USB 포트를 덮지 않는 후면 체결입니다.

체결 기준: [공식 D400 데이터시트 Fig. 10-9](https://www.realsenseai.com/wp-content/uploads/2023/07/Intel-RealSense-D400-Series-Datasheet-July-2023.pdf#page=146). RGB/깊이 좌표는 공식 URDF의 명목값이며 실측 카메라 보정값이 아닙니다.

## 손목 마운트 위치

XLeRobot URDF의 조립 위치를 기준으로 SO-101의 동일 서보 체결 형상에 맞춰 좌표를 변환했습니다. 공통 형상 표면의 trimmed RMS는 약 **0.082mm**입니다. 교체 그리퍼 고정 홀더와의 겹침을 없애도록 전체 마운트를 측면으로 **1.85mm** 이동했고, 카메라는 원본 지지면에서 **0.05mm** 여유를 두었습니다. 원본 부품 형상·크기는 유지했습니다. XLeRobot 파일에는 별도 실물 구멍 보정값이 없으므로 체결 간격은 실측 확인이 필요합니다.

원본: 커밋 `3d14695e40c9c68229c0aacffca6053c75cd3eb6`, `simulation/xlerobot_urdf.zip`. 로컬 원본 ZIP의 Git blob SHA는 `193bb61c6f37b9067df11452432850371edec979`로 조회한 upstream 파일과 일치합니다. 소스·라이선스·해시는 `source_xlerobot/`에 보관합니다.

## 검증

`verification.json`에 실제 계산 결과를 저장합니다.

- STL 2개 모두 닫힌 단일 솔리드, 양의 체적.
- 카메라·마운트·고정 그리퍼 간 메시 겹침 없음.
- 그리퍼 0~100°를 5° 간격으로 검사: 21자세에서 손목 마운트·카메라와 움직이는 손가락 간 겹침 없음.
- 기본 준비 자세에서 다른 팔 링크·상단 기둥과 손목 카메라의 겹침 없음.
- Isaac Sim 실제 URDF 렌더링: 전체, 상단, 상단 정면, 손목 확대.

실물 체결·출력 공차·강도·기둥 휨·케이블과 모든 팔 동작 경로는 검증 전입니다. 카메라 외부 파라미터는 **CAD 명목값**으로 표시하며 모터 캘리브레이션을 대체하지 않습니다.

## 재생성

Test_IL 폴더에서 순서대로 실행합니다.

```bash
/path/to/isaac/env_isaaclab/bin/python scripts/build_camera_mounts.py
/path/to/isaac/env_isaaclab/bin/python scripts/check_camera_mounts.py
/path/to/venv/bin/python scripts/prepare_so101.py
/path/to/isaac/env_isaaclab/bin/python scripts/preview_camera_mounts.py
```

부울 연산 의존성은 `.cad_runtime/manifold3d` 3.5.3입니다. 원본 SO-ARM100 기둥은 커밋 `eecbe3e0a9ebb23e25ad7b2759b03884c6660903`, `OpenCV_teach/calibration/mount_geometry/source/`를 사용합니다. 다른 프로젝트의 원본은 수정하지 않았습니다.
