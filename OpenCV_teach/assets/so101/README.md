# SO-101 티칭 모델 · 손목 카메라 없음

2026-09-21부터 티칭 GUI는 상단 카메라를 사용합니다.
3D 모델에서 **RealSense D435, 카메라 마운트, 부착 나사 머리 5개, 와셔 2개**와 해당 전용 메시를 제거했습니다.
SO-101 본체와 XLeRobot 고정·가동 홀더 및 파란 TPU 손가락 2개는 유지합니다.
Final_Arm의 Demo·Sim_real 원본 모델은 변경하지 않았습니다.

- `so101_modified.urdf`: 원본 관절 구조와 기구학 계산 기준입니다.
- `so101_modified.xml`: MuJoCo 3D 표시용 모델입니다.
- `preview_scene.xml`: 조명과 바닥을 포함한 화면입니다.
- `provenance.json`: 원본 출처 및 현재 파일의 해시입니다.

몸체는 new/midrange calibration 기준입니다. 기본 관절 각도는 도→라디안 변환,
그리퍼 0~100은 0~1.74533 rad입니다. TCP 계산 기준은 `gripper_frame_link`이며
현재 팔로워 TCP는 이 프레임에서 (+1.773410, +0.602657, +6.549654) mm 이동한 고정 TPU 집게 끝선 중심입니다. 조립 메시로 계산한 모델 기준이며, 실물 오차는 별도 실측으로 맞춥니다. 계산 근거와 그림은 `calibration/tcp_geometry`에 있습니다.
카메라 기능에서 등록한 관절 영점 오프셋은 기구학 및 해당 3D 미리보기에 적용합니다.
모터 보정 JSON이나 모터 EEPROM은 수정하지 않습니다.

이 모델은 위치·자세 표시용입니다. 충돌·접촉·TPU 변형·실물 파지 판정을 수행하지 않습니다.

| 대상 | 출처 | 라이선스 |
|---|---|---|
| 본체 | TheRobotStudio/SO-ARM100, 로컬 vendor HEAD `eecbe3e0a9ebb23e25ad7b2759b03884c6660903` | LICENSE-SO-ARM100 |
| TPU 그리퍼 | Vector-Wangel/XLeRobot, `3d14695e40c9c68229c0aacffca6053c75cd3eb6` | LICENSE-XLeRobot |

두 원본 모두 Apache-2.0입니다. RealSense CAD는 현재 배포 모델에서 제거되었습니다.

2026-09-21 제공된 3MF 기반 링크 질량 재계산 결과는 [질량 계산 보고서](../../calibration/link_mass/README.md)에 있습니다. 출력물·서보 합계는 약 636.259 g이며 패드·별도 체결부품·제어보드 등 미산정 항목이 있습니다. 물리 시험용 `Test_teach/assets/so101/so101_teach.urdf`에 질량과 추정 관성을 적용했습니다. 이 폴더의 위치·자세 표시용 URDF/XML 질량은 유지하며, OpenCV 실물 위치 제어는 질량·마찰을 토크 보정으로 사용하지 않습니다.
