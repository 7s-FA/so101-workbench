# Demo — 기존 전체 공정 데모

2026-09-16 분리 보관. 9월 14일 최종 3×2m 데모의 팔 모델·공정·속도 설정을 사용합니다. 시작 버튼, 사전 적재/처음부터 상차 선택, 원재료 18개와 완제품 6개 공정을 포함합니다.

`final`로 가상환경과 프로젝트 루트에 들어간 뒤 실행합니다.

```bash
final
./Demo/start.sh --headless
```

사용자가 실행하면 기존 웹 모니터는 http://127.0.0.1:8767 에서 열립니다. 시작 버튼으로 공정을 시작합니다. 기존 `scripts/demo.sh`도 이 폴더 안에서 사용할 수 있습니다. 새 실행 결과는 `outputs/demo/`에 저장됩니다. 분리 전 기준 화면은 `outputs/final_demo_reference/`에 있습니다.

코드·설정·모델·출력은 Sim_real과 별도입니다. `.venv-io`는 공용 환경 `/home/fresh/.virtualenvs/final-arm-io`에 연결되며, Isaac Sim 설치 `/path/to/isaac/env_isaaclab`도 공용입니다. 최초 복사 시 포함된 다른 보조 스크립트보다 이 폴더의 `start.sh`를 기본 진입점으로 사용하세요. 상세 공정 설명은 `docs/demo.md`입니다.
