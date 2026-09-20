# 과거 분석·시각화 파일 안내

**현재 측정/총점은 [`evaluate.py`](../evaluate.py)를 사용합니다.** [현재 방법](current_evaluation.md)과 [2026-09-20 결과](../experiments/2026-09-20/README.md)가 최신 기준입니다.

기존 파일은 논문 준비 중 비교했던 방법과 시각화의 출처를 남기기 위해 보존했습니다. 아래 프로그램의 결과를 현재 상대 정렬 지표·동일 비중 총점과 섞지 마세요.

| 파일/폴더 | 용도 |
|---|---|
| `main.py`, `analysis/`, `visualization/` | 기존 포즈·SAM2·회전/헤드업 시각화 경로 |
| `validate_scoring.py`, `scoring/raw_metrics.py`, `scoring/score_mapper.py`, `scoring/extraction_cache.py` | 과거 3D 회전 기반 채점·캐시 경로. 과거 기본 가중치는 상체/헤드업/회전 0.5/0.3/0.2 |
| `extract_depth_metrics.py`, `extract_all_metrics.py`, `extract_selected_metrics.py`, `extract_in_out.py` | 기존 캐시·터치/3D 회전 지표 추출. 현재 관절 추출은 `extract_pose.py` |
| `retrack_balls.py`, `sam2_ball_tracker.py` | 과거 SAM2 공 재추적. 현재 공 후보는 `redetect_balls.py` |
| `make_skeleton_videos.py` | 기존 시각화 출력. 현재 점수의 시각적 증명이 아니며 공 표시는 과거 캐시 사용 |
| `analyze_*.py`, `diagnose_rotation*.py`, `compare_methods.py`, `plot_no_drift.py`, `validate_rotation.py` | XZ 방향각·깊이·필터·등급별 회전량을 살펴본 진단 스크립트 |
| `extract_rotation_score.py`, `compare_spike_methods.py`, `visualize_rotation_trace.py`, `visualize_3d_pose 9_1 62frame .py` | 과거 회전량·좌표·이상치 시각화 |
| `build_p2_model.py`, `test_sapiens.py` | 별도 학습/외부 모델 시험. 현재 평가에 필요하지 않음 |

과거 진단 일부는 특정 로컬 캐시/영상이나 추가 패키지를 전제로 하고 실행 즉시 그래프를 생성합니다. 현재 경로의 패키지 설치만으로 모든 과거 실험을 재실행할 수 있다는 의미는 아닙니다. 자동 테스트 범위는 `tests/`이며 모델 다운로드·학습 실험을 테스트로 실행하지 않습니다.

[보관된 이전 README](legacy_workflow.md), [과거 채점 설명](scoring_explained.md), [Hampel 실험](rotation_spike_filter_analysis.md)은 당시 방법의 기록입니다. 현재 결과 해석에는 [현재 계산식](current_evaluation.md)을 따릅니다.
