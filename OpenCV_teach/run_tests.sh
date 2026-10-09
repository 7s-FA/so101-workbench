#!/usr/bin/env bash
set -Eeuo pipefail

OPENCV_TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENCV_TEST_PYTHON="${OPENCV_TEACH_PYTHON:-$HOME/venv/il/bin/python}"
OPENCV_TEST_NEW_LOGS=()
for OPENCV_TEST_LOG in "$OPENCV_TEST_DIR/MUJOCO_LOG.TXT" "$OPENCV_TEST_DIR/../MUJOCO_LOG.TXT"; do
    # Keep any log that predates this test, including dangling symbolic links.
    if [[ ! -e "$OPENCV_TEST_LOG" && ! -L "$OPENCV_TEST_LOG" ]]; then
        OPENCV_TEST_NEW_LOGS+=("$OPENCV_TEST_LOG")
    fi
done

opencv_test_cleanup() {
    local test_status=$?
    trap - EXIT
    for OPENCV_TEST_LOG in "${OPENCV_TEST_NEW_LOGS[@]}"; do
        if [[ -f "$OPENCV_TEST_LOG" && ! -L "$OPENCV_TEST_LOG" ]]; then
            if [[ "$test_status" -ne 0 ]]; then
                tail -n 40 -- "$OPENCV_TEST_LOG" >&2 || true
            fi
            rm -f -- "$OPENCV_TEST_LOG" || true
        fi
    done
    exit "$test_status"
}
trap opencv_test_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$OPENCV_TEST_DIR"
unset PYTHONPATH PYTHONHOME
if [[ $# -eq 0 ]]; then
    # Default smoke coverage: 21 cases, below the user's per-task limit of 30.
    # Pass explicit module/case names when checking a different change.
    set -- test_workflow_cleanup test_quick_reference test_app_navigation \
        test_motor_calibration.SavedCalibrationTests.test_imported_file_uses_exact_offsets_and_ranges_without_rehoming \
        test_follower_only.FollowerTests.test_hardware_initialization_never_constructs_leader_or_opens_usb
fi
"$OPENCV_TEST_PYTHON" -m unittest -v "$@"
