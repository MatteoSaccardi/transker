#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/tests"
PYTHON_BIN="${PYTHON:-python3}"
if [[ "$PYTHON_BIN" == */* && "$PYTHON_BIN" != /* ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/${PYTHON_BIN}"
fi

TEST_KEYS=("smoke" "c2c" "c2g" "levy" "sip" "ilt-sip-plot")
TEST_NAMES=("smoke" "c2c" "c2g" "levy" "sip" "ilt-sip-plot")
TEST_FILES=("smoke_test.py" "c2c_regulated_test.py" "c2g_test.py" "levy_test.py" "sip_test.py" "ilt_sip_plot.py")
TEST_ESTIMATES=("4 seconds" "10m 40s" "6m 50s" "2m 04s" "31m 54s" "10m 50s")

format_seconds() {
    local total="$1"
    local hours=$((total / 3600))
    local minutes=$(((total % 3600) / 60))
    local seconds=$((total % 60))

    if (( hours > 0 )); then
        printf "%dh %02dm %02ds" "$hours" "$minutes" "$seconds"
    else
        printf "%dm %02ds" "$minutes" "$seconds"
    fi
}

print_menu() {
    echo "Select tests to run by number, separated by spaces."
    echo
    echo "  1) all              estimated: 1h 2m 22s"
    echo "  2) smoke            estimated: ${TEST_ESTIMATES[0]}"
    echo "  3) c2c              estimated: ${TEST_ESTIMATES[1]}"
    echo "  4) c2g              estimated: ${TEST_ESTIMATES[2]}"
    echo "  5) levy             estimated: ${TEST_ESTIMATES[3]}"
    echo "  6) sip              estimated: ${TEST_ESTIMATES[4]}"
    echo "  7) ilt-sip-plot     estimated: ${TEST_ESTIMATES[5]}"
    echo
}

add_test_index() {
    local idx="$1"
    local existing

    for existing in "${SELECTED_INDICES[@]:-}"; do
        if [[ "$existing" == "$idx" ]]; then
            return
        fi
    done

    SELECTED_INDICES+=("$idx")
}

parse_selection() {
    local choice

    SELECTED_INDICES=()

    for choice in "$@"; do
        case "$choice" in
            1|all)
                SELECTED_INDICES=(0 1 2 3 4 5)
                return 0
                ;;
            2|smoke)
                add_test_index 0
                ;;
            3|c2c)
                add_test_index 1
                ;;
            4|c2g)
                add_test_index 2
                ;;
            5|levy)
                add_test_index 3
                ;;
            6|sip)
                add_test_index 4
                ;;
            7|ilt-sip-plot|ilt_sip_plot|ilt)
                add_test_index 5
                ;;
            *)
                echo "Unknown selection: ${choice}" >&2
                return 1
                ;;
        esac
    done

    if (( ${#SELECTED_INDICES[@]} == 0 )); then
        echo "No tests selected." >&2
        return 1
    fi
}

run_test() {
    local idx="$1"
    local name="${TEST_NAMES[$idx]}"
    local file="${TEST_FILES[$idx]}"
    local start
    local end
    local elapsed

    if [[ ! -f "$file" ]]; then
        echo "[$name] Missing file: ${TEST_DIR}/${file}"
        return 1
    fi

    echo
    echo "[$name] Running ${file} from ${TEST_DIR}"
    start="$(date +%s)"
    "$PYTHON_BIN" "$file"
    status="$?"
    end="$(date +%s)"
    elapsed=$((end - start))

    ACTUAL_TIMES+=("${name}: $(format_seconds "$elapsed")")

    if (( status != 0 )); then
        echo "[$name] Failed after $(format_seconds "$elapsed") with exit code ${status}."
        return "$status"
    fi

    echo "[$name] Done in $(format_seconds "$elapsed")."
}

main() {
    local selection
    local idx
    local overall_start
    local overall_end
    local status=0

    if [[ ! -d "$TEST_DIR" ]]; then
        echo "Tests directory not found: ${TEST_DIR}" >&2
        exit 1
    fi

    print_menu

    if (( $# > 0 )); then
        parse_selection "$@" || exit 1
    else
        read -r -p "Tests to run: " selection
        # shellcheck disable=SC2086
        parse_selection $selection || exit 1
    fi

    echo
    echo "Selected:"
    for idx in "${SELECTED_INDICES[@]}"; do
        echo "  - ${TEST_KEYS[$idx]} (${TEST_FILES[$idx]}), estimated: ${TEST_ESTIMATES[$idx]}"
    done

    cd "$TEST_DIR" || exit 1

    ACTUAL_TIMES=()
    overall_start="$(date +%s)"

    for idx in "${SELECTED_INDICES[@]}"; do
        if ! run_test "$idx"; then
            status=1
            break
        fi
    done

    overall_end="$(date +%s)"

    echo
    echo "Elapsed times:"
    if (( ${#ACTUAL_TIMES[@]} == 0 )); then
        echo "  No tests completed."
    else
        printf "  %s\n" "${ACTUAL_TIMES[@]}"
    fi
    echo "  Total: $(format_seconds "$((overall_end - overall_start))")"

    exit "$status"
}

main "$@"
