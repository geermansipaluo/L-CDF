#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# DensityNet 批量闭环评测脚本：通过 roslaunch 启动完整仿真环境
#
# 放置位置：
#   /home/guo/L-CDF/src/sensor_cdf/scripts/run_eval_sweep.sh
#   /home/guo/L-CDF/src/sensor_cdf/scripts/traj_eval_sweep.py
#   /home/guo/L-CDF/src/scene/launch/traj_eval_sweep.launch
#
# 运行：
#   cd /home/guo/L-CDF/src/sensor_cdf/scripts
#   bash run_eval_sweep.sh
# ============================================================

LAUNCH_PKG="${LAUNCH_PKG:-scene}"
LAUNCH_FILE="${LAUNCH_FILE:-traj_eval_sweep.launch}"

# network_arch:
#   densitynet : 原始 DensityNet / L-CDF 模型
#   pointnet2  : PointNet++-BC 学习型 baseline
NETWORK_ARCH="${NETWORK_ARCH:-densitynet}"
if [[ "${NETWORK_ARCH}" != "densitynet" && "${NETWORK_ARCH}" != "pointnet2" ]]; then
  echo "[ERROR] NETWORK_ARCH 只能是 densitynet 或 pointnet2，当前: ${NETWORK_ARCH}"
  exit 1
fi

# 当前模型默认目录；也可运行前覆盖：
#   DensityNet:
#     MODEL_ROOT=/path/to/saved_models/learnable bash run_eval_sweep.sh
#   PointNet++:
#     NETWORK_ARCH=pointnet2 MODEL_ROOT=/path/to/pointnet2_models bash run_eval_sweep.sh
if [[ "${NETWORK_ARCH}" == "pointnet2" ]]; then
  DEFAULT_MODEL_ROOT="/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/pn2"
  DEFAULT_MODEL_SUBDIR_PREFIX="PointNet2"
  DEFAULT_CSV_PATH="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics_pointnet2.csv"
  DEFAULT_TRAJ_DIR="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories_pointnet2"
else
  DEFAULT_MODEL_ROOT="/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/ablation/wo_safe"
  DEFAULT_MODEL_SUBDIR_PREFIX="DensityNet"
  DEFAULT_CSV_PATH="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics.csv"
  DEFAULT_TRAJ_DIR="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories"
fi

MODEL_ROOT="${MODEL_ROOT:-${DEFAULT_MODEL_ROOT}}"
MODEL_SUBDIR_PREFIX="${MODEL_SUBDIR_PREFIX:-${DEFAULT_MODEL_SUBDIR_PREFIX}}"
# 可选：如果你的训练保存目录不是 PointNet2-demo48-dseed0-seed0 这种标准格式，
# 可以用模板显式指定，例如：
# MODEL_FILE_TEMPLATE="{MODEL_ROOT}/UnknownGym-DensityNet-PointNet2_BC-arch-pointnet2-ablation-no_safety-demo{N}-dseed{DS}-seed{TS}-xxx/model_best_parametric_bc.pt"
MODEL_FILE_TEMPLATE="${MODEL_FILE_TEMPLATE:-}"
MODEL_FIND_MAXDEPTH="${MODEL_FIND_MAXDEPTH:-4}"

ALT_MODEL_ROOT="/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models"

if [[ ! -d "${MODEL_ROOT}" && -d "${ALT_MODEL_ROOT}" ]]; then
  echo "[INFO] MODEL_ROOT 不存在，自动切换到: ${ALT_MODEL_ROOT}"
  MODEL_ROOT="${ALT_MODEL_ROOT}"
fi

CSV_PATH="${CSV_PATH:-${DEFAULT_CSV_PATH}}"
TRAJ_DIR="${TRAJ_DIR:-${DEFAULT_TRAJ_DIR}}"

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
TEST_TARGET_SEED="${TEST_TARGET_SEED:-2026}"
MAX_EPISODE_TIME="${MAX_EPISODE_TIME:-80.0}"
HOLD_BEFORE_EPISODE="${HOLD_BEFORE_EPISODE:-2.0}"
GOAL_RADIUS="${GOAL_RADIUS:-0.4}"
TERMINATE_ON_COLLISION="${TERMINATE_ON_COLLISION:-false}"
USE_RVIZ="${USE_RVIZ:-false}"

# 现在默认专门评测终点 y=0。
# 可选: fixed_y0 / fixed / random_y0 / random
TARGET_MODE="${TARGET_MODE:-fixed_y0}"
FIXED_TARGET_X="${FIXED_TARGET_X:-15.0}"
FIXED_TARGET_Y="${FIXED_TARGET_Y:-0.0}"
TARGET_X_MIN="${TARGET_X_MIN:-14.0}"
TARGET_X_MAX="${TARGET_X_MAX:-16.0}"
TARGET_Y_MIN="${TARGET_Y_MIN:--2.0}"
TARGET_Y_MAX="${TARGET_Y_MAX:-2.0}"

# 当前模型结构参数：必须与训练时一致
STATE_DIM="${STATE_DIM:-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
GRAPH_K="${GRAPH_K:-5}"
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-25.0}"
QP_LIMIT="${QP_LIMIT:-1.2}"
if [[ "${NETWORK_ARCH}" == "pointnet2" ]]; then
  ABLATION="${ABLATION:-no_safety}"
else
  ABLATION="${ABLATION:-no_safety}"
fi
NOMINAL_SPEED="${NOMINAL_SPEED:-1.2}"

# PointNet++-BC 参数：仅 NETWORK_ARCH=pointnet2 时使用
POINTNET2_MAX_POINTS="${POINTNET2_MAX_POINTS:-200}"
POINTNET2_NPOINT1="${POINTNET2_NPOINT1:-64}"
POINTNET2_RADIUS1="${POINTNET2_RADIUS1:-0.5}"
POINTNET2_NSAMPLE1="${POINTNET2_NSAMPLE1:-16}"
POINTNET2_NPOINT2="${POINTNET2_NPOINT2:-16}"
POINTNET2_RADIUS2="${POINTNET2_RADIUS2:-1.0}"
POINTNET2_NSAMPLE2="${POINTNET2_NSAMPLE2:-16}"
POINTNET2_PADDING_VALUE="${POINTNET2_PADDING_VALUE:-99.0}"

# qpth=评估当前模型内部 safety_layer；
# nominal=只测 head；
# post_cdfqp/bc_cdfqp/jax=网络 nominal + 解析/JAX SDF-CDF-QP 后处理。
# 评测 Pure BC + test-time CDF-QP 推荐:
#   ABLATION=no_safety RUNTIME_QP_MODE=bc_cdfqp MODEL_ROOT=/path/to/pure_bc_models bash run_eval_sweep.sh
if [[ "${NETWORK_ARCH}" == "pointnet2" ]]; then
  # PointNet++-BC 没有内部 qpth safety layer。
  # 纯 PointNet++-BC 用 nominal；PointNet++-BC + CDFQP 可运行前覆盖 RUNTIME_QP_MODE=jax。
  RUNTIME_QP_MODE="${RUNTIME_QP_MODE:-nominal}"
else
  RUNTIME_QP_MODE="${RUNTIME_QP_MODE:-nominal}"
fi
ENABLE_TEST_TIME_CDFQP="${ENABLE_TEST_TIME_CDFQP:-false}"
ANALYTIC_CDFQP_NUM_POINTS="${ANALYTIC_CDFQP_NUM_POINTS:-200}"
ANALYTIC_CDFQP_PADDING_VALUE="${ANALYTIC_CDFQP_PADDING_VALUE:-99.0}"
ANALYTIC_CDFQP_FALLBACK_TO_NOMINAL="${ANALYTIC_CDFQP_FALLBACK_TO_NOMINAL:-true}"

# qpth safety layer 数值参数：与新版 model.py 对齐
USE_QP_BOX_CONSTRAINTS="${USE_QP_BOX_CONSTRAINTS:-true}"
QP_JITTER="${QP_JITTER:-0.0001}"
QP_NORMALIZE_CONSTRAINTS="${QP_NORMALIZE_CONSTRAINTS:-false}"
QP_CONSTRAINT_SCALE_FLOOR="${QP_CONSTRAINT_SCALE_FLOOR:-1.0}"
QP_BOX_EPS="${QP_BOX_EPS:-0.0001}"
QP_MAX_ITER="${QP_MAX_ITER:-100}"
QP_EPS="${QP_EPS:-0.0001}"
QP_NOT_IMPROVED_LIM="${QP_NOT_IMPROVED_LIM:-20}"
QP_FAIL_MODE="${QP_FAIL_MODE:-fallback}"
QP_DEBUG_MAX_PRINT="${QP_DEBUG_MAX_PRINT:-20}"
QP_CHECK_INVALID_CONSTRAINTS="${QP_CHECK_INVALID_CONSTRAINTS:-true}"
QP_INVALID_G_NORM_EPS="${QP_INVALID_G_NORM_EPS:-1e-8}"
QP_INVALID_H_EPS="${QP_INVALID_H_EPS:-1e-6}"
QP_INVALID_CONSTRAINT_MODE="${QP_INVALID_CONSTRAINT_MODE:-warn}"
QP_INVALID_DEBUG_MAX_PRINT="${QP_INVALID_DEBUG_MAX_PRINT:-20}"
QP_SANITIZE_REDUNDANT_CONSTRAINTS="${QP_SANITIZE_REDUNDANT_CONSTRAINTS:-true}"
QP_REDUNDANT_CONSTRAINT_H="${QP_REDUNDANT_CONSTRAINT_H:-1.0}"
QP_VERIFY_SOLUTION="${QP_VERIFY_SOLUTION:-true}"
QP_SOLUTION_VIOLATION_TOL="${QP_SOLUTION_VIOLATION_TOL:-0.001}"
QP_SOLUTION_DEBUG_MAX_PRINT="${QP_SOLUTION_DEBUG_MAX_PRINT:-20}"
QP_SUPPRESS_QPTH_WARNINGS="${QP_SUPPRESS_QPTH_WARNINGS:-true}"
QPTH_FAIL_FALLBACK_TO_JAX="${QPTH_FAIL_FALLBACK_TO_JAX:-true}"

# checkpoint 中如果含 lambda_raw/lambda_prior，评测节点会自动强制 true；这里默认 false 更兼容旧模型
if [[ "${NETWORK_ARCH}" == "pointnet2" ]]; then
  LEARNABLE_LAMBDA_SMOOTH="${LEARNABLE_LAMBDA_SMOOTH:-false}"
else
  LEARNABLE_LAMBDA_SMOOTH="${LEARNABLE_LAMBDA_SMOOTH:-true}"
fi
LAMBDA_SMOOTH_MIN="${LAMBDA_SMOOTH_MIN:-1}"
LAMBDA_SMOOTH_MAX="${LAMBDA_SMOOTH_MAX:-50.0}"
LAMBDA_REG_WEIGHT="${LAMBDA_REG_WEIGHT:-0.0001}"

# learnable CDF-G/h 参数：必须和训练端 argument.py 对齐
if [[ "${NETWORK_ARCH}" == "pointnet2" ]]; then
  USE_LEARNED_CDF_CONSTRAINTS="${USE_LEARNED_CDF_CONSTRAINTS:-false}"
else
  USE_LEARNED_CDF_CONSTRAINTS="${USE_LEARNED_CDF_CONSTRAINTS:-false}"
fi
CDF_L_K="${CDF_L_K:-0.33}"
CDF_R_EGO="${CDF_R_EGO:-0.31}"
CDF_SENSE_RANGE="${CDF_SENSE_RANGE:-3.0}"
CDF_ALPHA_INIT="${CDF_ALPHA_INIT:-0.5}"
CDF_ALPHA_MIN="${CDF_ALPHA_MIN:-0.10}"
CDF_ALPHA_MAX="${CDF_ALPHA_MAX:-0.55}"
LEARNABLE_CDF_ALPHA="${LEARNABLE_CDF_ALPHA:-false}"
CDF_EPSILON_INIT="${CDF_EPSILON_INIT:-0.05}"
CDF_EPSILON_MIN="${CDF_EPSILON_MIN:-0.05}"
CDF_EPSILON_MAX="${CDF_EPSILON_MAX:-0.20}"
LEARNABLE_CDF_EPSILON="${LEARNABLE_CDF_EPSILON:-false}"
CDF_RHO_FLOOR_INIT="${CDF_RHO_FLOOR_INIT:-0.0}"
LEARNABLE_CDF_RHO_FLOOR="${LEARNABLE_CDF_RHO_FLOOR:-false}"
CDF_MARGIN_INIT="${CDF_MARGIN_INIT:-0.0}"
LEARNABLE_CDF_MARGIN="${LEARNABLE_CDF_MARGIN:-false}"
CDF_VALID_POINT_ABS_MAX="${CDF_VALID_POINT_ABS_MAX:-50.0}"
CDF_PADDING_VALUE="${CDF_PADDING_VALUE:-99.0}"
if [[ "${NETWORK_ARCH}" == "pointnet2" ]]; then
  LAMBDA_GH="${LAMBDA_GH:-0.0}"
else
  LAMBDA_GH="${LAMBDA_GH:-0.001}"
fi

# 默认评估完整 5 个 demo 数量 × 3 个 dataset seed × 3 个 train seed = 45 个模型。
# 想少跑可以覆盖，例如：
#   DEMO_LIST_STR="48" DSEED_LIST_STR="0" TRAIN_SEED_LIST_STR="0 1 2" bash run_eval_sweep.sh
DEMO_LIST_STR="${DEMO_LIST_STR:-48 32 16 8 2}"
DSEED_LIST_STR="${DSEED_LIST_STR:-0 1 2}"
TRAIN_SEED_LIST_STR="${TRAIN_SEED_LIST_STR:-0 1 2}"
read -r -a DEMO_LIST <<< "${DEMO_LIST_STR}"
read -r -a DSEED_LIST <<< "${DSEED_LIST_STR}"
read -r -a TRAIN_SEED_LIST <<< "${TRAIN_SEED_LIST_STR}"

# 重新跑完整实验时删除旧 CSV；续跑时设置 RESET_CSV=false。
RESET_CSV="${RESET_CSV:-true}"
if [[ "${RESET_CSV}" == "true" ]]; then
  rm -f "${CSV_PATH}"
fi
mkdir -p "${TRAJ_DIR}"

cleanup_ros() {
  rosnode cleanup -y >/dev/null 2>&1 || true
}
trap cleanup_ros EXIT

cat <<EOF2
============================================================
DensityNet 批量评测开始
LAUNCH                  = ${LAUNCH_PKG} ${LAUNCH_FILE}
NETWORK_ARCH            = ${NETWORK_ARCH}
MODEL_ROOT              = ${MODEL_ROOT}
MODEL_SUBDIR_PREFIX     = ${MODEL_SUBDIR_PREFIX}
MODEL_FILE_TEMPLATE     = ${MODEL_FILE_TEMPLATE:-<auto>}
CSV_PATH                = ${CSV_PATH}
TRAJ_DIR                = ${TRAJ_DIR}
DEMO_LIST               = ${DEMO_LIST_STR}
DSEED_LIST              = ${DSEED_LIST_STR}
TRAIN_SEED_LIST         = ${TRAIN_SEED_LIST_STR}
NUM_EVAL_EPISODES       = ${NUM_EVAL_EPISODES}
TARGET_MODE             = ${TARGET_MODE}
FIXED_TARGET            = (${FIXED_TARGET_X}, ${FIXED_TARGET_Y})
ABLATION                = ${ABLATION}
RUNTIME_QP_MODE         = ${RUNTIME_QP_MODE}
ENABLE_TEST_TIME_CDFQP  = ${ENABLE_TEST_TIME_CDFQP}
ANALYTIC_CDFQP          = points=${ANALYTIC_CDFQP_NUM_POINTS}, padding=${ANALYTIC_CDFQP_PADDING_VALUE}, fallback=${ANALYTIC_CDFQP_FALLBACK_TO_NOMINAL}
GRAPH_K/HIDDEN/LAMBDA   = ${GRAPH_K}/${HIDDEN_DIM}/${LAMBDA_SMOOTH}
POINTNET2               = max_points=${POINTNET2_MAX_POINTS}, sa1=(${POINTNET2_NPOINT1},${POINTNET2_RADIUS1},${POINTNET2_NSAMPLE1}), sa2=(${POINTNET2_NPOINT2},${POINTNET2_RADIUS2},${POINTNET2_NSAMPLE2})
QP box/normalize/fail   = ${USE_QP_BOX_CONSTRAINTS}/${QP_NORMALIZE_CONSTRAINTS}/${QP_FAIL_MODE}
LEARNED_CDF             = ${USE_LEARNED_CDF_CONSTRAINTS}, alpha=[${CDF_ALPHA_MIN},${CDF_ALPHA_INIT},${CDF_ALPHA_MAX}], eps=[${CDF_EPSILON_MIN},${CDF_EPSILON_INIT},${CDF_EPSILON_MAX}]
============================================================
EOF2

resolve_model_file() {
  local n="$1"
  local ds="$2"
  local ts="$3"
  local model_file=""

  if [[ -n "${MODEL_FILE_TEMPLATE}" ]]; then
    model_file="${MODEL_FILE_TEMPLATE}"
    model_file="${model_file//\{MODEL_ROOT\}/${MODEL_ROOT}}"
    model_file="${model_file//\{N\}/${n}}"
    model_file="${model_file//\{DS\}/${ds}}"
    model_file="${model_file//\{TS\}/${ts}}"
    model_file="${model_file//\{NETWORK_ARCH\}/${NETWORK_ARCH}}"
    model_file="${model_file//\{ABLATION\}/${ABLATION}}"
  else
    model_file="${MODEL_ROOT}/${MODEL_SUBDIR_PREFIX}-demo${n}-dseed${ds}-seed${ts}/model_best_parametric_bc.pt"
  fi

  if [[ -f "${model_file}" ]]; then
    echo "${model_file}"
    return 0
  fi

  # 兼容 main.py 默认保存的 timestamp 目录，例如：
  # ...arch-pointnet2-ablation-no_safety-demo48-dseed0-seed0-2026.../model_best_parametric_bc.pt
  local candidates=()
  mapfile -t candidates < <(
    find "${MODEL_ROOT}" -maxdepth "${MODEL_FIND_MAXDEPTH}" -type f \
      -path "*demo${n}-dseed${ds}-seed${ts}*/model_best_parametric_bc.pt" 2>/dev/null | sort
  )

  if [[ "${#candidates[@]}" -gt 0 ]]; then
    if [[ "${#candidates[@]}" -gt 1 ]]; then
      echo "[WARN] 找到多个候选模型，默认使用第一个:" >&2
      printf '  %s\n' "${candidates[@]}" >&2
    fi
    echo "${candidates[0]}"
    return 0
  fi

  echo "${model_file}"
}

for N in "${DEMO_LIST[@]}"; do
  for DS in "${DSEED_LIST[@]}"; do
    for TS in "${TRAIN_SEED_LIST[@]}"; do
      MODEL_FILE="$(resolve_model_file "${N}" "${DS}" "${TS}")"
      MODEL_DIR="$(dirname "${MODEL_FILE}")"

      echo ""
      echo "============================================================"
      echo "Evaluating: num_demos=${N}, dseed=${DS}, seed=${TS}"
      echo "MODEL_FILE=${MODEL_FILE}"
      echo "============================================================"

      if [[ ! -f "${MODEL_FILE}" ]]; then
        echo "[WARN] 找不到模型文件，跳过: ${MODEL_FILE}"
        continue
      fi

      roslaunch "${LAUNCH_PKG}" "${LAUNCH_FILE}" \
        model_path:="${MODEL_FILE}" \
        output_csv:="${CSV_PATH}" \
        trajectory_save_dir:="${TRAJ_DIR}" \
        num_demos:="${N}" \
        demo_seed:="${DS}" \
        train_seed:="${TS}" \
        num_eval_episodes:="${NUM_EVAL_EPISODES}" \
        test_target_seed:="${TEST_TARGET_SEED}" \
        target_mode:="${TARGET_MODE}" \
        fixed_target_x:="${FIXED_TARGET_X}" \
        fixed_target_y:="${FIXED_TARGET_Y}" \
        target_x_min:="${TARGET_X_MIN}" \
        target_x_max:="${TARGET_X_MAX}" \
        target_y_min:="${TARGET_Y_MIN}" \
        target_y_max:="${TARGET_Y_MAX}" \
        max_episode_time:="${MAX_EPISODE_TIME}" \
        hold_before_episode:="${HOLD_BEFORE_EPISODE}" \
        goal_radius:="${GOAL_RADIUS}" \
        terminate_on_collision:="${TERMINATE_ON_COLLISION}" \
        state_dim:="${STATE_DIM}" \
        hidden_dim:="${HIDDEN_DIM}" \
        graph_k:="${GRAPH_K}" \
        lambda_smooth:="${LAMBDA_SMOOTH}" \
        qp_limit:="${QP_LIMIT}" \
        network_arch:="${NETWORK_ARCH}" \
        ablation:="${ABLATION}" \
        nominal_speed:="${NOMINAL_SPEED}" \
        pointnet2_max_points:="${POINTNET2_MAX_POINTS}" \
        pointnet2_npoint1:="${POINTNET2_NPOINT1}" \
        pointnet2_radius1:="${POINTNET2_RADIUS1}" \
        pointnet2_nsample1:="${POINTNET2_NSAMPLE1}" \
        pointnet2_npoint2:="${POINTNET2_NPOINT2}" \
        pointnet2_radius2:="${POINTNET2_RADIUS2}" \
        pointnet2_nsample2:="${POINTNET2_NSAMPLE2}" \
        pointnet2_padding_value:="${POINTNET2_PADDING_VALUE}" \
        runtime_qp_mode:="${RUNTIME_QP_MODE}" \
        enable_test_time_cdfqp:="${ENABLE_TEST_TIME_CDFQP}" \
        analytic_cdfqp_num_points:="${ANALYTIC_CDFQP_NUM_POINTS}" \
        analytic_cdfqp_padding_value:="${ANALYTIC_CDFQP_PADDING_VALUE}" \
        analytic_cdfqp_fallback_to_nominal:="${ANALYTIC_CDFQP_FALLBACK_TO_NOMINAL}" \
        use_qp_box_constraints:="${USE_QP_BOX_CONSTRAINTS}" \
        qp_jitter:="${QP_JITTER}" \
        qp_normalize_constraints:="${QP_NORMALIZE_CONSTRAINTS}" \
        qp_constraint_scale_floor:="${QP_CONSTRAINT_SCALE_FLOOR}" \
        qp_box_eps:="${QP_BOX_EPS}" \
        qp_max_iter:="${QP_MAX_ITER}" \
        qp_eps:="${QP_EPS}" \
        qp_not_improved_lim:="${QP_NOT_IMPROVED_LIM}" \
        qp_fail_mode:="${QP_FAIL_MODE}" \
        qp_debug_max_print:="${QP_DEBUG_MAX_PRINT}" \
        qp_check_invalid_constraints:="${QP_CHECK_INVALID_CONSTRAINTS}" \
        qp_invalid_g_norm_eps:="${QP_INVALID_G_NORM_EPS}" \
        qp_invalid_h_eps:="${QP_INVALID_H_EPS}" \
        qp_invalid_constraint_mode:="${QP_INVALID_CONSTRAINT_MODE}" \
        qp_invalid_debug_max_print:="${QP_INVALID_DEBUG_MAX_PRINT}" \
        qp_sanitize_redundant_constraints:="${QP_SANITIZE_REDUNDANT_CONSTRAINTS}" \
        qp_redundant_constraint_h:="${QP_REDUNDANT_CONSTRAINT_H}" \
        qp_verify_solution:="${QP_VERIFY_SOLUTION}" \
        qp_solution_violation_tol:="${QP_SOLUTION_VIOLATION_TOL}" \
        qp_solution_debug_max_print:="${QP_SOLUTION_DEBUG_MAX_PRINT}" \
        qp_suppress_qpth_warnings:="${QP_SUPPRESS_QPTH_WARNINGS}" \
        qpth_fail_fallback_to_jax:="${QPTH_FAIL_FALLBACK_TO_JAX}" \
        learnable_lambda_smooth:="${LEARNABLE_LAMBDA_SMOOTH}" \
        lambda_smooth_min:="${LAMBDA_SMOOTH_MIN}" \
        lambda_smooth_max:="${LAMBDA_SMOOTH_MAX}" \
        lambda_reg_weight:="${LAMBDA_REG_WEIGHT}" \
        use_learned_cdf_constraints:="${USE_LEARNED_CDF_CONSTRAINTS}" \
        cdf_l_k:="${CDF_L_K}" \
        cdf_r_ego:="${CDF_R_EGO}" \
        cdf_sense_range:="${CDF_SENSE_RANGE}" \
        cdf_alpha_init:="${CDF_ALPHA_INIT}" \
        cdf_alpha_min:="${CDF_ALPHA_MIN}" \
        cdf_alpha_max:="${CDF_ALPHA_MAX}" \
        learnable_cdf_alpha:="${LEARNABLE_CDF_ALPHA}" \
        cdf_epsilon_init:="${CDF_EPSILON_INIT}" \
        cdf_epsilon_min:="${CDF_EPSILON_MIN}" \
        cdf_epsilon_max:="${CDF_EPSILON_MAX}" \
        learnable_cdf_epsilon:="${LEARNABLE_CDF_EPSILON}" \
        cdf_rho_floor_init:="${CDF_RHO_FLOOR_INIT}" \
        learnable_cdf_rho_floor:="${LEARNABLE_CDF_RHO_FLOOR}" \
        cdf_margin_init:="${CDF_MARGIN_INIT}" \
        learnable_cdf_margin:="${LEARNABLE_CDF_MARGIN}" \
        cdf_valid_point_abs_max:="${CDF_VALID_POINT_ABS_MAX}" \
        cdf_padding_value:="${CDF_PADDING_VALUE}" \
        lambda_gh:="${LAMBDA_GH}" \
        use_rviz:="${USE_RVIZ}"

      RET=$?
      echo "[INFO] roslaunch exited with code ${RET} for demo=${N}, dseed=${DS}, seed=${TS}"
      echo "[OK] Finished num_demos=${N}, dseed=${DS}, seed=${TS}"

      cleanup_ros
      sleep 3
    done
  done
done

echo "============================================================"
echo "所有评测结束。结果 CSV: ${CSV_PATH}"
echo "============================================================"