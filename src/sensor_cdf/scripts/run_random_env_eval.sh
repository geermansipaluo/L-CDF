#!/usr/bin/env bash
set -eo pipefail

# LCDF v0.1.2: 20个可复现静态随机环境闭环测试。
# 可在命令前用同名环境变量覆盖参数，例如：
#   CDF_R_EGO=0.31 RANDOM_ENV_SEED=20260731 bash .../run_random_env_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 本仓库的PyG/qpth/JAX依赖位于cdflearn环境。可用CONDA_ENV覆盖；
# 显式设为none时不自动激活。
CONDA_ENV="${CONDA_ENV:-cdflearn}"
if [[ "${CONDA_ENV}" != "none" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到conda，无法激活依赖环境 ${CONDA_ENV}。"
    exit 1
  fi
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if ! conda activate "${CONDA_ENV}"; then
    echo "[ERROR] 无法激活conda环境 ${CONDA_ENV}。可用 CONDA_ENV=实际环境名 覆盖。"
    exit 1
  fi
fi

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "${REPO_ROOT}/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/devel/setup.bash"
else
  echo "[ERROR] 未找到 ${REPO_ROOT}/devel/setup.bash，请先在仓库根目录执行 catkin build。"
  exit 1
fi
set -u

if ! python3 -c "import torch, torch_geometric, torch_cluster, rospy, qpth, jax, jaxproxqp" >/dev/null 2>&1; then
  echo "[ERROR] 当前Python缺少评测依赖。"
  echo "当前解释器: $(command -v python3)"
  echo "请使用包含 torch_geometric、torch_cluster、qpth、jax、jaxproxqp 的环境。"
  exit 1
fi

EXPERIMENT_VERSION="${EXPERIMENT_VERSION:-v0.1.2}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${EXPERIMENT_VERSION}_${RUN_TIMESTAMP}}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/random_environment/${RUN_ID}}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/saved_models/learnable/DensityNet-demo48-dseed0-seed0/model_best_parametric_bc.pt}"

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-20}"
RANDOM_ENV_SEED="${RANDOM_ENV_SEED:-20260730}"
CDF_R_EGO="${CDF_R_EGO:-0.36}"
CDF_L_K="${CDF_L_K:-0.33}"
FIXED_TARGET_X="${FIXED_TARGET_X:-15.0}"
MAX_EPISODE_TIME="${MAX_EPISODE_TIME:-80.0}"
TERMINATE_ON_COLLISION="${TERMINATE_ON_COLLISION:-true}"
USE_RVIZ="${USE_RVIZ:-false}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] 模型不存在: ${MODEL_PATH}"
  echo "可用 MODEL_PATH=/绝对路径/model_best_parametric_bc.pt 覆盖。"
  exit 1
fi

mkdir -p "${RESULT_DIR}" "${RESULT_DIR}/legacy_trajectories"

{
  echo "experiment_version=${EXPERIMENT_VERSION}"
  echo "run_id=${RUN_ID}"
  echo "model_path=${MODEL_PATH}"
  echo "num_eval_episodes=${NUM_EVAL_EPISODES}"
  echo "random_env_seed=${RANDOM_ENV_SEED}"
  echo "start=0,0,0"
  echo "target=${FIXED_TARGET_X},0"
  echo "robot_physical_radius=0.31"
  echo "obstacle_physical_radius=0.25"
  echo "cdf_r_ego=${CDF_R_EGO}"
  echo "cdf_l_k=${CDF_L_K}"
} > "${RESULT_DIR}/launcher_settings.txt"

echo "[INFO] 开始LCDF随机环境测试"
echo "[INFO] 结果目录: ${RESULT_DIR}"
echo "[INFO] ${NUM_EVAL_EPISODES}个场景可由 seed=${RANDOM_ENV_SEED} 逐场景复现"
echo "[INFO] 物理半径=0.31m，CDF半径=${CDF_R_EGO}m"

set +e
roslaunch scene traj_eval_sweep.launch \
  model_path:="${MODEL_PATH}" \
  experiment_version:="${EXPERIMENT_VERSION}" \
  run_id:="${RUN_ID}" \
  result_dir:="${RESULT_DIR}" \
  output_csv:="${RESULT_DIR}/aggregate_compat.csv" \
  trajectory_save_dir:="${RESULT_DIR}/legacy_trajectories" \
  num_eval_episodes:="${NUM_EVAL_EPISODES}" \
  environment_mode:="random" \
  random_env_seed:="${RANDOM_ENV_SEED}" \
  random_min_obstacles:="3" \
  random_max_obstacles:="6" \
  force_path_obstacle:="true" \
  obstacle_physical_radius:="0.25" \
  collision_audit_obstacle_radius:="0.25" \
  target_mode:="fixed_y0" \
  fixed_target_x:="${FIXED_TARGET_X}" \
  fixed_target_y:="0.0" \
  max_episode_time:="${MAX_EPISODE_TIME}" \
  terminate_on_collision:="${TERMINATE_ON_COLLISION}" \
  runtime_qp_mode:="qpth" \
  ablation:="full" \
  use_learned_cdf_constraints:="true" \
  cdf_l_k:="${CDF_L_K}" \
  cdf_r_ego:="${CDF_R_EGO}" \
  qpth_fail_fallback_to_jax:="false" \
  qp_fail_mode:="fallback" \
  use_rviz:="${USE_RVIZ}" \
  2>&1 | tee "${RESULT_DIR}/run.log"
ROS_EXIT=${PIPESTATUS[0]}
set -e

echo "${ROS_EXIT}" > "${RESULT_DIR}/roslaunch_exit_code.txt"
RUN_STATUS="$(
  python3 -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('status','unknown'))" \
    "${RESULT_DIR}/config.json" 2>/dev/null || true
)"
if [[ "${ROS_EXIT}" -ne 0 || "${RUN_STATUS}" != "completed" ]]; then
  echo "[ERROR] 评测没有完整完成：roslaunch退出码=${ROS_EXIT}, 结果状态=${RUN_STATUS:-missing}"
  echo "[ERROR] 日志与已完成场景保存在 ${RESULT_DIR}"
  [[ "${ROS_EXIT}" -ne 0 ]] && exit "${ROS_EXIT}"
  exit 2
fi

echo "[OK] 测试完成。先查看: ${RESULT_DIR}/summary.md"
