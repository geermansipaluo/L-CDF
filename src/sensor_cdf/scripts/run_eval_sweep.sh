#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DensityNet 批量闭环评测脚本
# 模型目录约定：
#   ${MODEL_ROOT}/DensityNet-demo${N}-dseed${S}-seed${S}/model_best_parametric_bc.pt
# 其中：
#   N ∈ {2, 8, 16, 32, 48}
#   S ∈ {0, 1, 2}
# ============================================================

PKG="sensor_cdf"
NODE="traj_eval_sweep.py"

MODEL_ROOT="/home/guo/L-CDF/src/sensor_cdf/scripts/save_models"
CSV_PATH="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics.csv"
TRAJ_DIR="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories"

NUM_EVAL_EPISODES=10
TEST_TARGET_SEED=2026
MAX_EPISODE_TIME=80.0

DEMO_LIST=(2 8 16 32 48)
SEED_LIST=(0 1 2)

# 如果想续写旧 CSV，把下面 rm 注释掉。
rm -f "${CSV_PATH}"
mkdir -p "${TRAJ_DIR}"

echo "============================================================"
echo "DensityNet 批量评测开始"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "CSV_PATH=${CSV_PATH}"
echo "TRAJ_DIR=${TRAJ_DIR}"
echo "NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES}"
echo "TEST_TARGET_SEED=${TEST_TARGET_SEED}"
echo "============================================================"

for N in "${DEMO_LIST[@]}"; do
  for S in "${SEED_LIST[@]}"; do
    MODEL_DIR="${MODEL_ROOT}/DensityNet-demo${N}-dseed${S}-seed${S}"
    MODEL_FILE="${MODEL_DIR}/model_best_parametric_bc.pt"

    echo ""
    echo "============================================================"
    echo "Evaluating: num_demos=${N}, seed=${S}"
    echo "MODEL_FILE=${MODEL_FILE}"
    echo "============================================================"

    if [[ ! -f "${MODEL_FILE}" ]]; then
      echo "[WARN] 找不到模型文件，跳过: ${MODEL_FILE}"
      continue
    fi

    rosrun "${PKG}" "${NODE}" \
      _model_path:="${MODEL_FILE}" \
      _output_csv:="${CSV_PATH}" \
      _trajectory_save_dir:="${TRAJ_DIR}" \
      _num_demos:="${N}" \
      _demo_seed:="${S}" \
      _train_seed:="${S}" \
      _num_eval_episodes:="${NUM_EVAL_EPISODES}" \
      _test_target_seed:="${TEST_TARGET_SEED}" \
      _max_episode_time:="${MAX_EPISODE_TIME}" \
      _target_x_min:=14.0 \
      _target_x_max:=16.0 \
      _target_y_min:=-2.0 \
      _target_y_max:=2.0 \
      _goal_radius:=0.4

    echo "[OK] Finished num_demos=${N}, seed=${S}"
    sleep 2
  done
done

echo "============================================================"
echo "所有评测结束。结果 CSV: ${CSV_PATH}"
echo "============================================================"
