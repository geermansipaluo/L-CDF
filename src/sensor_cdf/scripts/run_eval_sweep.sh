#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# DensityNet 批量闭环评测脚本：通过 roslaunch 启动完整仿真环境
#
# 先把 dynamic_eval.launch 放到：
#   /home/guo/L-CDF/src/scene/launch/dynamic_eval.launch
# 把 traj_eval_sweep.py 放到：
#   /home/guo/L-CDF/src/sensor_cdf/scripts/traj_eval_sweep.py
#
# 然后运行：
#   cd /home/guo/L-CDF/src/sensor_cdf/scripts
#   bash run_eval_sweep_roslaunch.sh
# ============================================================

LAUNCH_PKG="scene"
LAUNCH_FILE="traj_eval_sweep.launch"

# 你之前文字里写的是 save_models，但终端输出里是 saved_models。
# 这里默认先用 save_models；如果不存在但 saved_models 存在，则自动切换。
MODEL_ROOT="${MODEL_ROOT:-/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/new_loss}"
ALT_MODEL_ROOT="/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models"

if [[ ! -d "${MODEL_ROOT}" && -d "${ALT_MODEL_ROOT}" ]]; then
  echo "[INFO] MODEL_ROOT 不存在，自动切换到: ${ALT_MODEL_ROOT}"
  MODEL_ROOT="${ALT_MODEL_ROOT}"
fi

CSV_PATH="${CSV_PATH:-/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics.csv}"
TRAJ_DIR="${TRAJ_DIR:-/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories}"

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
TEST_TARGET_SEED="${TEST_TARGET_SEED:-2026}"
MAX_EPISODE_TIME="${MAX_EPISODE_TIME:-80.0}"

DEMO_LIST=(48 32 16 8 2)
SEED_LIST=(0 1 2)

# 重新跑完整实验时删除旧 CSV；想续跑就注释掉这一行。
rm -f "${CSV_PATH}"
mkdir -p "${TRAJ_DIR}"

# 避免上一次异常退出残留 ROS/Gazebo 进程影响下一轮。
cleanup_ros() {
  rosnode cleanup -y >/dev/null 2>&1 || true
}

trap cleanup_ros EXIT

cat <<EOF2
============================================================
DensityNet 批量评测开始
LAUNCH          = ${LAUNCH_PKG} ${LAUNCH_FILE}
MODEL_ROOT      = ${MODEL_ROOT}
CSV_PATH        = ${CSV_PATH}
TRAJ_DIR        = ${TRAJ_DIR}
NUM_EVAL_EPISODES = ${NUM_EVAL_EPISODES}
TEST_TARGET_SEED  = ${TEST_TARGET_SEED}
============================================================
EOF2

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

    roslaunch "${LAUNCH_PKG}" "${LAUNCH_FILE}" \
      model_path:="${MODEL_FILE}" \
      output_csv:="${CSV_PATH}" \
      trajectory_save_dir:="${TRAJ_DIR}" \
      num_demos:="${N}" \
      demo_seed:="${S}" \
      train_seed:="${S}" \
      num_eval_episodes:="${NUM_EVAL_EPISODES}" \
      test_target_seed:="${TEST_TARGET_SEED}" \
      max_episode_time:="${MAX_EPISODE_TIME}" \
      target_x_min:=14.0 \
      target_x_max:=16.0 \
      target_y_min:=-2.0 \
      target_y_max:=2.0 \
      goal_radius:=0.4 \
      terminate_on_collision:=false \
      use_rviz:=false

    RET=$?
    echo "[INFO] roslaunch exited with code ${RET} for demo=${N}, seed=${S}"
    echo "[OK] Finished num_demos=${N}, seed=${S}"

    cleanup_ros
    sleep 3
  done
done

echo "============================================================"
echo "所有评测结束。结果 CSV: ${CSV_PATH}"
echo "============================================================"