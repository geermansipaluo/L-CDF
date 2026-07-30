#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GP-CBF num_rays sweep
# 每个 num_rays 跑 10 轮，并把结果合并到一个 CSV。
# 说明：
#   1) 不直接修改你的原始 launch 文件；
#   2) 每次生成一个 /tmp 临时 launch，把 perfect_lidar_bridge 的 num_rays 替换成当前值；
#   3) 每个 num_rays 先输出临时 CSV；
#   4) 脚本把临时 CSV 合并成 FINAL_CSV，并额外添加 num_rays 列。
# ============================================================

# 你的 GP-CBF launch 文件路径
BASE_LAUNCH="/home/guo/L-CDF/src/sensor_cdf/launch/traj_eval_sweep_gpcbf.launch"

# 最终汇总 CSV。脚本每次运行会覆盖这个文件。
FINAL_CSV="/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics_gpcbf.csv"

# 每个点云分辨率测试 10 轮
NUM_EVAL_EPISODES=10

# 要测试的输入 ray 数量
RAYS_LIST=(256 128 64 32 8)

# 目标设置。这里保持和你当前 launch 默认一致；需要 y=0 时改成 fixed_y0。
TARGET_MODE="fixed"
FIXED_TARGET_X="15.0"
FIXED_TARGET_Y="-1.0"
TEST_TARGET_SEED="2026"

# GP-CBF 参数。一般先保持一致，只扫 num_rays。
GP_LENGTH_SCALE="0.6"
GP_NOISE="0.01"
GP_H_SHIFT="0.6"
CBF_GAMMA="0.8"
NOMINAL_SPEED="1.0"

TMP_ROOT="/tmp/gpcbf_num_rays_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TMP_ROOT}"
mkdir -p "$(dirname "${FINAL_CSV}")"
rm -f "${FINAL_CSV}"

if [[ ! -f "${BASE_LAUNCH}" ]]; then
    echo "[ERROR] BASE_LAUNCH 不存在: ${BASE_LAUNCH}"
    exit 1
fi

cleanup() {
    echo "[INFO] 临时文件保留在: ${TMP_ROOT}"
}
trap cleanup EXIT

for NUM_RAYS in "${RAYS_LIST[@]}"; do
    echo "============================================================"
    echo "[RUN] GP-CBF num_rays=${NUM_RAYS}, episodes=${NUM_EVAL_EPISODES}"
    echo "============================================================"

    TMP_LAUNCH="${TMP_ROOT}/traj_eval_sweep_gpcbf_numrays_${NUM_RAYS}.launch"
    TMP_CSV="${TMP_ROOT}/eval_metrics_gpcbf_numrays_${NUM_RAYS}.csv"

    # 生成临时 launch：只替换 perfect_lidar_bridge 中的 num_rays 参数。
    python3 - "${BASE_LAUNCH}" "${TMP_LAUNCH}" "${NUM_RAYS}" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
num_rays = sys.argv[3]
text = src.read_text()

pattern = r'(<param\s+name="num_rays"\s+type="int"\s+value=")\d+("\s*/>)'
new_text, n = re.subn(pattern, rf'\g<1>{num_rays}\g<2>', text)

if n != 1:
    raise RuntimeError(
        f"期望只替换 1 个 num_rays 参数，但实际替换了 {n} 个。请检查 launch 文件。"
    )

dst.write_text(new_text)
print(f"[INFO] 临时 launch 已生成: {dst}")
PY

    rm -f "${TMP_CSV}"

    # 运行当前 num_rays 的 10 轮评测。
    # 注意：roslaunch 支持直接传入 launch 文件绝对路径。
    roslaunch "${TMP_LAUNCH}" \
        output_csv:="${TMP_CSV}" \
        num_eval_episodes:="${NUM_EVAL_EPISODES}" \
        target_mode:="${TARGET_MODE}" \
        fixed_target_x:="${FIXED_TARGET_X}" \
        fixed_target_y:="${FIXED_TARGET_Y}" \
        test_target_seed:="${TEST_TARGET_SEED}" \
        gp_length_scale:="${GP_LENGTH_SCALE}" \
        gp_noise:="${GP_NOISE}" \
        gp_h_shift:="${GP_H_SHIFT}" \
        cbf_gamma:="${CBF_GAMMA}" \
        nominal_speed:="${NOMINAL_SPEED}"

    if [[ ! -f "${TMP_CSV}" ]]; then
        echo "[ERROR] 当前 num_rays=${NUM_RAYS} 没有生成 CSV: ${TMP_CSV}"
        exit 1
    fi

    # 把临时 CSV 合并到 FINAL_CSV，并增加 num_rays 列。
    python3 - "${TMP_CSV}" "${FINAL_CSV}" "${NUM_RAYS}" <<'PY'
import csv
import os
import sys
from pathlib import Path

tmp_csv = Path(sys.argv[1])
final_csv = Path(sys.argv[2])
num_rays = sys.argv[3]

with tmp_csv.open("r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    source_fields = reader.fieldnames or []

if not rows:
    raise RuntimeError(f"临时 CSV 为空: {tmp_csv}")

for row in rows:
    row["num_rays"] = num_rays

new_fields = ["num_rays"] + [k for k in source_fields if k != "num_rays"]

file_exists = final_csv.exists() and final_csv.stat().st_size > 0
if file_exists:
    with final_csv.open("r", newline="") as f:
        existing_reader = csv.reader(f)
        existing_fields = next(existing_reader)
    fieldnames = existing_fields
else:
    fieldnames = new_fields

with final_csv.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

print(f"[INFO] 已合并 num_rays={num_rays} 到 {final_csv}")
PY

    echo "[DONE] num_rays=${NUM_RAYS} 完成"
    sleep 2

done

echo "============================================================"
echo "[ALL DONE] GP-CBF num_rays sweep 完成"
echo "[CSV] ${FINAL_CSV}"
echo "============================================================"
