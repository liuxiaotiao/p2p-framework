# GLM-4.7-Flash 这一轮实验的环境变量。每开一个新终端先：
#
#     cd /home/ubuntu/p2p-framework && source glm47flash/use_plan.sh petals_cap8
#
# （use_plan.sh 会先 source 这个文件。）与 qwen3-30b/env.sh 的区别：仓库、权重目录、清单不同；
# Qwen 的 task 选择器（128 专家）对 GLM（64 专家）不适用，LR_MODEL 指向 GLM 自己的选择器。

export WORKDIR=/home/ubuntu/p2p-framework                 # 控制机与 15 台节点上的代码目录
export NODE_PY=/home/ubuntu/miniforge3/envs/moe/bin/python  # 节点上跑 torch 的解释器（只要 torch + safetensors，不要 transformers）
export REPO=zai-org/GLM-4.7-Flash
export WEIGHTS=$WORKDIR/weights-glm47                     # 与 Qwen 的 weights-q3-30b 分开放，换回 Qwen 不用重下
export PLAN=glm47flash/plan_petals_cap8.json              # Petals 的分配：2 条链、12 台、专家全装
export TASKS=gsm8k=1,mbpp=1,no_robots=1
export PROFILE=task_glm47/profile_prefill.json            # 激活画像（层号 2–47 = MoE 层；动态模式的直方图识别从它来）
# task 选择器：GLM 自己的（Qwen 的 128 专家对不上）。看前 2 个 MoE 层（checkpoint 第 1、2 层 = 规划器第 2、3 层），
# 前段要 L₀ ≥ 3。特征 = 每层平均门控权重 wsum/count，按**整条 prompt** 的 token 数（不只 body），
# 所以 prompt 必须与 verify 采集时同一套模板（非思考模式 enable_thinking=False）。
# 文件不在就置空 —— 动态模式退回直方图识别（PROFILE）
export LR_MODEL=selector/task_clf_glm47flash_L1L2_meanw.npz
[ -f "$LR_MODEL" ] || { echo "⚠ 没有 $LR_MODEL —— 识别退回直方图"; LR_MODEL=; }
export DUMP_FRONT=results/glm47_front.jsonl

if [ -z "${ADVERTISE:-}" ]; then
  echo "⚠ 还要 export ADVERTISE=<控制机对节点可见的IP>"
fi
[ -f task/hosts.txt ] || echo "⚠ $(pwd)/task/hosts.txt 不存在"
echo "WORKDIR=$WORKDIR  WEIGHTS=$WEIGHTS  PLAN=$PLAN  REPO=$REPO"
