# GLM-4.7-Flash 这一轮实验的环境变量。每开一个新终端先：
#
#     cd /home/ubuntu/p2p-framework && source glm47flash/use_plan.sh petals_cap8
#
# （use_plan.sh 会先 source 这个文件。）与 qwen3-30b/env.sh 的区别：仓库、权重目录、清单不同；
# Qwen 的 task 选择器（128 专家）对 GLM（64 专家）不适用，LR_MODEL 置空。

export WORKDIR=/home/ubuntu/p2p-framework                 # 控制机与 15 台节点上的代码目录
export NODE_PY=/home/ubuntu/miniforge3/envs/moe/bin/python  # 节点上跑 torch 的解释器（只要 torch + safetensors，不要 transformers）
export REPO=zai-org/GLM-4.7-Flash
export WEIGHTS=$WORKDIR/weights-glm47                     # 与 Qwen 的 weights-q3-30b 分开放，换回 Qwen 不用重下
export PLAN=glm47flash/plan_petals_cap8.json              # Petals 的分配：2 条链、12 台、专家全装
export TASKS=gsm8k=1,mbpp=1,no_robots=1
export LR_MODEL=                                          # GLM 的选择器还没训；空 = 动态模式退回直方图识别
export DUMP_FRONT=results/glm47_front.jsonl

if [ -z "${ADVERTISE:-}" ]; then
  echo "⚠ 还要 export ADVERTISE=<控制机对节点可见的IP>"
fi
[ -f task/hosts.txt ] || echo "⚠ $(pwd)/task/hosts.txt 不存在"
echo "WORKDIR=$WORKDIR  WEIGHTS=$WEIGHTS  PLAN=$PLAN  REPO=$REPO"
