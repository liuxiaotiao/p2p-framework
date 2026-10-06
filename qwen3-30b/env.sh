# Qwen3-30B-A3B + task 选择器这一轮实验的环境变量。每开一个新终端先：
#
#     cd /home/ubuntu/p2p-framework && source qwen3-30b/env.sh
#
# 为什么要有这个文件：deploy_15.sh 里的 WORKDIR、WEIGHTS 等只在脚本内部有默认值，
# 终端里没有 —— 在终端里写 $WORKDIR/... 会展开成 /...（已经踩过两次）。

export WORKDIR=/home/ubuntu/p2p-framework                 # 控制机与 15 台节点上的代码目录
export NODE_PY=/home/ubuntu/anaconda3/envs/moe/bin/python  # 节点上跑 torch 的解释器（绝对路径）
export REPO=Qwen/Qwen3-30B-A3B
export WEIGHTS=$WORKDIR/weights-q3-30b                    # 与 Qwen3-Next 的权重分开放
export PLAN=qwen3-30b/plan_es90_fill.json                 # 15 台全上、5 条通道
export PROFILE=qwen3-30b/profile_es90.json                # 后段驻留集（expert_sets_90 并集）
export TASKS=gsm8k=1,mbpp=1,no_robots=1
export LR_MODEL=selector/selector_body_L8_params.npz      # task 选择器（前 8 层，body token）
export PROMPTS=qwen3-30b/sel_cases.jsonl                  # 训练样本抽的测试集，带样本编号
export DUMP_FRONT=results/sel_front.jsonl                 # 前段逐层计数落盘，给 lr_tool check
export PROBE=Delta_30B/probe_30b_ref20/prefill_Qwen3-30B-A3B   # 解压 Delta_30B/delta_30b.tgz 得到

# ADVERTISE（控制机对节点可见的 IP）每台控制机不同，这里不写死
if [ -z "${ADVERTISE:-}" ]; then
  echo "⚠ 还要 export ADVERTISE=<控制机对节点可见的IP>"
fi
[ -f task/hosts.txt ] || echo "⚠ $(pwd)/task/hosts.txt 不存在 —— 从旧目录拷过来：cp /home/ubuntu/P2P_MoE/task/hosts.txt task/"
[ -d "$PROBE" ] || echo "⚠ 还没解压 probe 数据：tar xzf Delta_30B/delta_30b.tgz -C Delta_30B"
echo "WORKDIR=$WORKDIR  WEIGHTS=$WEIGHTS  PLAN=$PLAN"
