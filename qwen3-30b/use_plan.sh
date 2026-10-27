# 新开终端后一条命令设好全部环境：env.sh + 指定清单 + 与清单对应的 hosts。
#
#     cd /home/ubuntu/p2p-framework && conda activate moe
#     source qwen3-30b/use_plan.sh vllm          # 或 full_2ch / doc_v2 / doc_v2_5ch / full / doc_l8 / es90_fill
#     source qwen3-30b/use_plan.sh qwen3-30b/某个清单.json   # 也可以直接给路径
#
# hosts 不再手写排除哪几台：从 task/hosts.txt 里只留清单用到的节点，写到
# task/hosts_<名字>.txt —— 清单与 hosts 永远对得上（之前 vLLM 方案配了完整模型那份 hosts，
# N12 漏拉、N13 报「清单里没有」就是这么来的）。
#
# 必须用 source 执行（要改的是当前终端的环境变量）。

_up_name=${1:-}
if [ -z "$_up_name" ]; then
  echo "用法：source qwen3-30b/use_plan.sh <vllm|full_2ch|doc_v2|doc_v2_5ch|full|doc_l8|es90_fill|清单路径>"
  return 1 2>/dev/null || exit 1
fi
case "$_up_name" in
  *.json) _up_plan=$_up_name; _up_tag=$(basename "$_up_name" .json); _up_tag=${_up_tag#plan_} ;;
  *)      _up_plan=qwen3-30b/plan_${_up_name}.json; _up_tag=$_up_name ;;
esac
if [ ! -f "$_up_plan" ]; then
  echo "✗ 没有 $_up_plan（当前目录 $(pwd)；要在 p2p-framework 目录下 source）"
  return 1 2>/dev/null || exit 1
fi

source qwen3-30b/env.sh >/dev/null
export ADVERTISE=${ADVERTISE:-192.168.3.4}
export PLAN=$_up_plan
export HOSTS_ALL=task/hosts.txt                  # 停 agent 时用全部 15 台：HOSTS=$HOSTS_ALL bash ./deploy_15.sh stop
export HOSTS=task/hosts_${_up_tag}.txt

python3 - "$PLAN" "$HOSTS_ALL" "$HOSTS" <<'PYEOF'
import json, re, sys
plan, src, dst = sys.argv[1:]
need = {n["node"] for n in json.load(open(plan, encoding="utf-8"))["nodes"]}
keep, have = [], set()
for line in open(src, encoding="utf-8"):
    body = re.sub(r"#.*", "", line).split()
    if len(body) >= 2 and body[0] in need:
        keep.append(line if line.endswith("\n") else line + "\n")
        have.add(body[0])
open(dst, "w", encoding="utf-8").writelines(keep)
miss = sorted(need - have)
print(f"  清单用到 {len(need)} 台 → {dst}（{len(keep)} 台）" + (f"  ✗ hosts.txt 里没有 {miss}" if miss else ""))
PYEOF

_up_ok=1
_up_chk() { if [ "$2" = "$3" ]; then echo "  ✓ $1=$2"; else echo "  ✗ $1=$2（应为 $3）"; _up_ok=0; fi; }
_up_chk WORKDIR "$WORKDIR" /home/ubuntu/p2p-framework
_up_chk NODE_PY "$NODE_PY" /home/ubuntu/anaconda3/envs/moe/bin/python
_up_chk REPO    "$REPO"    Qwen/Qwen3-30B-A3B
_up_chk WEIGHTS "$WEIGHTS" /home/ubuntu/p2p-framework/weights-q3-30b
echo "  PLAN=$PLAN  HOSTS=$HOSTS  ADVERTISE=$ADVERTISE"
[ "$_up_ok" = 1 ] || echo "  ⚠ 上面有 ✗ —— 先别 fetch / sync"
mkdir -p results
unset _up_name _up_plan _up_tag _up_ok
unset -f _up_chk
