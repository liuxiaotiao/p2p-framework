#!/usr/bin/env bash
# 清掉某个旧模型（默认 Qwen3.6-35B-A3B）在控制机与各节点上留下的权重与缓存。
#
#   bash ./purge_model_cache.sh            # 只列出来（默认，什么都不删）
#   bash ./purge_model_cache.sh --yes      # 真删
#   bash ./purge_model_cache.sh --local    # 只看/删本机，不 ssh 到节点
#
# 怎么认「是不是那个模型」—— **按 config.json，不按名字猜**：
#   · 目录里有 config.json，且 num_hidden_layers=40、num_experts=256（task2 统计里的
#     40 层 × 256 专家）→ 是；
#   · 没有 config.json（下到一半的），但目录名像 Qwen3.6-35B → 是；
#   · 名字像、config 却是别的模型 → **不删**，打出来让你看；
#   · config 是 48 层的（Qwen3-30B-A3B 128 专家 / Qwen3-Next 512 专家）→ 永远不碰。
#
# 删哪里：
#   · HF 缓存   $HF_HOME/hub 或 ~/.cache/huggingface/hub 下的 models--*（连同 .locks）
#   · 权重目录  $WORKDIR、$HOME、以及 EXTRA_ROOTS 里任何含 config.json 的目录
#
# 正在被 agent 打开的文件删了也不腾空间（cmd_disk 里那个「幽灵字节」）——
# 所以有进程占着的目录会跳过，并提示先 `bash ./deploy_15.sh stop force`。
#
# 环境变量（与 deploy_15.sh 同一套）：HOSTS WORKDIR SSH_USER SSH_OPTS
#   MATCH_LAYERS=40 MATCH_EXPERTS=256   换成别的模型时改这两个
#   NAME_RE='qwen3[._-]?6.*35b'         名字匹配（不区分大小写），只用于没有 config 的目录
#   EXTRA_ROOTS="/data /mnt/ssd"         额外要扫的目录（空格分隔）
#   KERNEL_CACHE=1                       顺带清 triton / torch inductor 编译缓存（与模型无关，
#                                        但会被旧模型的形状塞满；清了下次首个请求会慢一点）
set -uo pipefail

HOSTS=${HOSTS:-task/hosts.txt}
WORKDIR=${WORKDIR:-/home/ubuntu/P2P_MoE}
MATCH_LAYERS=${MATCH_LAYERS:-40}
MATCH_EXPERTS=${MATCH_EXPERTS:-256}
NAME_RE=${NAME_RE:-'qwen3[._-]?6.*35b|35b.*qwen3[._-]?6'}
EXTRA_ROOTS=${EXTRA_ROOTS:-}
KERNEL_CACHE=${KERNEL_CACHE:-0}

DO_DELETE=0
LOCAL_ONLY=0
for a in "$@"; do
  case "$a" in
    --yes) DO_DELETE=1 ;;
    --local) LOCAL_ONLY=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "不认识的参数 $a（--yes / --local / --help）"; exit 2 ;;
  esac
done

# --------------------------------------------------------------------------- #
# 每台机器上跑的那一段。本机直接跑，节点经 ssh 'bash -s' 喂进去 —— 同一份代码。
# --------------------------------------------------------------------------- #
scanner() {
cat <<'EOS'
set -uo pipefail
DO_DELETE=$1; WORKDIR=$2; ML=$3; ME=$4; NAME_RE=$5; EXTRA=$6; KC=$7
HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub
ROOTS=""
for r in "$WORKDIR" "$HOME" $EXTRA; do [ -d "$r" ] && ROOTS="$ROOTS $r"; done
PY=$(command -v python3 || command -v python || true)

# config.json → "层数 专家数"（读不出来就 "? ?"）
shape() {
  if [ -n "$PY" ]; then
    "$PY" - "$1" <<'PYX' 2>/dev/null || echo "? ?"
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("text_config") or {}
g = lambda *k: next((x for x in (c.get(i, t.get(i)) for i in k) if x is not None), "?")
print(g("num_hidden_layers"), g("num_experts", "num_local_experts", "n_routed_experts"))
PYX
  else
    l=$(grep -o '"num_hidden_layers"[^,}]*' "$1" | grep -o '[0-9]\+' | head -1)
    e=$(grep -o '"num\(_local\)\?_experts"[^,}]*' "$1" | grep -o '[0-9]\+' | head -1)
    echo "${l:-?} ${e:-?}"
  fi
}
namehit() { echo "$1" | grep -qiE "$NAME_RE"; }
size() { du -sh "$1" 2>/dev/null | cut -f1; }
# 有没有进程开着这个目录下的文件
busy() {
  local d; d=$(readlink -f "$1")
  for f in /proc/[0-9]*/fd/*; do
    t=$(readlink "$f" 2>/dev/null) || continue
    case "$t" in "$d"/*) echo "${f#/proc/}" | cut -d/ -f1; return 0 ;; esac
  done
  return 1
}

declare -A SEEN
CANDS=()
note() { printf '  %-6s %-8s %s  %s\n' "$1" "$2" "$3" "$4"; }

# 1) HF 缓存：一个仓库一个 models--org--name 目录
if [ -d "$HUB" ]; then
  for m in "$HUB"/models--*; do
    [ -d "$m" ] || continue
    cfg=$(ls "$m"/snapshots/*/config.json 2>/dev/null | head -1)
    if [ -n "$cfg" ]; then
      read -r L E < <(shape "$cfg")
      if [ "$L" = "$ML" ] && [ "$E" = "$ME" ]; then
        CANDS+=("$m"); note "删" "HF缓存" "$(size "$m")" "$m  （config: ${L}层×${E}专家）"
      elif namehit "$m"; then
        note "跳过" "HF缓存" "$(size "$m")" "$m  名字像，但 config 是 ${L}层×${E}专家 —— 不是它"
      fi
    elif namehit "$m"; then
      CANDS+=("$m"); note "删" "HF缓存" "$(size "$m")" "$m  （没有 config，按名字：多半是下到一半）"
    fi
  done
fi

# 2) 权重目录：任何含 config.json 的目录
under_cand() {   # 已经被某个候选目录包含了？
  local x
  for x in "${CANDS[@]}"; do case "$1" in "$x"|"$x"/*) return 0 ;; esac; done
  return 1
}
weighty() {      # 目录里（两层内）有权重文件，或者是空的
  [ -z "$(ls -A "$1" 2>/dev/null)" ] && return 0
  find "$1" -maxdepth 2 \( -name '*.safetensors' -o -name '*.safetensors.index.json' \
       -o -name '*.bin' -o -name '*.part' -o -name '*.incomplete' -o -name '*.tmp' \) \
       -print -quit 2>/dev/null | grep -q .
}
if [ -n "$ROOTS" ]; then
  while IFS= read -r cfg; do
    d=$(dirname "$cfg")
    case "$d" in "$HUB"/*|*/p2pmoe/*|*/tests/*|*/.git/*) continue ;; esac
    [ -n "${SEEN[$d]:-}" ] && continue; SEEN[$d]=1
    # 只认像权重目录的：有 safetensors / index，或者名字像
    if ! weighty "$d" && ! namehit "$d"; then continue; fi
    read -r L E < <(shape "$cfg")
    if [ "$L" = "$ML" ] && [ "$E" = "$ME" ]; then
      CANDS+=("$d"); note "删" "权重" "$(size "$d")" "$d  （config: ${L}层×${E}专家）"
    elif namehit "$d"; then
      note "跳过" "权重" "$(size "$d")" "$d  名字像，但 config 是 ${L}层×${E}专家 —— 不是它"
    else
      note "保留" "权重" "$(size "$d")" "$d  （${L}层×${E}专家）"
    fi
  done < <(find $ROOTS -maxdepth 5 -name config.json 2>/dev/null | sort -u)

  # 没有 config 的半截下载：名字像、且里面是权重文件（或空目录）才算 ——
  # 名字像但装的是统计结果、日志之类的，一律不碰
  while IFS= read -r d; do
    case "$d" in "$HUB"|"$HUB"/*|*/.git/*|*/.git) continue ;; esac
    [ -n "${SEEN[$d]:-}" ] && continue; SEEN[$d]=1
    [ -f "$d/config.json" ] && continue
    under_cand "$d" && continue
    if weighty "$d"; then
      CANDS+=("$d"); note "删" "半截" "$(size "$d")" "$d  （没有 config.json，按名字；里面是权重文件或为空）"
    else
      note "跳过" "其它" "$(size "$d")" "$d  名字像，但里面不是权重文件 —— 不碰"
    fi
  done < <(find $ROOTS -maxdepth 4 -type d 2>/dev/null | grep -iE "$NAME_RE" | sort -u)
fi

if [ "$KC" = 1 ]; then
  for k in "$HOME/.triton/cache" "$HOME/.cache/torch/kernels" "/tmp/torchinductor_$(id -un)"; do
    [ -d "$k" ] && { CANDS+=("$k"); note "删" "编译缓存" "$(size "$k")" "$k"; }
  done
fi

[ ${#CANDS[@]} -eq 0 ] && { echo "  （没找到）"; exit 0; }
if [ "$DO_DELETE" != 1 ]; then
  echo "  —— 只列出，没删。确认无误后加 --yes"; exit 0
fi

before=$(df -Pk "$HOME" | awk 'NR==2{print $4}')
fail=0
for d in "${CANDS[@]}"; do
  if pid=$(busy "$d"); then
    echo "  ✗ $d 正被进程 $pid 打开 —— 删了也不腾空间。先 bash ./deploy_15.sh stop force"
    fail=1; continue
  fi
  if rm -rf -- "$d" 2>/dev/null; then
    echo "  ✓ 已删 $d"
    base=$(basename "$d")
    case "$d" in "$HUB"/models--*) rm -rf -- "$HUB/.locks/$base" 2>/dev/null ;; esac
  else
    echo "  ✗ 删不掉 $d（权限？属主 $(stat -c %U "$d" 2>/dev/null)）—— 需要的话 sudo rm -rf 它"
    fail=1
  fi
done
after=$(df -Pk "$HOME" | awk 'NR==2{print $4}')
awk -v a="$after" -v b="$before" -v h="$HOME" \
  'BEGIN{printf "  腾出约 %.1f GB（%s 所在盘）\n", (a-b)/1048576, h}'
exit $fail
EOS
}

ARGS=("$DO_DELETE" "$WORKDIR" "$MATCH_LAYERS" "$MATCH_EXPERTS" "$NAME_RE" "$EXTRA_ROOTS" "$KERNEL_CACHE")
mode=$([ "$DO_DELETE" = 1 ] && echo "删除" || echo "只列出")
echo "目标：${MATCH_LAYERS} 层 × ${MATCH_EXPERTS} 专家的模型（名字匹配 /$NAME_RE/i）—— $mode"
echo "48 层的（Qwen3-30B-A3B / Qwen3-Next）不会被碰。"

printf '\n\033[1m── 本机 %s\033[0m\n' "$(hostname)"
bash -c "$(scanner)" _ "${ARGS[@]}"
rc=$?

if [ "$LOCAL_ONLY" = 0 ]; then
  [ -f "$HOSTS" ] || { echo "没有 $HOSTS —— 只处理了本机（或设 HOSTS=...）"; exit $rc; }
  d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
  ids=()
  while read -r id ip; do
    ids+=("$id:$ip")
    q=$(printf ' %q' "${ARGS[@]}")
    ( scanner | ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} \
        "${SSH_USER:+$SSH_USER@}$ip" "bash -s --$q" > "$d/$id.out" 2>&1
      echo $? > "$d/$id.rc" ) &
  done < <(awk '{sub(/#.*/,"")} NF>=2 {split($2,a,":"); print $1, a[1]}' "$HOSTS")
  wait
  for e in "${ids[@]}"; do
    id=${e%%:*}; ip=${e#*:}
    printf '\n\033[1m── %s @ %s\033[0m\n' "$id" "$ip"
    r=$(cat "$d/$id.rc" 2>/dev/null || echo 255)
    if [ "$r" = 255 ]; then echo "  ✗ ssh 连不上：$(head -1 "$d/$id.out")"; rc=1; continue; fi
    cat "$d/$id.out"
    [ "$r" != 0 ] && rc=1
  done
fi
exit $rc
