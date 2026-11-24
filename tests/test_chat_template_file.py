"""对话模板单独存成 chat_template.jinja 的 checkpoint（GLM-4.7-Flash）。

tokenizer_config.json 里没有 chat_template 键，模板在旁边的 chat_template.jinja 里。
`tests/data/glm47_chat_template.jinja` 是 zai-org/GLM-4.7-Flash 的模板（2026-11 取的）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("jinja2")

from p2pmoe.runtime.text import TextIO, Tokenizer, _read_chat_template_file  # noqa: E402

TPL = (ROOT / "tests" / "data" / "glm47_chat_template.jinja").read_text(encoding="utf-8")


def _tok() -> Tokenizer:
    return Tokenizer(None, config={"chat_template": TPL})


def test_template_file_is_found(tmp_path):
    assert _read_chat_template_file(tmp_path) is None
    (tmp_path / "chat_template.jinja").write_text(TPL, encoding="utf-8")
    assert _read_chat_template_file(tmp_path) == TPL
    (tmp_path / "chat_template.jinja").unlink()
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "x"}))
    assert _read_chat_template_file(tmp_path) == "x"


def test_glm_template_renders_user_turn():
    out = _tok().apply_chat_template([{"role": "user", "content": "1+1=?"}])
    assert out.startswith("[gMASK]<sop>") and "<|user|>1+1=?" in out
    assert out.endswith("<|assistant|><think>")                 # 模板默认：思考模式
    off = _tok().apply_chat_template([{"role": "user", "content": "1+1=?"}],
                                     enable_thinking=False)
    assert off.endswith("<|assistant|></think>")                # 非思考模式（verify 的统计口径）


def test_tojson_with_explicit_ensure_ascii_does_not_crash():
    """模板里写的是 tojson(ensure_ascii=False)：以前的过滤器会报「ensure_ascii 给了两次」。"""
    out = _tok().apply_chat_template([{"role": "user", "content": "hi"}],
                                     tools=[{"name": "查天气", "parameters": {}}])
    assert '"name": "查天气"' in out


def test_body_split_works_with_glm_template():
    """encode_body 靠 sentinel 切出前缀 / 后缀：GLM 的模板必须是「前缀 + 内容 + 后缀」。"""
    io = TextIO(tok=_tok(), chat=True, template_kw={"enable_thinking": False})
    render = lambda c: io.tok.apply_chat_template(  # noqa: E731
        [{"role": "user", "content": c}], add_generation_prompt=True, **io.template_kw)
    pre, post = render("@@S@@").split("@@S@@", 1)
    assert render("Problem: 2+3") == pre + "Problem: 2+3" + post


def test_from_model_dir_picks_up_the_jinja(tmp_path):
    tk = pytest.importorskip("tokenizers")
    from tokenizers import models

    t = tk.Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    t.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|endoftext|>"}))
    (tmp_path / "chat_template.jinja").write_text(TPL, encoding="utf-8")
    tok = Tokenizer.from_model_dir(tmp_path)
    assert tok.chat_template == TPL
