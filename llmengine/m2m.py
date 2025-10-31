# m2m/m2m.py
import os
import argparse
from traceback import format_exc
from appPublic.registerfunction import RegisterFunction
from appPublic.worker import awaitify
from appPublic.log import debug, exception
from ahserver.serverenv import ServerEnv
from ahserver.webapp import webserver

from .fanyi_m2m import *
from .base_m2m import get_llm_class


helptext = """M2M100 翻译 API:
POST /v1/m2m
Headers:
    Content-Type: application/json

Body:
{
  "model": "m2m100_1.2B",
  "texts": "将图片中的车变为淡粉色。",
  "src_lang": "zh",
  "tgt_lang": "en"
}

Response:
{
    "data": {
        "text": "将图片中的车变为淡粉色。",
        "translation": "The car in the picture turned pink."
    },
    "object": "translation.result",
    "model": "m2m100_1.2B"
}
"""


def init():
    rf = RegisterFunction()
    rf.register('m2m', m2m)
    rf.register('docs', docs)


async def docs(request, params_kw, *params, **kw):
    return helptext


async def m2m(request, params_kw, *params, **kw):
    debug(f'{params_kw=}')
    se = ServerEnv()
    engine = se.engine

    text = params_kw.text
    src_lang = params_kw.src_lang or "zh"
    tgt_lang = params_kw.tgt_lang or "en"
    model = params_kw.model or "m2m100_1.2B"

    if not text or not isinstance(text, str):
        raise Exception("`text` must be a non-empty string")

    f = awaitify(engine.m2m)
    translation = await f(text, src_lang, tgt_lang)

    ret = {
        "data": {
            "text": text,
            "translation": translation
        },
        "object": "translation.result",
        "model": engine.model_name
    }
    return ret


def main():
    parser = argparse.ArgumentParser(prog="M2M100 Translate API")
    parser.add_argument('-w', '--workdir', default=os.getcwd())
    parser.add_argument('-p', '--port', type=int, default=8883)
    parser.add_argument('model_path', help="Path to m2m100 model, e.g. /share/models/facebook/m2m100_1.2B")
    args = parser.parse_args()

    Klass = get_llm_class(args.model_path)
    if Klass is None:
        raise Exception(f"Model {args.model_path} not registered. Should contain 'm2m100'")

    se = ServerEnv()
    se.engine = Klass(args.model_path)

    debug(f"Starting M2M100 service on port {args.port}")
    webserver(init, args.workdir, args.port)


if __name__ == '__main__':
    main()