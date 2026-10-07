"""只监听本机的采掘模拟界面。"""

import argparse
import base64
import json
import re
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from data_io import export_dataset, images, png, single_archive
from mine_model import default_request, parse_request, request_dict, simulate

ROOT = Path(__file__).resolve().parent
JOBS = {}
LOCK = threading.Lock()
COMPUTE = threading.BoundedSemaphore(2)


def preview(request):
    with COMPUTE:
        arrays, metadata = simulate(request)
        encoded = {
            key: "data:image/png;base64," + base64.b64encode(png(value)).decode()
            for key, value in images(arrays).items()
        }
    row = metadata["profile_row"]
    return {
        "images": encoded,
        "metadata": metadata,
        "profile": {
            key: (arrays[key][row] * 1000).tolist()
            for key in ("subsidence_before_m", "subsidence_after_m", "delta_down_m")
        },
    }


def status(job_id):
    with LOCK:
        if job_id not in JOBS:
            raise ValueError("找不到这次生成任务")
        return {key: value for key, value in JOBS[job_id].items() if key != "stop"}


def batch(body):
    settings, faces = parse_request(body.get("request", {}))
    request = request_dict(settings, faces)
    count = body.get("count", 50)
    if type(count) is not int or not 50 <= count <= 2000:
        raise ValueError("样本数需为 50–2000 的整数")
    name = body.get("name", "mine_faces")
    if not isinstance(name, str) or not re.fullmatch(r"[\w-]{1,40}", name):
        raise ValueError("名称请使用 1–40 个中文、字母、数字、下划线或短横线")
    key = uuid.uuid4().hex[:10]
    root = ROOT / "exports" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}_{key[:4]}"
    with LOCK:
        if any(item["state"] in ("running", "stopping") for item in JOBS.values()):
            raise ValueError("已有一组数据正在生成")
        JOBS[key] = {
            "id": key,
            "state": "running",
            "done": 0,
            "total": count,
            "path": str(root),
            "error": None,
            "stop": threading.Event(),
        }

    def progress(done, total):
        with LOCK:
            JOBS[key]["done"] = done

    def run():
        try:
            result = export_dataset(root, count, request, progress, JOBS[key]["stop"])
            with LOCK:
                JOBS[key].update(
                    state="cancelled" if result["cancelled"] else "complete", result=result
                )
        except Exception as error:
            with LOCK:
                JOBS[key].update(state="failed", error=str(error))

    threading.Thread(target=run, daemon=True).start()
    return status(key)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, body, mime="application/json", code=200, filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed(self):
        allowed = (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")
        return self.headers.get("Host") in allowed and self.headers.get("Origin") in (
            None,
            *("http://" + host for host in allowed),
        )

    def do_GET(self):
        if not self.allowed():
            return self.respond({"error": "请通过本机地址访问"}, code=403)
        route = self.path.split("?", 1)[0]
        if route == "/api/defaults":
            return self.respond(
                {
                    "request": default_request(),
                    "exports": str(ROOT / "exports"),
                    "reference_available": (ROOT / "web/reference.jpg").is_file(),
                }
            )
        if route == "/api/jobs":
            with LOCK:
                keys = list(JOBS)
            return self.respond({"jobs": [status(key) for key in keys]})
        if route.startswith("/api/jobs/"):
            try:
                return self.respond(status(route.rsplit("/", 1)[-1]))
            except ValueError as error:
                return self.respond({"error": str(error)}, code=404)
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/reference.jpg": ("reference.jpg", "image/jpeg"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if route in files:
            filename, mime = files[route]
            path = ROOT / "web" / filename
            if not path.is_file():
                return self.respond({"error": "Optional asset unavailable"}, code=404)
            return self.respond(path.read_bytes(), mime)
        return self.respond({"error": "页面不存在"}, code=404)

    def do_POST(self):
        if not self.allowed():
            return self.respond({"error": "请求来源不符"}, code=403)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.respond({"error": "需要 JSON 参数"}, code=415)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("请求过大或为空")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("参数格式错误")
            if self.path == "/api/preview":
                return self.respond(preview(body))
            if self.path == "/api/export":
                with COMPUTE:
                    archive = single_archive(body)
                return self.respond(archive, "application/zip", filename="mine_face_scene.zip")
            if self.path == "/api/batch":
                return self.respond(batch(body), code=202)
            if self.path == "/api/stop":
                with LOCK:
                    item = JOBS.get(body.get("id"))
                    if item is None:
                        raise ValueError("任务不存在")
                    if item["state"] == "running":
                        item["stop"].set()
                        item["state"] = "stopping"
                return self.respond(status(body["id"]))
            return self.respond({"error": "接口不存在"}, code=404)
        except (ValueError, TypeError, KeyError) as error:
            return self.respond({"error": str(error)}, code=400)
        except Exception as error:
            return self.respond({"error": f"生成失败：{error}"}, code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(
            1, f"无法启动本机服务：{error}。已有实例时请打开原地址，或使用 --port 指定其他端口。\n"
        )
    print(f"采掘干涉模拟实验室：http://127.0.0.1:{args.port}/", flush=True)
    if args.open:
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
