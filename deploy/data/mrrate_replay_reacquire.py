#!/usr/bin/env python3
# 序列② T7 前置(issue #254,父 #247):MR-RATE replay 语料 raw 重获取——sugon 数据恢复作业
#
# 背景:服务器重组后 replay 的 MR-RATE raw 树(/root/private_data/p1_replay/raw)随易失盘
#   消失,而 T7 决定改动①(编码 clip=True)一致施于全部训练输入(BraTS 主 list + MR-RATE
#   replay 各 7404)——replay 需从 raw 重新以 clip=True 编码,故先重获取 raw。本脚本按
#   replay list 逐 case 从 HuggingFace `Forithmus/MR-RATE`(镜像端点)拉取每 subject 的
#   整包 zip,解出 list 点名的那一个 scan(`<case>/img/<image  basename>`)落进 raw 树,
#   随即删 zip(流式,控盘占用)。variant=diagnostic:不动任何冻结仪器/包络/判定线。
#
# 设计约束(实测):
#   - hf-mirror 的 python `hf_hub_download`(requests 流)经代理易 `IncompleteRead` 断流;
#     裸 curl 直连 resolve 端点稳定(实测 ~5-8 MB/s)。故下载用 curl 子进程
#     (--retry --retry-all-errors -C - 续传),不用 python 库。
#   - zip 内成员命名 `<CASE>/img/<CASE>_<modality>-raw-<orient>[-NN].nii.gz`;
#     replay list 的 image basename 与之逐字对应(含 -NN 变体),按 basename 精确匹配。
#   - 可断点续传:目标 nii.gz 已存在且非空即跳过;失败逐条记 ledger,进程可安全重跑。
#
# 用法(直接跑;前置:双 source DTK+代理,HF_TOKEN/HF_ENDPOINT 经环境注入):
#   python3 deploy/data/mrrate_replay_reacquire.py --replay-list <p1_mrrate_replay.json> \
#       --raw-root <raw 根> --workers 8 [--limit N] [--report <ledger.json>]
#
# 环境变量:HF_TOKEN(HF 访问 token,勿硬编码)、HF_ENDPOINT(镜像端点)。token 经
#   --header @<tmpfile> 注入 curl(0600 临时头文件,进程退出即删),不上 argv(防 ps 旁观)。
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "Forithmus/MR-RATE"


class ReplayReacquirer:
    """按 replay list 逐 case 重获取 MR-RATE raw:curl 拉 zip → 解出点名 scan → 删 zip。

    一个实例持有整趟作业的配置与运行台帐(work ledger);线程池并行,每 worker
    独立 curl + zipfile,落盘即跳过已存在目标,失败不致命、逐条登记供重跑。
    """

    def _auth_header_file(self):
        """token 写 0600 临时头文件(curl -H @file),进程退出即删,防 ps 旁观 argv。"""
        handle = tempfile.NamedTemporaryFile("w", prefix="mrrate_auth_", suffix=".hdr", delete=False)
        handle.write(f"Authorization: Bearer {self._token}\n")
        handle.close()
        os.chmod(handle.name, 0o600)
        return handle.name

    def __init__(self, replay_list, raw_root, workers, endpoint, token):
        self._replay_list = Path(replay_list)
        self._raw_root = Path(raw_root)
        self._workers = workers
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._tmp_root = Path(tempfile.mkdtemp(prefix="mrrate_reacquire_"))
        self._auth_header = self._auth_header_file()

    def cleanup(self):
        """作业结束清理临时区:zip 下载目录与 token 头文件(幂等)。"""
        shutil.rmtree(self._tmp_root, ignore_errors=True)
        Path(self._auth_header).unlink(missing_ok=True)

    def work_items(self):
        """解析 replay list → (image, batch, case, fname, dest) 五元组,按 list 序。"""
        entries = json.loads(self._replay_list.read_text())["training"]
        items = []
        for entry in entries:
            image = entry["image"]
            parts = image.split("/")
            # image 形如 MR-RATE/<batchNN>/<CASE>/<fname>;末段为目标文件名
            batch, case, fname = parts[1], parts[2], parts[-1]
            dest = self._raw_root / "MR-RATE" / batch / case / fname
            items.append((image, batch, case, fname, dest))
        return items

    def _download_zip(self, batch, case, zip_path):
        url = f"{self._endpoint}/datasets/{REPO}/resolve/main/mri/{batch}/{case}.zip"
        cmd = [
            "curl",
            "-sfSL",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "30",
            "--max-time",
            "900",
            "-C",
            "-",  # 断点续传(配合 .part 临时文件)
            "-H",
            f"@{self._auth_header}",  # @file 注入 token,不上 argv
            "-o",
            str(zip_path),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1000)
        return result.returncode == 0 and zip_path.is_file() and zip_path.stat().st_size > 0

    def _extract_member(self, zip_path, case, fname, dest):
        """从 zip 解出 list 点名的 scan(精确 basename 匹配)到 dest;返回成败。"""
        member = f"{case}/img/{fname}"
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            if member not in names:
                # 兜底:任意路径下 basename 相同者(命名差一层的个例)
                candidates = [n for n in names if n.split("/")[-1] == fname and "/img/" in n]
                if not candidates:
                    return False
                member = candidates[0]
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src:
                payload = src.read()
            tmp_dest = dest.with_suffix(dest.suffix + ".part")
            tmp_dest.write_bytes(payload)
            tmp_dest.replace(dest)  # 原子就位,防半文件被误判为已完成
        return True

    def fetch(self, item):
        """单 case 全流程;返回 (image, status)。status: present/ok/具体失败因。"""
        image, batch, case, fname, dest = item
        if dest.is_file() and dest.stat().st_size > 0:
            return image, "present"
        zip_path = self._tmp_root / batch / f"{case}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not self._download_zip(batch, case, zip_path):
                return image, "download_fail"
            try:
                if not self._extract_member(zip_path, case, fname, dest):
                    return image, "member_missing"
            except zipfile.BadZipFile:
                return image, "bad_zip"
            return image, "ok"
        except (subprocess.TimeoutExpired, OSError) as error:
            return image, f"error:{type(error).__name__}"
        finally:
            zip_path.unlink(missing_ok=True)  # 流式:解完即删 zip,控盘占用

    def run(self, limit=None, progress_every=50):
        items = self.work_items()
        if limit:
            items = items[:limit]
        total = len(items)
        ledger = {"total": total, "ok": 0, "present": 0, "failures": {}}
        started = time.time()
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {pool.submit(self.fetch, item): item for item in items}
            for index, future in enumerate(as_completed(futures), 1):
                image, status = future.result()
                if status in ("ok", "present"):
                    ledger[status] += 1
                else:
                    ledger["failures"][image] = status
                if index % progress_every == 0 or index == total:
                    rate = index / max(time.time() - started, 1e-9)
                    print(
                        f"[reacquire] {index}/{total} ok={ledger['ok']} present={ledger['present']} "
                        f"fail={len(ledger['failures'])} ({rate:.1f} case/s)",
                        flush=True,
                    )
        return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-list", required=True)
    parser.add_argument("--raw-root", required=True, help="MR-RATE raw 落盘根(其下生 MR-RATE/<batch>/<case>/<file>)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条(冒烟用;0=全量)")
    parser.add_argument("--report", default=None, help="失败/统计 ledger 落盘路径(json)")
    args = parser.parse_args(argv)

    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[FATAL] HF_TOKEN 未设(HF 访问 token,经环境注入,勿硬编码)", file=sys.stderr)
        return 2

    reacquirer = ReplayReacquirer(args.replay_list, args.raw_root, args.workers, endpoint, token)
    try:
        ledger = reacquirer.run(limit=args.limit or None)
    finally:
        reacquirer.cleanup()
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    n_fail = len(ledger["failures"])
    print(f"[reacquire] done: total={ledger['total']} ok={ledger['ok']} present={ledger['present']} fail={n_fail}")
    if n_fail:
        shown = list(ledger["failures"].items())[:10]
        print(f"[reacquire] 失败样例(前 10): {shown}", file=sys.stderr)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
