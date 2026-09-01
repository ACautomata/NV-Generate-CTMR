#!/usr/bin/env python3
"""一次性运维(集群侧,#254 T7 前置):MR-RATE replay sidecar 重建。

replay 的旧 embedding 树(含 <emb>.json sidecar)随服务器重组消失;T7 loader
无条件读 sidecar 的 spacing/modality(reencode.ensure_sidecars 只支持从旧根
拷贝)。本脚本从 raw_mrrate 的 nii header 逐条重建 sidecar 落进旧 embedding 根
(默认 $PHASE_ROOT/embeddings,即 T4 收尾的 --sidecar-source),使其与 BraTS 段
一样经统一拷贝链路进 embeddings_cliptrue。spacing=raw header zooms 前三维,
modality=replay list 原值(mri_flair→token 11 等,与基线训练映射一致)。
幂等:目标已存在且含双键即跳过。

用法(直接跑;前置:双 source DTK+代理):
  python3 deploy/data/rebuild_replay_sidecars.py \
      [--replay-list <p1_mrrate_replay.json>] [--raw-root <raw_mrrate>] \
      [--sidecar-root <旧 embedding 根>]
默认值按 2026-09-01 实例持久盘布局。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib

DEFAULT_REPLAY_LIST = "/root/private_data/ctmr/runs/p1/lists/p1_mrrate_replay.json"
DEFAULT_RAW_ROOT = "/root/private_data/ctmr/data/phase/raw_mrrate"
DEFAULT_SIDECAR_ROOT = "/root/private_data/ctmr/data/phase/embeddings"


class ReplaySidecarRebuilder:
    def __init__(self, replay_list, raw_root, sidecar_root):
        self._entries = json.loads(Path(replay_list).read_text())["training"]
        self._raw_root = Path(raw_root)
        self._sidecar_root = Path(sidecar_root)

    @staticmethod
    def _sidecar_path(root, image):
        return root / (image.replace(".nii.gz", "_emb.nii.gz") + ".json")

    def rebuild(self):
        ok = present = fail = 0
        failures = []
        started = time.time()
        for pos, entry in enumerate(self._entries, 1):
            image = entry["image"]
            dest = self._sidecar_path(self._sidecar_root, image)
            try:
                if dest.is_file():
                    payload = json.loads(dest.read_text())
                    if "spacing" in payload and "modality" in payload:
                        present += 1
                        continue
                header = nib.load(str(self._raw_root / image)).header
                spacing = [float(v) for v in header.get_zooms()[:3]]
                payload = {"spacing": spacing, "modality": entry["modality"]}
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(payload))
                ok += 1
            except Exception as error:  # noqa: BLE001
                fail += 1
                failures.append((image, type(error).__name__))
            if pos % 1000 == 0:
                print(f"[sidecar] {pos}/{len(self._entries)} ok={ok} present={present} fail={fail} ({time.time() - started:.0f}s)", flush=True)
        print(f"[sidecar] done: total={len(self._entries)} ok={ok} present={present} fail={fail} failures(前5)={failures[:5]}", flush=True)
        return 1 if fail else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay-list", default=DEFAULT_REPLAY_LIST)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--sidecar-root", default=DEFAULT_SIDECAR_ROOT)
    args = parser.parse_args(argv)
    return ReplaySidecarRebuilder(args.replay_list, args.raw_root, args.sidecar_root).rebuild()


if __name__ == "__main__":
    sys.exit(main())
