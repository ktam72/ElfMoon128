"""モジュール②: バイト予算つき LRU 常駐キャッシュ。

ホットexpertをGPU/ユニファイドメモリに常駐させ、予算超過時はLRUで退避。
DS4 の ds4_ssd_auto_cache_plan（予算×4/5、非routed差引、expert数算出）に対応。
命中率がElfMoonの速度を決める中核指標なので hit/miss を記録する。
"""

import os
import subprocess
from collections import OrderedDict

# 予算に対する常駐率。KV キャッシュ・活性化・他アプリの取り分を残す。
# 200B+ MoE では KV が数 GB 規模になるため DS4 の 0.8 より保守的に取る。
DEFAULT_HEADROOM = 0.75


def detect_ram_bytes():
    """物理 RAM のバイト数。取得できなければ 0。"""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return int(out.stdout.strip())
    except Exception:
        pass
    return 0


def detect_working_set_bytes():
    """GPU の推奨ワーキングセット上限。取得できなければ 0。

    Apple Silicon では MLX バッファがこの上限に対して計上される。128GB 機でも
    上限は 115GB 程度と物理 RAM より小さいため、予算はこちらで頭打ちにする。
    """
    try:
        import mlx.core as mx

        info = mx.device_info()
        return int(info.get("max_recommended_working_set_size", 0))
    except Exception:
        return 0


def budget_bytes_from_env():
    """常駐予算のバイト数。

    ELFMOON_MEM_BUDGET_GB があればその値。無ければ物理 RAM と GPU ワーキング
    セット上限の小さい方（上限超過は算術が正しくても確保に失敗するため）。
    """
    env = os.environ.get("ELFMOON_MEM_BUDGET_GB")
    if env:
        try:
            return int(float(env) * 1024**3)
        except ValueError:
            print(f"  ELFMOON_MEM_BUDGET_GB={env!r} は数値でない: 無視")
    ram = detect_ram_bytes()
    ws = detect_working_set_bytes()
    if ram and ws:
        return min(ram, ws)
    return ram or ws


def plan_cache_experts(
    budget_bytes, non_expert_bytes, per_expert_bytes, max_experts=None, headroom=0.8
):
    """常駐予算からホットexpert数を算出（DS4方式）。"""
    target = int(budget_bytes * headroom)
    cache_bytes = max(0, target - non_expert_bytes)
    n = cache_bytes // per_expert_bytes if per_expert_bytes else 0
    if max_experts is not None:
        n = min(n, max_experts)
    return max(1, int(n))


class ResidentCache:
    """(layer, expert) → 重み dict の LRU キャッシュ。

    capacity は「常駐expert数」。get(key, loader) で命中/ミスを扱う。
    ミス時に loader() で読み込み、満杯なら最古を退避（参照を落として解放）。
    """

    def __init__(self, capacity):
        self.capacity = max(1, int(capacity))
        self._d = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __contains__(self, key):
        return key in self._d

    def prime(self, key, weights):
        """ホットリスト起動時プライム用: 命中/ミス統計を汚さず投入。"""
        self._d[key] = weights
        self._d.move_to_end(key)
        self._evict()

    def get(self, key, loader):
        if key in self._d:
            self.hits += 1
            self._d.move_to_end(key)
            return self._d[key]
        self.misses += 1
        w = loader()
        self._d[key] = w
        self._d.move_to_end(key)
        self._evict()
        return w

    def _evict(self):
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)  # 最古(LRU)を退避

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t if t else 0.0

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "resident": len(self._d),
            "capacity": self.capacity,
        }
