"""常駐容量のメモリ予算導出（ElfMoon128 の中核）のテスト。

実モデル不要。ExpertStore / mx.get_active_memory はスタブで置き換える。
"""

import os

import stream_model as sm
from resident_cache import (
    DEFAULT_HEADROOM,
    ResidentCache,
    budget_bytes_from_env,
    detect_ram_bytes,
    detect_working_set_bytes,
    plan_cache_experts,
)

GB = 1024**3
MB = 1024**2


class _FakeStore:
    def __init__(self, per_expert_bytes):
        self._pe = per_expert_bytes

    def per_expert_bytes(self):
        return self._pe


class _FakeMx:
    """mx.get_active_memory() / mx.eval() を差し替えるスタブ。"""

    def __init__(self, active):
        self._active = active

    def get_active_memory(self):
        return self._active

    def eval(self, *a, **kw):
        pass


def _autotune(cache, store, active_bytes, model_path=None, perf=False, auto=True):
    """mx を差し替えて autotune_capacity を実行する。"""
    real_mx = sm.mx
    sm.mx = _FakeMx(active_bytes)
    try:
        sm.autotune_capacity(cache, store, model_path, auto, perf)
    finally:
        sm.mx = real_mx


def test_detect_ram():
    """物理 RAM が正の値で取れる（macOS の sysctl 経路）。"""
    assert detect_ram_bytes() > 0


def test_budget_env_override():
    """ELFMOON_MEM_BUDGET_GB が物理 RAM より優先される。"""
    old = os.environ.get("ELFMOON_MEM_BUDGET_GB")
    try:
        os.environ["ELFMOON_MEM_BUDGET_GB"] = "128"
        assert budget_bytes_from_env() == 128 * GB
        # 数値でない値は無視して自動検出（物理 RAM とワーキングセット上限の小さい方）
        os.environ["ELFMOON_MEM_BUDGET_GB"] = "abc"
        assert budget_bytes_from_env() == min(
            detect_ram_bytes(), detect_working_set_bytes()
        )
    finally:
        if old is None:
            os.environ.pop("ELFMOON_MEM_BUDGET_GB", None)
        else:
            os.environ["ELFMOON_MEM_BUDGET_GB"] = old


def test_non_expert_is_subtracted():
    """非 expert 重みの分だけ容量が減る（OOM 回避の核心）。"""
    pe = 2 * MB
    lean = plan_cache_experts(128 * GB, 0, pe, headroom=DEFAULT_HEADROOM)
    heavy = plan_cache_experts(128 * GB, 40 * GB, pe, headroom=DEFAULT_HEADROOM)
    assert heavy < lean
    # 予算×headroom から非 expert を引いた分が expert に回る
    assert heavy == int(128 * GB * DEFAULT_HEADROOM - 40 * GB) // pe


def test_max_experts_caps():
    """全部載るモデルで expert 総数を超えるスロットを確保しない。"""
    n = plan_cache_experts(128 * GB, 4 * GB, 2 * MB, max_experts=5120)
    assert n == 5120


def test_budget_exhausted_returns_one():
    """非 expert だけで予算を食い潰しても 0 スロットにはしない。"""
    assert plan_cache_experts(8 * GB, 100 * GB, 2 * MB) == 1


def test_autotune_sets_capacity():
    """autotune_capacity が暫定容量を予算導出値に置き換える。"""
    os.environ["ELFMOON_MEM_BUDGET_GB"] = "128"
    try:
        cache = ResidentCache(sm._PROVISIONAL_CAPACITY)
        _autotune(cache, _FakeStore(2 * MB), active_bytes=20 * GB)
        assert cache.capacity != sm._PROVISIONAL_CAPACITY
        assert cache.capacity == int(128 * GB * DEFAULT_HEADROOM - 20 * GB) // (2 * MB)
    finally:
        os.environ.pop("ELFMOON_MEM_BUDGET_GB", None)


def test_autotune_noop_when_explicit():
    """容量を明示指定したときは自動導出が働かない（後方互換）。"""
    cache = ResidentCache(6144)
    _autotune(cache, _FakeStore(2 * MB), active_bytes=20 * GB, auto=False)
    assert cache.capacity == 6144


def test_autotune_gives_up_without_store():
    """expert サイズが取れない（store 未生成等）なら暫定容量のまま続行する。"""
    cache = ResidentCache(sm._PROVISIONAL_CAPACITY)
    _autotune(cache, _FakeStore(0), active_bytes=20 * GB)
    assert cache.capacity == sm._PROVISIONAL_CAPACITY


def test_autotune_rejects_implausible_non_expert():
    """非 expert の実測値が過小（＝遅延評価で未実体化）なら容量を確定しない。

    ここで 0 付近を信じると予算を丸ごと expert に配って OOM するため、
    暫定容量のまま続行する方が安全。
    """
    os.environ["ELFMOON_MEM_BUDGET_GB"] = "128"
    try:
        cache = ResidentCache(sm._PROVISIONAL_CAPACITY)
        _autotune(cache, _FakeStore(2 * MB), active_bytes=1 * MB)
        assert cache.capacity == sm._PROVISIONAL_CAPACITY
    finally:
        os.environ.pop("ELFMOON_MEM_BUDGET_GB", None)


def test_budget_capped_by_working_set():
    """予算が GPU ワーキングセット上限を超えない（超えると確保に失敗する）。"""
    ws = detect_working_set_bytes()
    assert ws > 0, "Apple Silicon ではワーキングセット上限が取得できるはず"
    assert budget_bytes_from_env() <= ws


def test_perf_mode_is_larger():
    """性能モードは省メモリモードより多くのスロットを確保する。"""
    os.environ["ELFMOON_MEM_BUDGET_GB"] = "128"
    try:
        lean = ResidentCache(0)
        fast = ResidentCache(0)
        _autotune(lean, _FakeStore(2 * MB), active_bytes=20 * GB, perf=False)
        _autotune(fast, _FakeStore(2 * MB), active_bytes=20 * GB, perf=True)
        assert fast.capacity > lean.capacity
    finally:
        os.environ.pop("ELFMOON_MEM_BUDGET_GB", None)


def test_count_experts_missing_config():
    """config.json が読めなければ上限なし（None）として扱う。"""
    assert sm._count_experts(None) is None
    assert sm._count_experts("/nonexistent/path") is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"=== {name} ===")
            fn()
    print("All tests passed.")
