import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["extract_2d_core", "interference_readout", "orthonormalize_pair", "plane_power",
         "subtractive_mix", "sweep_phase", "total_power", "unitary_interferometer"]


def _run(code, cwd, extra_env=None):
    env = dict(os.environ, **(extra_env or {}))
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_phase_e_shim_reexports_real_module():
    r = _run("import residual_interferometer as m; print(all(hasattr(m,n) for n in %r))" % NAMES,
             ROOT / "packages" / "residual_isa_phase_e")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"


def test_toolkit_interferometer_imports_from_any_cwd(tmp_path):
    pkg = ROOT / "packages" / "residual_causal_gpt2" / "residual_toolkit" / "interferometer.py"
    code = ("import importlib.util as u; s=u.spec_from_file_location('ifm', %r); m=u.module_from_spec(s); "
            "s.loader.exec_module(m); print(m.extract_2d_core.__module__)" % str(pkg))
    r = _run(code, tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "residual_interferometer"


def test_random_control_seed_is_stable_across_processes():
    code = ("import sys; sys.path.insert(0, %r); from residual_poly_14d import causal_port; import numpy as np; "
            "X=np.random.default_rng(0).normal(size=(64,6)); d=np.eye(6)[0]; print(causal_port(X,X*0,d,'p')['rand'])"
            % str(ROOT / "experiments" / "core"))
    outs = {_run(code, ROOT, {"PYTHONHASHSEED": s}).stdout for s in ("1", "2", "3")}
    assert len(outs) == 1 and next(iter(outs)).strip()


def test_missing_phase_modules_fail_clearly():
    for n in ("a", "c"):
        r = subprocess.run([sys.executable, str(ROOT / "packages/residual_causal_gpt2/scripts" / f"phase_{n}.py")],
                           capture_output=True, text=True)
        assert r.returncode != 0
        assert "not in this repository" in r.stderr
