"""Sandboxed execution for verifiers and agent runs.

Deny by default: no network, no home directory, no host credentials or agent
sockets, an empty environment plus an explicit allowlist, read-only inputs, one
writable scratch directory, a wall-clock timeout and resource limits.

IMPORTANT: bubblewrap and Docker reduce the blast radius of *accidents* in trusted
fixture code. Neither is a security boundary against deliberately hostile code
(kernel attack surface, resource exhaustion, side channels). See docs/security.md.
"""

from __future__ import annotations

import contextlib
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

MAX_OUTPUT = 64 * 1024


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class SandboxPolicy:
    """What a sandboxed process may see and do."""

    mounts_rw: dict[str, Path] = field(default_factory=dict)  # name -> host dir
    mounts_ro: dict[str, Path] = field(default_factory=dict)
    workdir: str = "work"  # mount name used as cwd
    network: bool = False
    timeout_s: float = 120.0
    memory_mb: int = 2048
    cpu_s: int = 300
    max_file_mb: int = 64
    max_open_files: int = 512
    env: dict[str, str] = field(default_factory=dict)  # explicit allowlist only
    trusted_fixture: bool = True  # False = generated/untrusted code: stricter callers

    def __post_init__(self) -> None:
        if self.workdir not in self.mounts_rw:
            raise ValueError(f"workdir mount {self.workdir!r} must be a read-write mount")


@dataclass
class ExecResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    backend: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(Protocol):
    name: str

    def available(self) -> bool: ...

    def mount_point(self, name: str, policy: SandboxPolicy) -> str: ...

    def run(self, argv: list[str], policy: SandboxPolicy) -> ExecResult: ...


def _limits(policy: SandboxPolicy) -> None:  # runs in the child before exec
    mb = 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (policy.memory_mb * mb, policy.memory_mb * mb))
    resource.setrlimit(resource.RLIMIT_CPU, (policy.cpu_s, policy.cpu_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (policy.max_file_mb * mb, policy.max_file_mb * mb))
    resource.setrlimit(resource.RLIMIT_NOFILE, (policy.max_open_files, policy.max_open_files))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _truncate(b: bytes) -> str:
    s = b.decode("utf-8", errors="replace")
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + "\n[... truncated ...]"


def run_process(
    argv: list[str],
    policy: SandboxPolicy,
    backend: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    apply_limits: bool = True,
) -> ExecResult:
    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env if env is not None else {},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=(lambda: _limits(policy)) if apply_limits else None,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=policy.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
    return ExecResult(
        argv=argv,
        exit_code=proc.returncode if not timed_out else -9,
        stdout=_truncate(out),
        stderr=_truncate(err),
        duration_s=round(time.monotonic() - start, 3),
        timed_out=timed_out,
        backend=backend,
    )


def toolchain_paths() -> list[Path]:
    """Read-only interpreter + virtualenv prefixes needed to run Python verifiers.

    These are bound read-only at their own paths. The home directory itself is not
    mounted.
    """
    out: list[Path] = []
    for p in (Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()):  # venv first
        if p.exists() and not str(p).startswith("/usr") and p not in out:
            out.append(p)
    return out


def symlinked_ancestors(paths: list[Path]) -> list[tuple[Path, Path]]:
    """(link, resolved) for symlinked directories on the way to each interpreter.

    uv virtualenvs point ``bin/python`` at e.g. ``.../cpython-3.12-linux/bin/python3.12``
    where ``cpython-3.12-linux`` is itself a symlink; the sandbox needs that alias too.
    """
    out: list[tuple[Path, Path]] = []
    for prefix in paths:
        exe = prefix / "bin" / "python"
        target = Path(os.readlink(exe)) if exe.is_symlink() else None
        if target is None or not target.is_absolute():
            continue
        for anc in [target, *target.parents]:
            if anc.is_symlink() and (anc, anc.resolve()) not in out:
                out.append((anc, anc.resolve()))
    return out


def default_env(
    extra: dict[str, str] | None = None, path_dirs: list[str] | None = None
) -> dict[str, str]:
    path = ":".join([*(path_dirs or []), "/usr/local/bin", "/usr/bin", "/bin"])
    env = {
        "PATH": path,
        "HOME": "/tmp/home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": "/tmp",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    env.update(extra or {})
    return env


class BwrapSandbox:
    name = "bwrap"
    _ETC = (
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/alternatives",
        "/etc/localtime",
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
    )
    _ETC_NET = ("/etc/resolv.conf", "/etc/hosts", "/etc/ssl", "/etc/ca-certificates", "/etc/pki")

    def __init__(self, extra_ro: list[Path] | None = None):
        self.extra_ro = [p.resolve() for p in (extra_ro or [])]

    def available(self) -> bool:
        if not shutil.which("bwrap"):
            return False
        probe = subprocess.run(
            [
                "bwrap",
                "--unshare-all",
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/bin",
                "/bin",
                "--symlink",
                "usr/lib",
                "/lib",
                "--symlink",
                "usr/lib64",
                "/lib64",
                "/usr/bin/true",
            ],
            capture_output=True,
            check=False,
        )
        return probe.returncode == 0

    def mount_point(self, name: str, policy: SandboxPolicy) -> str:
        return f"/{name}"

    def run(self, argv: list[str], policy: SandboxPolicy) -> ExecResult:
        if not shutil.which("bwrap"):
            raise SandboxUnavailable("bwrap is not installed")
        args = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for link, target in (
            ("/bin", "usr/bin"),
            ("/sbin", "usr/sbin"),
            ("/lib", "usr/lib"),
            ("/lib64", "usr/lib64"),
            ("/lib32", "usr/lib32"),
        ):
            p = Path(link)
            if p.is_symlink():
                args += ["--symlink", target, link]
            elif p.exists():
                args += ["--ro-bind", link, link]
        for etc in self._ETC:
            args += ["--ro-bind-try", etc, etc]
        if policy.network:
            args.append("--share-net")
            for etc in self._ETC_NET:
                args += ["--ro-bind-try", etc, etc]
        args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/tmp/home"]
        tool_dirs = toolchain_paths()
        for p in [*tool_dirs, *self.extra_ro]:
            args += ["--ro-bind", str(p), str(p)]
        for link, resolved in symlinked_ancestors(tool_dirs):
            args += ["--ro-bind", str(resolved), str(link)]
        for name, host in policy.mounts_ro.items():
            args += ["--ro-bind", str(host.resolve()), f"/{name}"]
        for name, host in policy.mounts_rw.items():
            args += ["--bind", str(host.resolve()), f"/{name}"]
        env = default_env(policy.env, [f"{p}/bin" for p in tool_dirs])
        args += ["--clearenv"]
        for k, v in sorted(env.items()):
            args += ["--setenv", k, v]
        args += ["--chdir", f"/{policy.workdir}", "--", *argv]
        return run_process(args, policy, self.name)


class DockerSandbox:
    name = "docker"

    def __init__(self, image: str = "skill-foundry-verifier:1"):
        self.image = image

    def available(self) -> bool:
        if not shutil.which("docker"):
            return False
        r = subprocess.run(
            ["docker", "image", "inspect", self.image], capture_output=True, check=False
        )
        return r.returncode == 0

    def mount_point(self, name: str, policy: SandboxPolicy) -> str:
        return f"/{name}"

    def run(self, argv: list[str], policy: SandboxPolicy) -> ExecResult:
        if not shutil.which("docker"):
            raise SandboxUnavailable("docker is not installed")
        args = [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=256m",
            "--memory",
            f"{policy.memory_mb}m",
            "--cpus",
            "1",
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--network",
            "bridge" if policy.network else "none",
        ]
        for name, host in policy.mounts_ro.items():
            args += ["-v", f"{host.resolve()}:/{name}:ro"]
        for name, host in policy.mounts_rw.items():
            args += ["-v", f"{host.resolve()}:/{name}:rw"]
        env = default_env(policy.env)
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        for k, v in sorted(env.items()):
            args += ["-e", f"{k}={v}"]
        args += ["-w", f"/{policy.workdir}", self.image, *argv]
        # Resource limits are enforced by Docker; rlimits would also hit the client.
        return run_process(args, policy, self.name, env=dict(os.environ), apply_limits=False)


class UnsafeHostSandbox:
    """Runs directly on the host with a scrubbed environment. Explicit opt-in only."""

    name = "unsafe-host"

    def available(self) -> bool:
        return True

    def mount_point(self, name: str, policy: SandboxPolicy) -> str:
        host = policy.mounts_rw.get(name) or policy.mounts_ro.get(name)
        if host is None:
            raise KeyError(name)
        return str(host.resolve())

    def run(self, argv: list[str], policy: SandboxPolicy) -> ExecResult:
        cwd = policy.mounts_rw[policy.workdir]
        env = default_env(policy.env, [str(Path(sys.executable).parent)])
        env["HOME"] = str(cwd)
        env["TMPDIR"] = str(cwd)
        return run_process(argv, policy, self.name, cwd=cwd, env=env)


def make_sandbox(name: str, *, allow_unsafe_host: bool = False) -> Sandbox:
    if name == "auto":
        for sb in (BwrapSandbox(), DockerSandbox()):
            if sb.available():
                return sb
        raise SandboxUnavailable(
            "no sandbox backend available (install bubblewrap or Docker, or pass "
            "--sandbox unsafe-host --unsafe-host for trusted fixtures only)"
        )
    if name == "bwrap":
        return BwrapSandbox()
    if name == "docker":
        return DockerSandbox()
    if name == "unsafe-host":
        if not allow_unsafe_host:
            raise SandboxUnavailable(
                "unsafe-host execution requires the explicit --unsafe-host flag"
            )
        return UnsafeHostSandbox()
    raise ValueError(f"unknown sandbox {name!r}")
