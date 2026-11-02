#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML PRO · проверка окружения перед курсом.

Запуск:  python check_env.py        (или  uv run check_env.py)
Скрипт ничего не устанавливает и ничего не меняет. Только проверяет и подсказывает.
Результат дублируется в файл check_env_report.txt.

Требует только стандартную библиотеку Python 3.8+.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field

# Windows-консоль по умолчанию не всегда в UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, fix: str = "") -> None:
        self.checks.append(Check(name, status, detail, fix))


def run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    """Запустить команду, вернуть (код, вывод). Не бросает исключений."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, "не найдено"
    except subprocess.TimeoutExpired:
        return 124, "таймаут"
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


# ---------------------------------------------------------------- проверки
def check_python(r: Report) -> None:
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) >= (3, 11):
        r.add("Python 3.11+", OK, detail)
    else:
        r.add("Python 3.11+", FAIL, detail,
              "Установите Python 3.11 или новее. Проще всего через uv: `uv python install 3.11`")


def check_uv(r: Report) -> None:
    code, out = run(["uv", "--version"])
    if code == 0:
        r.add("uv", OK, out)
    else:
        r.add("uv", FAIL, "не найден в PATH",
              "Windows (PowerShell): `powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"`\n"
              "        macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`\n"
              "        Либо `pip install uv`. После установки перезапустите терминал")


def check_git(r: Report) -> None:
    code, out = run(["git", "--version"])
    if code != 0:
        r.add("git", FAIL, "не найден",
              "Windows: https://git-scm.com/download/win · macOS: `xcode-select --install` · Linux: `sudo apt install git`")
        return
    r.add("git", OK, out)
    _, name = run(["git", "config", "--global", "user.name"])
    _, email = run(["git", "config", "--global", "user.email"])
    if name and email and "не найдено" not in name:
        r.add("git config", OK, f"{name} <{email}>")
    else:
        r.add("git config", WARN, "не заданы имя и почта",
              "`git config --global user.name \"Имя Фамилия\"` и `git config --global user.email you@example.com`")


def check_docker(r: Report) -> None:
    code, out = run(["docker", "--version"])
    if code != 0:
        r.add("docker", FAIL, "не найден",
              "Установите Docker Desktop: https://www.docker.com/products/docker-desktop/ "
              "(Windows: сначала WSL2, см. гайд)")
        return
    r.add("docker", OK, out.splitlines()[0])
    code, out = run(["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}}/{{.Architecture}}"], timeout=40)
    if code == 0 and out and "error" not in out.lower():
        r.add("docker daemon", OK, f"сервер {out}")
    else:
        r.add("docker daemon", FAIL, "Docker установлен, но демон не отвечает",
              "Запустите Docker Desktop и дождитесь зелёного индикатора. На Windows проверьте, что включён WSL2-бэкенд")
        return
    code, out = run(["docker", "run", "--rm", "hello-world"], timeout=120)
    if code == 0:
        r.add("docker run", OK, "hello-world запустился")
    else:
        r.add("docker run", FAIL, out[-300:],
              "Проверьте доступ к Docker Hub (VPN, корпоративный прокси) и что Docker Desktop полностью загрузился")


def check_kind_kubectl(r: Report) -> None:
    code, out = run(["kind", "--version"])
    if code == 0:
        r.add("kind", OK, out)
    else:
        r.add("kind", FAIL, "не найден",
              "Windows: `winget install Kubernetes.kind` или `choco install kind` · macOS: `brew install kind` · "
              "Linux: https://kind.sigs.k8s.io/docs/user/quick-start/#installation")
    code, out = run(["kubectl", "version", "--client", "--output=yaml"])
    if code == 0:
        ver = next((l.strip() for l in out.splitlines() if "gitVersion" in l), out.splitlines()[0] if out else "")
        r.add("kubectl", OK, ver)
    else:
        r.add("kubectl", FAIL, "не найден",
              "Идёт в комплекте с Docker Desktop (включите Kubernetes в настройках или добавьте в PATH), "
              "либо `winget install Kubernetes.kubectl` / `brew install kubectl`")


def total_ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024**3
        if sys.platform == "darwin":
            code, out = run(["sysctl", "-n", "hw.memsize"])
            return int(out) / 1024**3 if code == 0 else None
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1024**2
    except Exception:  # noqa: BLE001
        return None
    return None


def check_hardware(r: Report) -> None:
    ram = total_ram_gb()
    if ram is None:
        r.add("RAM", WARN, "не удалось определить")
    elif ram >= 15:
        r.add("RAM", OK, f"{ram:.0f} ГБ")
    elif ram >= 7.5:
        r.add("RAM", WARN, f"{ram:.0f} ГБ: хватит на недели 1–2, кластер с MLflow и Airflow в неделе 3 будет тесно",
              "Закройте браузер с сотней вкладок на семинарах, либо возьмите GitHub Codespaces (60 бесплатных часов в месяц)")
    else:
        r.add("RAM", FAIL, f"{ram:.0f} ГБ: локально курс не пройти",
              "GitHub Codespaces или VPS 4 vCPU / 8 ГБ. Напишите куратору, подберём вариант")
    free = shutil.disk_usage(os.path.expanduser("~")).free / 1024**3
    if free >= 20:
        r.add("Диск", OK, f"свободно {free:.0f} ГБ")
    elif free >= 10:
        r.add("Диск", WARN, f"свободно {free:.0f} ГБ, образы Docker быстро съедят",
              "Освободите место или перенесите Docker data-root на другой диск")
    else:
        r.add("Диск", FAIL, f"свободно {free:.0f} ГБ", "Нужно минимум 20 ГБ под образы и данные")
    arch = platform.machine()
    note = "Apple Silicon: образы arm64, на семинарах будут оговорки" if arch in ("arm64", "aarch64") else ""
    r.add("Система", OK, f"{platform.system()} {platform.release()} · {arch} " + note)
    if sys.platform == "win32":
        code, out = run(["wsl", "--status"], timeout=15)
        if code == 0:
            r.add("WSL2", OK, "установлен")
        else:
            r.add("WSL2", WARN, "не обнаружен или не отвечает",
                  "Docker Desktop на Windows работает через WSL2: в PowerShell от администратора `wsl --install`, затем перезагрузка")


def check_ports(r: Report) -> None:
    for port, who in ((8000, "FastAPI"), (5432, "PostgreSQL"), (5000, "MLflow"), (8080, "port-forward")):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", port))
            r.add(f"порт {port}", OK, f"свободен ({who})")
        except OSError:
            r.add(f"порт {port}", WARN, f"занят, а нужен для {who}",
                  "Найдите процесс: Windows `netstat -ano | findstr :%d`, macOS/Linux `lsof -i :%d`. "
                  "Часто это старый контейнер: `docker ps`" % (port, port))
        finally:
            s.close()


def check_network(r: Report) -> None:
    targets = (("Docker Hub", "https://registry-1.docker.io/v2/"),
               ("ghcr.io", "https://ghcr.io/v2/"),
               ("PyPI", "https://pypi.org/simple/"),
               ("GitHub", "https://github.com"))
    for name, url in targets:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "mlpro-check-env"})
            urllib.request.urlopen(req, timeout=8)
            r.add(f"сеть: {name}", OK, "доступен")
        except urllib.error.HTTPError as e:
            # 401/403/404 значит хост ответил: нам этого достаточно
            r.add(f"сеть: {name}", OK, f"отвечает (HTTP {e.code})")
        except Exception as e:  # noqa: BLE001
            r.add(f"сеть: {name}", FAIL, f"недоступен: {type(e).__name__}",
                  "Проверьте VPN/прокси. Без Docker Hub и PyPI не соберётся ни один образ")


# ---------------------------------------------------------------- вывод
def render(r: Report) -> str:
    width = max(len(c.name) for c in r.checks) + 2
    lines = ["ML PRO · проверка окружения", "=" * 72]
    for c in r.checks:
        lines.append(f"[{c.status}] {c.name:<{width}} {c.detail}")
        if c.fix and c.status != OK:
            for i, fl in enumerate(c.fix.splitlines()):
                lines.append(("        как починить: " if i == 0 else "        ") + fl.strip())
    fails = sum(c.status == FAIL for c in r.checks)
    warns = sum(c.status == WARN for c in r.checks)
    lines.append("=" * 72)
    if fails == 0 and warns == 0:
        lines.append("Всё готово. Ждём на лекции 7 сентября.")
    elif fails == 0:
        lines.append(f"Критичных проблем нет, предупреждений: {warns}. Можно начинать, но загляните в подсказки.")
    else:
        lines.append(f"Проблем: {fails}, предупреждений: {warns}. Почините по подсказкам выше до первого семинара.")
    return "\n".join(lines)


def main() -> int:
    r = Report()
    for fn in (check_python, check_uv, check_git, check_docker, check_kind_kubectl,
               check_hardware, check_ports, check_network):
        try:
            fn(r)
        except Exception as e:  # noqa: BLE001
            r.add(fn.__name__, WARN, f"проверка упала: {e}")
    text = render(r)
    print(text)
    with open("check_env_report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nОтчёт сохранён в check_env_report.txt")
    return 1 if any(c.status == FAIL for c in r.checks) else 0


if __name__ == "__main__":
    sys.exit(main())