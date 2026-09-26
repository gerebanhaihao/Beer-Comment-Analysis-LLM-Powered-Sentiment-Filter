"""Small Windows desktop front end for the existing CLI workflows."""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from beer_sentiment.cli import main as cli_main
from beer_sentiment.config import PROJECT_ROOT, load_config


def command_args(mode: str, values: dict[str, str | bool]) -> list[str]:
    """Turn form values into the same arguments accepted by the CLI."""
    root = Path(str(values["root"])).expanduser().resolve()
    args = ["--config-dir", str(root / "config"), mode]
    session = "morning" if values.get("session") == "上午" else "afternoon"
    if mode == "ingest":
        args += ["--input-dir", str(values["input"]), "--data-dir", str(values["data"])]
        if values.get("move"):
            args.append("--move")
    elif mode == "prepare":
        args += ["--input-dir", str(values["input"]), "--output-dir", str(values["output"]), "--session", session]
        if values.get("date"):
            args += ["--date", str(values["date"])]
    elif mode == "build":
        args += ["--review-csv", str(values["review"]), "--output-dir", str(values["output"]), "--session", session]
    elif mode == "run":
        args += ["--input-dir", str(values["input"]), "--output-dir", str(values["output"]), "--session", session, "--model", str(values["model"])]
        if values.get("date"):
            args += ["--date", str(values["date"])]
        if values.get("all_time"):
            args.append("--all-time")
        if values.get("no_rag"):
            args.append("--no-rag")
    elif mode == "eval":
        args += ["--benchmark", str(values["benchmark"]), "--artifacts-dir", str(values["output"]), "--models", str(values["model"])]
    else:
        raise ValueError(f"未知操作：{mode}")
    return args


class QueueWriter(io.TextIOBase):
    def __init__(self, events: queue.Queue):
        self.events = events

    def write(self, text: str) -> int:
        if text:
            self.events.put(("log", text))
        return len(text)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("啤酒舆情筛选")
        self.geometry("920x690")
        self.minsize(760, 580)
        self.events: queue.Queue = queue.Queue()
        root = PROJECT_ROOT
        self.vars = {
            "root": tk.StringVar(value=str(root)),
            "input": tk.StringVar(value=str(root / "data")),
            "data": tk.StringVar(value=str(root / "data")),
            "output": tk.StringVar(value=str(root / "output")),
            "review": tk.StringVar(value=""),
            "benchmark": tk.StringVar(value=str(root / "benchmark" / "beer_sentiment_benchmark.jsonl")),
            "session": tk.StringVar(value="上午"),
            "date": tk.StringVar(value=dt.datetime.now().astimezone().date().isoformat()),
            "model": tk.StringVar(value="mock"),
            "all_time": tk.BooleanVar(value=True),
            "no_rag": tk.BooleanVar(value=True),
            "move": tk.BooleanVar(value=False),
        }
        self.mode = tk.StringVar(value="run")
        self._build()
        self.after(100, self._drain)

    def _row(self, parent, row: int, label: str, key: str, kind: str = "dir") -> None:
        ttk.Label(parent, text=label, width=13).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="浏览…", command=lambda: self._browse(key, kind)).grid(row=row, column=2)

    def _browse(self, key: str, kind: str) -> None:
        current = str(self.vars[key].get())
        if kind == "file":
            chosen = filedialog.askopenfilename(initialdir=str(Path(current).parent), filetypes=[("CSV/JSONL", "*.csv *.jsonl"), ("所有文件", "*.*")])
        else:
            chosen = filedialog.askdirectory(initialdir=current if Path(current).is_dir() else str(PROJECT_ROOT))
        if chosen:
            self.vars[key].set(chosen)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="啤酒舆情筛选", font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w")
        ttk.Label(outer, text="选择操作、文件夹和模型，运行结果会显示在下方。模拟模型仅用于流程演示。", foreground="#555").pack(anchor="w", pady=(3, 12))
        modes = ttk.Frame(outer)
        modes.pack(fill="x")
        for label, value in [("自动筛选", "run"), ("生成待筛选 CSV", "prepare"), ("人工标注转 Excel", "build"), ("导入 CSV", "ingest"), ("模型评测", "eval")]:
            ttk.Radiobutton(modes, text=label, value=value, variable=self.mode, command=self._refresh).pack(side="left", padx=(0, 16))
        self.form = ttk.Frame(outer, padding=(0, 12, 0, 8))
        self.form.pack(fill="x")
        self.form.columnconfigure(1, weight=1)
        self.options = ttk.Frame(outer)
        self.options.pack(fill="x", pady=(0, 10))
        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        self.run_button = ttk.Button(actions, text="开始运行", command=self._start)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="打开输出目录", command=self._open_output).pack(side="left", padx=10)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(actions, textvariable=self.status).pack(side="right")
        self.log = tk.Text(outer, wrap="word", height=14, state="disabled", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, pady=(12, 0))
        self._refresh()

    def _refresh(self) -> None:
        for parent in (self.form, self.options):
            for child in parent.winfo_children():
                child.destroy()
        mode = self.mode.get()
        root = Path(self.vars["root"].get())
        if mode == "ingest" and self.vars["input"].get() == str(root / "data"):
            self.vars["input"].set(str(root / "incoming"))
        elif mode != "ingest" and self.vars["input"].get() == str(root / "incoming"):
            self.vars["input"].set(str(root / "data"))
        if mode == "eval" and self.vars["output"].get() == str(root / "output"):
            self.vars["output"].set(str(root / "artifacts"))
        elif mode != "eval" and self.vars["output"].get() == str(root / "artifacts"):
            self.vars["output"].set(str(root / "output"))
        self._row(self.form, 0, "程序目录", "root")
        if mode == "ingest":
            self._row(self.form, 1, "待导入目录", "input")
            self._row(self.form, 2, "数据目录", "data")
            ttk.Checkbutton(self.options, text="导入后移动原文件", variable=self.vars["move"]).pack(side="left")
        elif mode == "build":
            self._row(self.form, 1, "待筛选 CSV", "review", "file")
            self._row(self.form, 2, "输出目录", "output")
            self._session_option()
        elif mode == "eval":
            self._row(self.form, 1, "评测数据", "benchmark", "file")
            self._row(self.form, 2, "报告目录", "output")
            self._model_option()
        else:
            self._row(self.form, 1, "输入目录", "input")
            self._row(self.form, 2, "输出目录", "output")
            self._session_option()
            ttk.Label(self.options, text="日期 YYYY-MM-DD").pack(side="left", padx=(15, 5))
            ttk.Entry(self.options, textvariable=self.vars["date"], width=13).pack(side="left")
            if mode == "run":
                self._model_option()
                ttk.Checkbutton(self.options, text="处理全部时间", variable=self.vars["all_time"]).pack(side="left", padx=10)
                ttk.Checkbutton(self.options, text="关闭 RAG", variable=self.vars["no_rag"]).pack(side="left")

    def _session_option(self) -> None:
        ttk.Label(self.options, text="场次").pack(side="left")
        ttk.Combobox(self.options, textvariable=self.vars["session"], values=("上午", "下午"), state="readonly", width=6).pack(side="left", padx=5)

    def _model_option(self) -> None:
        ttk.Label(self.options, text="模型").pack(side="left", padx=(15, 5))
        try:
            names = list(load_config(Path(self.vars["root"].get()) / "config").models)
        except (OSError, ValueError, KeyError):
            names = ["mock", "deepseek-v4", "qwen-max", "kimi-k3", "deepseek"]
        ttk.Combobox(self.options, textvariable=self.vars["model"], values=names, width=16).pack(side="left")

    def _start(self) -> None:
        mode = self.mode.get()
        values = {key: var.get() for key, var in self.vars.items()}
        try:
            root = Path(str(values["root"])).expanduser().resolve()
            if not (root / "config").is_dir():
                raise ValueError(f"程序目录缺少 config 文件夹：{root}")
            if values["date"] and mode in ("run", "prepare"):
                dt.date.fromisoformat(str(values["date"]))
            if mode == "build" and not Path(str(values["review"])).is_file():
                raise ValueError("请选择待筛选 CSV")
            args = command_args(mode, values)
        except ValueError as exc:
            messagebox.showerror("输入有误", str(exc))
            return
        self.run_button.state(["disabled"])
        self.status.set("运行中…")
        self._append(f"\n▶ {mode} 开始\n")
        threading.Thread(target=self._worker, args=(args,), daemon=True).start()

    def _worker(self, args: list[str]) -> None:
        writer = QueueWriter(self.events)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                cli_main(args)
        except SystemExit as exc:
            self.events.put(("done", (exc.code == 0 or exc.code is None, str(exc))))
        except Exception as exc:  # noqa: BLE001 - surface task errors in the GUI log
            self.events.put(("done", (False, f"{type(exc).__name__}: {exc}")))
        else:
            self.events.put(("done", (True, "")))

    def _append(self, value: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", value)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append(payload)
                else:
                    success, detail = payload
                    if detail:
                        self._append(detail + "\n")
                    self._append("✓ 完成\n" if success else "✗ 运行失败\n")
                    self.status.set("完成" if success else "失败，请查看日志")
                    self.run_button.state(["!disabled"])
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _open_output(self) -> None:
        target = Path(self.vars["output"].get())
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)  # Windows desktop app


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
