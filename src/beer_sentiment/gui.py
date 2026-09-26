"""Windows desktop front end for CSV import, filtering and model evaluation."""

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
from beer_sentiment.config import PROJECT_ROOT

MODELS = {
    "deepseek": {"label": "DeepSeek", "env": "DEEPSEEK_API_KEY"},
    "qwen": {"label": "Qwen", "env": "DASHSCOPE_API_KEY"},
    "kimi": {"label": "Kimi", "env": "MOONSHOT_API_KEY"},
}
MODEL_LABELS = {name: item["label"] for name, item in MODELS.items()}
MODEL_NAMES_BY_LABEL = {label: name for name, label in MODEL_LABELS.items()}


def command_args(mode: str, values: dict[str, str | bool]) -> list[str]:
    """Turn form values into arguments accepted by the existing CLI."""
    root = Path(str(values["root"])).expanduser().resolve()
    model = MODEL_NAMES_BY_LABEL.get(str(values["model"]), str(values["model"]))
    args = ["--config-dir", str(root / "config"), mode]
    if mode == "run":
        session = "morning" if values.get("session") == "上午" else "afternoon"
        args += [
            "--input-dir",
            str(values["input"]),
            "--output-dir",
            str(values["output"]),
            "--session",
            session,
            "--model",
            model,
            "--no-rag",
        ]
        if values.get("date"):
            args += ["--date", str(values["date"])]
        if values.get("all_time"):
            args.append("--all-time")
    elif mode == "ingest":
        args += [
            "--input-dir",
            str(values["input"]),
            "--data-dir",
            str(values["data"]),
        ]
        if values.get("move"):
            args.append("--move")
    elif mode == "eval":
        args += [
            "--benchmark",
            str(values["benchmark"]),
            "--artifacts-dir",
            str(values["output"]),
            "--models",
            model,
            "--no-rag",
        ]
    else:
        raise ValueError(f"未知操作：{mode}")
    return args


class QueueWriter(io.TextIOBase):
    def __init__(self, events: queue.Queue):
        self.events = events

    def write(self, value: str) -> int:
        if value:
            self.events.put(("log", value))
        return len(value)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("啤酒舆情筛选")
        self.geometry("920x650")
        self.minsize(780, 580)
        self.events: queue.Queue = queue.Queue()
        root = PROJECT_ROOT
        self.vars = {
            "root": tk.StringVar(value=str(root)),
            "input": tk.StringVar(value=str(root / "data")),
            "data": tk.StringVar(value=str(root / "data")),
            "output": tk.StringVar(value=str(root / "output")),
            "benchmark": tk.StringVar(
                value=str(root / "benchmark" / "beer_sentiment_benchmark.jsonl")
            ),
            "session": tk.StringVar(value="上午"),
            "date": tk.StringVar(value=dt.datetime.now().astimezone().date().isoformat()),
            "model": tk.StringVar(value="DeepSeek"),
            "api_key": tk.StringVar(value=""),
            "show_api_key": tk.BooleanVar(value=False),
            "all_time": tk.BooleanVar(value=True),
            "move": tk.BooleanVar(value=False),
        }
        self.mode = tk.StringVar(value="run")
        self.key_cache = {name: "" for name in MODELS}
        self.current_model = "deepseek"
        self.api_key_label: ttk.Label | None = None
        self.api_key_entry: ttk.Entry | None = None
        self._build()
        self.after(100, self._drain)

    def _path_row(self, parent, row: int, label: str, key: str, kind: str = "dir") -> None:
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(
            row=row, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="浏览…", command=lambda: self._browse(key, kind)).grid(
            row=row, column=2
        )

    def _browse(self, key: str, kind: str) -> None:
        current = Path(str(self.vars[key].get()))
        if kind == "file":
            chosen = filedialog.askopenfilename(
                initialdir=str(current.parent),
                filetypes=[("Benchmark", "*.jsonl"), ("所有文件", "*.*")],
            )
        else:
            initial = current if current.is_dir() else PROJECT_ROOT
            chosen = filedialog.askdirectory(initialdir=str(initial))
        if chosen:
            self.vars[key].set(chosen)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text="啤酒舆情筛选", font=("Microsoft YaHei UI", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="导入 CSV，或使用真实大模型进行自动筛选和评测。API 密钥不会保存。",
            foreground="#555",
        ).pack(anchor="w", pady=(3, 12))

        modes = ttk.Frame(outer)
        modes.pack(fill="x")
        for label, value in (
            ("自动筛选", "run"),
            ("导入 CSV", "ingest"),
            ("模型评测", "eval"),
        ):
            ttk.Radiobutton(
                modes,
                text=label,
                value=value,
                variable=self.mode,
                command=self._refresh,
            ).pack(side="left", padx=(0, 22))

        self.form = ttk.Frame(outer, padding=(0, 12, 0, 8))
        self.form.pack(fill="x")
        self.form.columnconfigure(1, weight=1)
        self.options = ttk.Frame(outer)
        self.options.pack(fill="x", pady=(0, 10))

        actions = ttk.Frame(outer)
        actions.pack(fill="x")
        self.run_button = ttk.Button(actions, text="开始运行", command=self._start)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="打开输出目录", command=self._open_output).pack(
            side="left", padx=10
        )
        self.status = tk.StringVar(value="就绪")
        ttk.Label(actions, textvariable=self.status).pack(side="right")

        self.log = tk.Text(
            outer, wrap="word", height=14, state="disabled", font=("Consolas", 10)
        )
        self.log.pack(fill="both", expand=True, pady=(12, 0))
        self._refresh()

    def _refresh(self) -> None:
        for parent in (self.form, self.options):
            for child in parent.winfo_children():
                child.destroy()
        root = Path(str(self.vars["root"].get()))
        mode = self.mode.get()
        if mode == "ingest" and self.vars["input"].get() == str(root / "data"):
            self.vars["input"].set(str(root / "incoming"))
        elif mode != "ingest" and self.vars["input"].get() == str(root / "incoming"):
            self.vars["input"].set(str(root / "data"))
        if mode == "eval" and self.vars["output"].get() == str(root / "output"):
            self.vars["output"].set(str(root / "artifacts"))
        elif mode == "run" and self.vars["output"].get() == str(root / "artifacts"):
            self.vars["output"].set(str(root / "output"))

        self._path_row(self.form, 0, "程序目录", "root")
        if mode == "run":
            self._path_row(self.form, 1, "输入目录", "input")
            self._path_row(self.form, 2, "输出目录", "output")
        elif mode == "ingest":
            self._path_row(self.form, 1, "待导入目录", "input")
            self._path_row(self.form, 2, "数据目录", "data")
        else:
            self._path_row(self.form, 1, "评测数据", "benchmark", "file")
            self._path_row(self.form, 2, "报告目录", "output")
        if mode != "ingest":
            self._model_rows()

        if mode == "run":
            ttk.Label(self.options, text="场次").pack(side="left")
            ttk.Combobox(
                self.options,
                textvariable=self.vars["session"],
                values=("上午", "下午"),
                state="readonly",
                width=6,
            ).pack(side="left", padx=5)
            ttk.Label(self.options, text="日期 YYYY-MM-DD").pack(side="left", padx=(15, 5))
            ttk.Entry(self.options, textvariable=self.vars["date"], width=13).pack(side="left")
            ttk.Checkbutton(
                self.options, text="处理全部时间", variable=self.vars["all_time"]
            ).pack(side="left", padx=12)
        elif mode == "ingest":
            ttk.Checkbutton(
                self.options, text="导入后移动原文件", variable=self.vars["move"]
            ).pack(side="left")

    def _model_rows(self) -> None:
        ttk.Label(self.form, text="模型", width=14).grid(row=3, column=0, sticky="w", pady=5)
        model_box = ttk.Combobox(
            self.form,
            textvariable=self.vars["model"],
            values=tuple(MODEL_LABELS.values()),
            state="readonly",
        )
        model_box.grid(row=3, column=1, sticky="ew", padx=6)
        model_box.bind("<<ComboboxSelected>>", self._on_model_changed)

        self.api_key_label = ttk.Label(self.form, width=14)
        self.api_key_label.grid(row=4, column=0, sticky="w", pady=5)
        self.api_key_entry = ttk.Entry(
            self.form, textvariable=self.vars["api_key"], show="•"
        )
        self.api_key_entry.grid(row=4, column=1, sticky="ew", padx=6)
        ttk.Checkbutton(
            self.form,
            text="显示",
            variable=self.vars["show_api_key"],
            command=self._toggle_api_key,
        ).grid(row=4, column=2)
        self._update_api_key_label()

    def _on_model_changed(self, _event=None) -> None:
        self.key_cache[self.current_model] = str(self.vars["api_key"].get()).strip()
        self.current_model = MODEL_NAMES_BY_LABEL[str(self.vars["model"].get())]
        self.vars["api_key"].set(self.key_cache[self.current_model])
        self._update_api_key_label()

    def _update_api_key_label(self) -> None:
        if self.api_key_label is not None:
            label = MODELS[self.current_model]["label"]
            self.api_key_label.configure(text=f"{label} API 密钥")

    def _toggle_api_key(self) -> None:
        if self.api_key_entry is not None:
            self.api_key_entry.configure(show="" if self.vars["show_api_key"].get() else "•")

    def _start(self) -> None:
        mode = self.mode.get()
        values = {key: var.get() for key, var in self.vars.items()}
        model = MODEL_NAMES_BY_LABEL[str(values["model"])] if mode != "ingest" else ""
        api_key = str(values["api_key"]).strip() if mode != "ingest" else ""
        try:
            root = Path(str(values["root"])).expanduser().resolve()
            if not (root / "config").is_dir():
                raise ValueError(f"程序目录缺少 config 文件夹：{root}")
            if mode != "ingest" and not api_key:
                raise ValueError(f"请输入 {MODELS[model]['label']} API 密钥")
            if mode == "run":
                if not Path(str(values["input"])).is_dir():
                    raise ValueError("请选择有效的输入目录")
                if values["date"]:
                    dt.date.fromisoformat(str(values["date"]))
            elif mode == "eval" and not Path(str(values["benchmark"])).is_file():
                raise ValueError("请选择有效的 Benchmark JSONL 文件")
            elif mode == "ingest":
                if not Path(str(values["input"])).is_dir():
                    raise ValueError("请选择有效的待导入目录")
                if Path(str(values["input"])).resolve() == Path(str(values["data"])).resolve():
                    raise ValueError("待导入目录不能与数据目录相同")
            args = command_args(mode, values)
        except (KeyError, ValueError) as exc:
            messagebox.showerror("输入有误", str(exc))
            return

        if model:
            self.key_cache[model] = api_key
        self.run_button.state(["disabled"])
        self.status.set("运行中…")
        if mode == "ingest":
            self._append("\n▶ 导入 CSV 开始\n")
            worker_args = (args, "", "")
        else:
            action = "自动筛选" if mode == "run" else "模型评测"
            self._append(f"\n▶ {MODELS[model]['label']}：{action}开始\n")
            worker_args = (args, MODELS[model]["env"], api_key)
        threading.Thread(target=self._worker, args=worker_args, daemon=True).start()

    def _worker(self, args: list[str], env_name: str, api_key: str) -> None:
        writer = QueueWriter(self.events)
        previous = os.environ.get(env_name) if env_name else None
        if env_name:
            os.environ[env_name] = api_key
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                cli_main(args)
        except SystemExit as exc:
            self.events.put(("done", (exc.code == 0 or exc.code is None, str(exc))))
        except Exception as exc:  # noqa: BLE001 - surface task errors in the GUI log
            self.events.put(("done", (False, f"{type(exc).__name__}: {exc}")))
        else:
            self.events.put(("done", (True, "")))
        finally:
            if env_name:
                if previous is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = previous

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
        target = Path(str(self.vars["output"].get()))
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)  # Windows desktop app


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
