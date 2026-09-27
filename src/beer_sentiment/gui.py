"""Windows desktop front end for CSV import, filtering and model evaluation."""

from __future__ import annotations

import binascii
import contextlib
import datetime as dt
import io
import os
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from beer_sentiment.cli import main as cli_main
from beer_sentiment.config import PROJECT_ROOT
from beer_sentiment.credentials import load_saved_keys, save_key
from beer_sentiment.io.ingest import recent_csv_files
from beer_sentiment.models import JudgedRow, Label

MODELS = {
    "deepseek": {"label": "DeepSeek", "env": "DEEPSEEK_API_KEY"},
    "qwen": {"label": "Qwen", "env": "DASHSCOPE_API_KEY"},
    "kimi": {"label": "Kimi", "env": "MOONSHOT_API_KEY"},
}
MODEL_LABELS = {name: item["label"] for name, item in MODELS.items()}
MODEL_NAMES_BY_LABEL = {label: name for name, label in MODEL_LABELS.items()}


def command_args(mode: str, values: dict[str, str | bool | list[str]]) -> list[str]:
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
        ]
        for path in values.get("run_files") or []:
            args += ["--input-file", str(path)]
        if values.get("date"):
            args += ["--date", str(values["date"])]
        if values.get("all_time"):
            args.append("--all-time")
    elif mode == "ingest":
        selected_files = values.get("source_files") or []
        args += ["--data-dir", str(values["data"])]
        if selected_files:
            for path in selected_files:
                args += ["--input-file", str(path)]
        else:
            args += ["--input-dir", str(values["source"])]
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


LABEL_TEXT = {Label.BLUE: "蓝色", Label.YELLOW: "黄色", Label.NONE: "不标色"}


def low_confidence_rows(rows: list[JudgedRow]) -> list[JudgedRow]:
    """Select uncertain rows regardless of the model's suggested color."""
    return [row for row in rows if row.low_confidence]


class ReviewDialog(tk.Toplevel):
    """Collect a human decision for each low-confidence model result."""

    def __init__(
        self,
        parent: tk.Tk,
        source_file: str,
        rows: list[JudgedRow],
        on_done: Callable[[dict[int, Label] | None], None],
    ) -> None:
        super().__init__(parent)
        self.title(f"人工复核：{source_file}")
        self.geometry("1040x640")
        self.minsize(800, 500)
        self.transient(parent)
        self.rows = {str(row.prepared.original_row_number): row for row in rows}
        self.decisions: dict[int, Label] = {}
        self.on_done = on_done
        self.choice = tk.StringVar(value="")
        self.progress = tk.StringVar()

        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text=f"{source_file}：以下 {len(rows)} 行模型置信度低，请逐行判断后生成 Excel。",
        ).pack(anchor="w", pady=(0, 10))

        columns = ("line", "model", "confidence", "decision", "preview")
        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse"
        )
        for name, title, width in (
            ("line", "原行号", 70),
            ("model", "模型建议", 90),
            ("confidence", "置信度", 75),
            ("decision", "人工判断", 90),
            ("preview", "内容预览", 600),
        ):
            self.table.heading(name, text=title)
            self.table.column(name, width=width, minwidth=width, stretch=name == "preview")
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scrollbar.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scrollbar.set)
        for row in rows:
            line = str(row.prepared.original_row_number)
            preview = " ".join(row.prepared.combined_text.split())[:110]
            self.table.insert(
                "",
                "end",
                iid=line,
                values=(line, LABEL_TEXT[row.result.label], f"{row.result.confidence:.2f}", "待判断", preview),
            )
        self.table.bind("<<TreeviewSelect>>", self._show_selected)

        ttk.Label(outer, text="完整内容与模型理由").pack(anchor="w", pady=(12, 3))
        self.detail = tk.Text(outer, height=10, wrap="word", state="disabled")
        self.detail.pack(fill="both", expand=True)

        choices = ttk.Frame(outer)
        choices.pack(fill="x", pady=(10, 0))
        ttk.Label(choices, text="人工判断：").pack(side="left")
        for label in (Label.BLUE, Label.YELLOW, Label.NONE):
            ttk.Radiobutton(
                choices,
                text=LABEL_TEXT[label],
                value=label.value,
                variable=self.choice,
                command=self._set_decision,
            ).pack(side="left", padx=(0, 14))
        ttk.Label(choices, textvariable=self.progress).pack(side="right")

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 0))
        self.confirm_button = ttk.Button(actions, text="确认并生成 Excel", command=self._confirm)
        self.confirm_button.pack(side="left")
        ttk.Button(actions, text="取消本文件", command=self._cancel).pack(side="left", padx=8)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self._update_progress()
        first = self.table.get_children()[0]
        self.table.selection_set(first)
        self.table.focus(first)
        self.grab_set()

    def _show_selected(self, _event=None) -> None:
        selected = self.table.selection()
        if not selected:
            return
        line = selected[0]
        row = self.rows[line]
        decision = self.decisions.get(int(line))
        self.choice.set(decision.value if decision else "")
        detail = (
            f"模型建议：{LABEL_TEXT[row.result.label]}，置信度：{row.result.confidence:.2f}\n"
            f"理由：{row.result.reason}\n\n{row.prepared.combined_text}"
        )
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", detail)
        self.detail.configure(state="disabled")

    def _set_decision(self) -> None:
        selected = self.table.selection()
        if not selected:
            return
        line = selected[0]
        decision = Label(self.choice.get())
        self.decisions[int(line)] = decision
        values = list(self.table.item(line, "values"))
        values[3] = LABEL_TEXT[decision]
        self.table.item(line, values=values)
        self._update_progress()

    def _update_progress(self) -> None:
        self.progress.set(f"已判断 {len(self.decisions)} / {len(self.rows)}")
        self.confirm_button.state(["!disabled"] if len(self.decisions) == len(self.rows) else ["disabled"])

    def _confirm(self) -> None:
        if len(self.decisions) != len(self.rows):
            return
        self.grab_release()
        self.destroy()
        self.on_done(self.decisions.copy())

    def _cancel(self) -> None:
        self.grab_release()
        self.destroy()
        self.on_done(None)


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
            "run_files_display": tk.StringVar(value=""),
            "source": tk.StringVar(value=""),
            "data": tk.StringVar(value=str(root / "data")),
            "output": tk.StringVar(value=str(root / "output")),
            "benchmark": tk.StringVar(
                value=str(root / "benchmark" / "beer_sentiment_benchmark_real.jsonl")
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
        self.selected_files: list[str] = []
        self.selected_run_files: list[str] = []
        self.run_files_manual = False
        try:
            saved_keys = load_saved_keys()
        except (OSError, TypeError, ValueError, UnicodeError, binascii.Error):
            saved_keys = {}
        self.key_cache = {name: saved_keys.get(name, "") for name in MODELS}
        self.vars["api_key"].set(self.key_cache["deepseek"])
        self.current_model = "deepseek"
        self.api_key_entry: ttk.Entry | None = None
        self.active_review: ReviewDialog | None = None
        self._select_latest_run_files()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain)

    def _path_row(self, parent, row: int, label: str, key: str, kind: str = "dir") -> None:
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(
            row=row, column=1, sticky="ew", padx=6
        )
        ttk.Button(parent, text="浏览…", command=lambda: self._browse(key, kind)).grid(
            row=row, column=2
        )

    def _source_row(self, parent, row: int) -> None:
        ttk.Label(parent, text="源数据目录", width=14).grid(
            row=row, column=0, sticky="w", pady=5
        )
        ttk.Entry(parent, textvariable=self.vars["source"], state="readonly").grid(
            row=row, column=1, sticky="ew", padx=6
        )
        choices = ttk.Frame(parent)
        choices.grid(row=row, column=2)
        ttk.Button(
            choices, text="选文件…", command=lambda: self._browse("source", "csv")
        ).pack(side="left")
        ttk.Button(
            choices, text="选文件夹…", command=lambda: self._browse("source", "dir")
        ).pack(side="left", padx=(5, 0))

    def _run_files_row(self, parent, row: int) -> None:
        ttk.Label(parent, text="输入文件", width=14).grid(
            row=row, column=0, sticky="w", pady=5
        )
        ttk.Entry(
            parent, textvariable=self.vars["run_files_display"], state="readonly"
        ).grid(row=row, column=1, sticky="ew", padx=6)
        choices = ttk.Frame(parent)
        choices.grid(row=row, column=2)
        ttk.Button(choices, text="选文件…", command=self._browse_run_files).pack(side="left")
        ttk.Button(choices, text="选最新两个", command=self._select_latest_run_files).pack(
            side="left", padx=(5, 0)
        )

    def _select_latest_run_files(self) -> None:
        data_dir = Path(str(self.vars["root"].get())).expanduser() / "data"
        self.selected_run_files = [str(path.resolve()) for path in recent_csv_files(data_dir)]
        self.run_files_manual = False
        self._update_run_files_display()

    def _update_run_files_display(self) -> None:
        names = [Path(path).name for path in self.selected_run_files]
        self.vars["run_files_display"].set(
            "、".join(names) if names else "data 中没有 CSV，请先导入"
        )

    def _browse_run_files(self) -> None:
        initial = Path(str(self.vars["root"].get())).expanduser() / "data"
        chosen = filedialog.askopenfilenames(
            initialdir=str(initial if initial.is_dir() else PROJECT_ROOT),
            filetypes=[("CSV 文件", "*.csv")],
        )
        if chosen:
            self.selected_run_files = list(chosen)
            self.run_files_manual = True
            self._update_run_files_display()

    def _browse(self, key: str, kind: str) -> None:
        current = Path(str(self.vars[key].get()))
        if kind == "csv":
            initial = current if current.is_dir() else current.parent if current.is_file() else PROJECT_ROOT
            chosen_files = filedialog.askopenfilenames(
                initialdir=str(initial),
                filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            )
            if chosen_files:
                self.selected_files = list(chosen_files)
                summary = self.selected_files[0]
                if len(self.selected_files) > 1:
                    summary += f"（共 {len(self.selected_files)} 个文件）"
                self.vars["source"].set(summary)
            return
        if kind == "file":
            chosen = filedialog.askopenfilename(
                initialdir=str(current.parent if current.is_file() else current),
                filetypes=[("Benchmark", "*.jsonl"), ("所有文件", "*.*")],
            )
        else:
            initial = current if current.is_dir() else PROJECT_ROOT
            chosen = filedialog.askdirectory(initialdir=str(initial))
        if chosen:
            if key == "source":
                self.selected_files = []
            self.vars[key].set(chosen)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text="啤酒舆情筛选", font=("Microsoft YaHei UI", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="导入 CSV，或使用真实大模型进行自动筛选和评测。API 密钥可点击保存。",
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
        self.open_button = ttk.Button(actions, text="打开输出目录", command=self._open_output)
        self.open_button.pack(
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
        self.open_button.configure(
            text="打开待导入目录" if mode == "ingest" else "打开输出目录"
        )
        if mode == "eval" and self.vars["output"].get() == str(root / "output"):
            self.vars["output"].set(str(root / "artifacts"))
        elif mode == "run" and self.vars["output"].get() == str(root / "artifacts"):
            self.vars["output"].set(str(root / "output"))

        self._path_row(self.form, 0, "程序目录", "root")
        if mode == "run":
            if not self.run_files_manual:
                self._select_latest_run_files()
            self._run_files_row(self.form, 1)
            self._path_row(self.form, 2, "输出目录", "output")
        elif mode == "ingest":
            self._source_row(self.form, 1)
            self._path_row(self.form, 2, "待导入目录", "data")
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

        ttk.Label(self.form, text="API 密钥", width=14).grid(
            row=4, column=0, sticky="w", pady=5
        )
        self.api_key_entry = ttk.Entry(
            self.form, textvariable=self.vars["api_key"], show="•"
        )
        self.api_key_entry.grid(row=4, column=1, sticky="ew", padx=6)
        key_actions = ttk.Frame(self.form)
        key_actions.grid(row=4, column=2)
        ttk.Checkbutton(
            key_actions,
            text="显示",
            variable=self.vars["show_api_key"],
            command=self._toggle_api_key,
        ).pack(side="left")
        ttk.Button(key_actions, text="保存", command=self._save_api_key).pack(
            side="left", padx=(5, 0)
        )

    def _on_model_changed(self, _event=None) -> None:
        self.key_cache[self.current_model] = str(self.vars["api_key"].get()).strip()
        self.current_model = MODEL_NAMES_BY_LABEL[str(self.vars["model"].get())]
        self.vars["api_key"].set(self.key_cache[self.current_model])

    def _save_api_key(self) -> None:
        key = str(self.vars["api_key"].get()).strip()
        if not key:
            messagebox.showerror("输入有误", "请先输入 API 密钥")
            return
        try:
            save_key(self.current_model, key)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.key_cache[self.current_model] = key
        self.status.set(f"{MODELS[self.current_model]['label']} API 密钥已保存")

    def _toggle_api_key(self) -> None:
        if self.api_key_entry is not None:
            self.api_key_entry.configure(show="" if self.vars["show_api_key"].get() else "•")

    def _start(self) -> None:
        mode = self.mode.get()
        values = {key: var.get() for key, var in self.vars.items()}
        if mode == "ingest" and self.selected_files:
            values["source_files"] = self.selected_files.copy()
        if mode == "run":
            values["run_files"] = self.selected_run_files.copy()
        model = MODEL_NAMES_BY_LABEL[str(values["model"])] if mode != "ingest" else ""
        api_key = str(values["api_key"]).strip() if mode != "ingest" else ""
        try:
            root = Path(str(values["root"])).expanduser().resolve()
            if not (root / "config").is_dir():
                raise ValueError(f"程序目录缺少 config 文件夹：{root}")
            if mode != "ingest" and not api_key:
                raise ValueError(f"请输入 {MODELS[model]['label']} API 密钥")
            if mode == "run":
                if not self.selected_run_files:
                    raise ValueError("请选择要筛选的 CSV 文件")
                for name in self.selected_run_files:
                    path = Path(name)
                    if not path.is_file() or path.suffix.lower() != ".csv":
                        raise ValueError(f"所选 CSV 文件不存在：{path}")
                if not values["all_time"] and values["date"]:
                    dt.date.fromisoformat(str(values["date"]))
            elif mode == "eval" and not Path(str(values["benchmark"])).is_file():
                raise ValueError("本地真实评测数据不存在，请选择有效的 Benchmark JSONL 文件")
            elif mode == "ingest":
                target = Path(str(values["data"])).expanduser()
                selected_files = values.get("source_files") or []
                sources = selected_files or [str(values["source"])]
                if not sources or not str(sources[0]).strip():
                    raise ValueError("请选择源数据目录或 CSV 文件")
                for source_name in sources:
                    source = Path(str(source_name)).expanduser()
                    if not source.exists() or not (source.is_file() or source.is_dir()):
                        raise ValueError(f"源数据目录或文件不存在：{source}")
                    if source.is_file() and source.suffix.lower() != ".csv":
                        raise ValueError(f"源数据文件不是 CSV：{source}")
                    if source.resolve() == target.resolve():
                        raise ValueError("源数据目录不能与待导入目录相同")
                    if source.is_file() and source.parent.resolve() == target.resolve():
                        raise ValueError(f"所选文件已在待导入目录中：{source.name}")
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
            worker_args = (args, "", "", mode)
        else:
            action = "自动筛选" if mode == "run" else "模型评测"
            self._append(f"\n▶ {MODELS[model]['label']}：{action}开始\n")
            worker_args = (args, MODELS[model]["env"], api_key, mode)
        threading.Thread(target=self._worker, args=worker_args, daemon=True).start()

    def _review_rows(self, source_file: str, rows: list[JudgedRow]) -> dict[int, Label] | None:
        low_confidence = low_confidence_rows(rows)
        if not low_confidence:
            self.events.put(("log", f"{source_file}：低置信度 0 行，无需人工复核\n"))
            return {}
        ready = threading.Event()
        response: dict[str, dict[int, Label] | None] = {}
        self.events.put(("review", (source_file, low_confidence, ready, response)))
        ready.wait()
        return response.get("decisions")

    def _worker(self, args: list[str], env_name: str, api_key: str, mode: str) -> None:
        writer = QueueWriter(self.events)
        previous = os.environ.get(env_name) if env_name else None
        if env_name:
            os.environ[env_name] = api_key
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                cli_main(args, review_callback=self._review_rows if mode == "run" else None)
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
                elif kind == "review":
                    source_file, rows, ready, response = payload
                    self._append(f"{source_file}：低置信度 {len(rows)} 行，等待人工判断\n")
                    self.status.set("等待人工复核")

                    def finish(decisions, response=response, ready=ready):
                        response["decisions"] = decisions
                        self.active_review = None
                        self.status.set("运行中…")
                        ready.set()

                    try:
                        self.active_review = ReviewDialog(self, source_file, rows, finish)
                    except Exception as exc:  # noqa: BLE001 - unblock the worker on UI errors
                        self._append(f"无法打开人工复核窗口：{exc}\n")
                        finish(None)
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

    def _on_close(self) -> None:
        if self.active_review is not None:
            self.active_review._cancel()
        self.destroy()

    def _open_output(self) -> None:
        key = "data" if self.mode.get() == "ingest" else "output"
        target = Path(str(self.vars[key].get()))
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)  # Windows desktop app


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
