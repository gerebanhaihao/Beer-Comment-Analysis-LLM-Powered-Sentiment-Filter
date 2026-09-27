import csv

import pytest

from beer_sentiment.cli import build_parser, main
from beer_sentiment.config import PROJECT_ROOT


def test_eval_defaults_to_local_real_benchmark():
    args = build_parser().parse_args(["eval"])
    assert args.benchmark == "benchmark/beer_sentiment_benchmark_real.jsonl"


def test_cli_eval_mock(tmp_path, config):
    benchmark = PROJECT_ROOT / "benchmark" / "beer_sentiment_benchmark.jsonl"
    main(
        [
            "--config-dir",
            str(config.config_dir),
            "eval",
            "--model",
            "mock",
            "--benchmark",
            str(benchmark),
            "--artifacts-dir",
            str(tmp_path),
        ]
    )
    assert list(tmp_path.rglob("*.json"))


def test_cli_ingest_empty_source_fails(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--input-dir", str(source), "--data-dir", str(tmp_path / "incoming")])
    assert exc.value.code == 1
    assert "源数据目录没有 CSV" in capsys.readouterr().out


def test_cli_ingest_requires_a_selected_source(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["ingest"])
    assert exc.value.code == 2
    assert "请选择源数据目录" in capsys.readouterr().out


def test_cli_ingest_multiple_selected_files(tmp_path):
    source = tmp_path / "source"
    data = tmp_path / "data"
    source.mkdir()
    for name in ("one.csv", "two.csv", "unselected.csv"):
        with (source / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["标题", "正文"])
            writer.writerow([name, "百威太难喝"])

    main(
        [
            "ingest",
            "--input-file",
            str(source / "one.csv"),
            "--input-file",
            str(source / "two.csv"),
            "--data-dir",
            str(data),
        ]
    )

    assert (data / "one.csv").exists()
    assert (data / "two.csv").exists()
    assert not (data / "unselected.csv").exists()
