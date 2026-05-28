from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2


def iter_tfrecord_files(input_dir: Path) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No TFRecord files found under {input_dir}.")
    return files


def split_tfrecords(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for package_path in tqdm(iter_tfrecord_files(input_dir), desc="TFRecord packages"):
        dataset = tf.data.TFRecordDataset(
            package_path.as_posix(),
            compression_type="",
            num_parallel_reads=1,
        )
        for tf_data in dataset:
            payload = bytes(tf_data.numpy())
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(payload)
            output_path = output_dir / f"{scenario.scenario_id}.tfrecords"
            if output_path.exists() and not overwrite:
                continue
            with tf.io.TFRecordWriter(output_path.as_posix()) as writer:
                writer.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split WOMD scenario TFRecord packages into one file per scenario."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    split_tfrecords(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
