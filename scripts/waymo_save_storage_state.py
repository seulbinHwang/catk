#!/usr/bin/env python3

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "secrets" / "waymo" / "waymo_storage_state.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a browser, let the user log in to Waymo manually, and save a "
            "Playwright storage_state.json file for later headless submissions."
        )
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT_PATH.as_posix(),
        help=(
            "Where to save storage_state.json. Defaults to the repo-local canonical path "
            f"{_DEFAULT_OUTPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--url",
        default="https://waymo.com/open/challenges/2025/sim-agents/",
        help="Challenge page to open before saving storage state.",
    )
    parser.add_argument(
        "--browser-name",
        default="chromium",
        choices=["chromium", "firefox", "webkit"],
        help="Playwright browser engine to launch.",
    )
    parser.add_argument(
        "--browser-channel",
        default=None,
        help="Optional browser channel such as chrome or msedge.",
    )
    parser.add_argument(
        "--browser-executable-path",
        default=None,
        help="Optional absolute path to a browser executable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required. Install it and run `python -m playwright install chromium`."
        ) from exc

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser_type = getattr(playwright, args.browser_name)
        launch_kwargs = {
            "headless": False,
        }
        if args.browser_channel:
            launch_kwargs["channel"] = args.browser_channel
        if args.browser_executable_path:
            launch_kwargs["executable_path"] = args.browser_executable_path
        if args.browser_name == "chromium":
            launch_kwargs["chromium_sandbox"] = False

        browser = browser_type.launch(**launch_kwargs)
        context = browser.new_context()
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded")

        print("1. Sign in to Waymo with the target Google account.")
        print("2. Accept challenge terms if Waymo asks for them.")
        print("3. Confirm the Sim Agents page shows the submission UI instead of the sign-in gate.")
        input("Press Enter here after the page is fully ready to save storage_state.json...")

        context.storage_state(path=output_path.as_posix())
        print(f"Saved Waymo storage state to {output_path}")

        browser.close()


if __name__ == "__main__":
    main()
