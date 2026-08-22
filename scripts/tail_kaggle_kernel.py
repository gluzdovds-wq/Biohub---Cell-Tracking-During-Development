"""Print the latest matching live Kaggle-kernel log lines without staying attached."""

from __future__ import annotations

import argparse
import re
import time
from collections import deque

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, KaggleEnv


def tail(kernel: str, pattern: str | None, count: int, idle_timeout: float) -> list[str]:
    matcher = re.compile(pattern, re.IGNORECASE) if pattern else None
    selected: deque[str] = deque(maxlen=count)
    api = KaggleApi()
    api.authenticate()
    owner, slug = api._split_kernel(kernel)

    with api.build_kaggle_client() as kaggle:
        http = kaggle._http_client
        http._init_session()
        base = http._endpoint if http._env == KaggleEnv.PROD else f"{http._endpoint}/api"
        url = f"{base}/v1/kernels/logs/stream/{owner}/{slug}"
        headers = dict(http._session.headers)
        headers["Accept"] = "text/event-stream, */*"
        headers.pop("Content-Type", None)
        response = http._session.get(
            url,
            stream=True,
            headers=headers,
            auth=http._session.auth,
            # The Kaggle proxy can take several seconds to return SSE headers.
            # A ten-second floor avoids confusing that startup latency with an idle tail.
            timeout=(10, max(10.0, idle_timeout)),
        )
        response.raise_for_status()
        try:
            content_type = (response.headers.get("Content-Type") or "").lower()
            events = (
                api._iter_sse_events(response)
                if content_type.startswith("text/event-stream")
                else api._iter_blob_lines(response)
            )
            for event in events:
                data = event.get("data")
                if data is None:
                    continue
                for line in data.replace("\r", "\n").splitlines():
                    line = line.strip()
                    if line and (matcher is None or matcher.search(line)):
                        selected.append(line)
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            # Reaching the current end of a live log normally produces an idle read timeout.
            pass
        finally:
            response.close()
    return list(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Kaggle kernel as owner/slug")
    parser.add_argument("--pattern", help="Case-insensitive regular expression")
    parser.add_argument("--lines", type=int, default=30)
    parser.add_argument("--idle-timeout", type=float, default=3.0)
    args = parser.parse_args()
    if args.lines < 1 or args.idle_timeout <= 0:
        parser.error("--lines and --idle-timeout must be positive")
    lines = None
    last_error: requests.exceptions.RequestException | None = None
    for attempt in range(3):
        try:
            lines = tail(args.kernel, args.pattern, args.lines, args.idle_timeout)
            break
        except requests.exceptions.RequestException as error:
            last_error = error
            if attempt < 2:
                time.sleep(2)
    if lines is None:
        assert last_error is not None
        raise last_error
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
