"""Polite, cached HTTP transport.

Everything that leaves this process goes through :class:`PoliteSession`, which
gives us four things in one place:

* **Retries** with exponential backoff on the transient statuses (429, 5xx).
* **Throttling** to roughly one request per second per host, so a burst of tool
  calls never hammers Petbarn.
* **A disk cache**, so repeated questions about the same product cost nothing
  and a demo stays fast.
* **Stale-while-broken**: if the network fails but we hold an expired copy, we
  serve the expired copy rather than the error. Combined with the bundled
  snapshot in :mod:`petbarn.tools`, this is what keeps the app answering.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config
from .models import Source


def now_iso() -> str:
    """Current UTC time as a second-precision ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _age_seconds(iso_timestamp: str | None) -> float:
    if not iso_timestamp:
        return float("inf")
    try:
        stamp = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return float("inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds()


class FetchError(RuntimeError):
    """A request failed and no usable cached copy was available."""


@dataclass(slots=True)
class FetchResult:
    """A response body plus where and when it came from."""

    url: str
    body: str
    source: Source
    fetched_at: str
    status_code: int | None = None

    def json(self) -> Any:
        return json.loads(self.body)


class PoliteSession:
    """A rate-limited, retrying, disk-cached ``requests`` session."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        ttl_seconds: int | None = None,
        min_interval: float | None = None,
        timeout: float | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.ttl_seconds = config.CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self.min_interval = config.MIN_REQUEST_INTERVAL if min_interval is None else min_interval
        self.timeout = config.REQUEST_TIMEOUT if timeout is None else timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent or config.USER_AGENT,
                "Accept-Language": "en-AU,en;q=0.9",
            }
        )
        retry = Retry(
            total=config.MAX_RETRIES,
            connect=config.MAX_RETRIES,
            read=config.MAX_RETRIES,
            backoff_factor=config.RETRY_BACKOFF_FACTOR,
            status_forcelist=config.RETRY_STATUS_FORCELIST,
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._last_request_at: dict[str, float] = {}
        self._throttle_lock = threading.Lock()

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
        ttl_seconds: int | None = None,
    ) -> FetchResult:
        """GET ``url``, preferring a fresh cached copy when one exists.

        Raises :class:`FetchError` only when the request fails *and* no cached
        copy (fresh or stale) exists.
        """
        cache_path = self._cache_path(url, params)
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds

        if use_cache:
            cached = self._read_cache(cache_path)
            if cached and _age_seconds(cached.fetched_at) < ttl:
                return cached

        try:
            response = self._request(url, params=params, headers=headers)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see below
            # The network is the least reliable part of this system. Before
            # surfacing a failure, fall back to any copy we still hold: a stale
            # price beats no answer at all, and the age travels with the data.
            stale = self._read_cache(cache_path) if use_cache else None
            if stale is not None:
                return stale
            raise FetchError(f"GET {url} failed: {exc}") from exc

        result = FetchResult(
            url=url,
            body=response.text,
            source="live",
            fetched_at=now_iso(),
            status_code=response.status_code,
        )
        if use_cache:
            self._write_cache(cache_path, result, params)
        return result

    def get_json(self, url: str, **kwargs: Any) -> tuple[Any, FetchResult]:
        """GET ``url`` and decode it as JSON, returning the payload and metadata."""
        result = self.get(url, **kwargs)
        try:
            return result.json(), result
        except json.JSONDecodeError as exc:
            raise FetchError(f"GET {url} returned invalid JSON: {exc}") from exc

    def close(self) -> None:
        self._session.close()

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> requests.Response:
        self._wait_turn(url)
        return self._session.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout,
            allow_redirects=True,
        )

    def _wait_turn(self, url: str) -> None:
        """Block until this host's minimum request interval has elapsed."""
        if self.min_interval <= 0:
            return
        host = urlparse(url).netloc
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at.get(host, 0.0)
            delay = self.min_interval - elapsed
            if delay > 0:
                time.sleep(delay)
            self._last_request_at[host] = time.monotonic()

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        canonical = url
        if params:
            canonical += "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> FetchResult | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        body = payload.get("body")
        if not isinstance(body, str):
            return None
        return FetchResult(
            url=payload.get("url", ""),
            body=body,
            source="cache",
            fetched_at=payload.get("fetched_at"),
            status_code=payload.get("status_code"),
        )

    def _write_cache(
        self,
        path: Path,
        result: FetchResult,
        params: dict[str, Any] | None,
    ) -> None:
        payload = {
            "url": result.url,
            "params": params,
            "fetched_at": result.fetched_at,
            "status_code": result.status_code,
            "body": result.body,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash mid-write cannot leave a torn entry
            # that would poison every later read of this URL.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # A read-only or full filesystem must not break the request that
            # already succeeded; the cache is an optimisation, not a dependency.
            pass


@lru_cache(maxsize=1)
def get_session() -> PoliteSession:
    """Return the process-wide session, creating it on first use."""
    return PoliteSession()
