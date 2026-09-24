from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace

from devices.ccd_models import DeviceError, SensorRelaySettings, SensorRelayStats, TriggerMode, TriggerSettings
from devices.interfaces import DigitalIo

# ============================================================
# Sensor relay (「Sensor 中繼」).
# The Sensor reaches the grabber only through a PCIe-1730 DI -> program -> DO path, so VisionFlow
# polls the DI and acts on each inactive -> active edge:
#   forward: External Trigger One Frame; pulse the DO wired to the grabber frame-trigger input from
#            the relay thread itself, so no GUI-thread latency is added.
#   snap:    Software Trigger; hand the edge to the GUI thread, which starts one Snap().
# A Sensor already active when the relay starts must clear first, so a part sitting in front of the
# Sensor never fires at start. Software latency is bounded by the poll interval plus OS scheduling;
# `max_poll_gap_ms` reports the worst gap actually observed.
# ============================================================

MODE_FORWARD = "forward"
MODE_SNAP = "snap"
MODE_LABELS = {MODE_FORWARD: "DI 轉 DO 給擷取卡", MODE_SNAP: "DI 觸發軟體擷取"}
# Sleeps shorter than this are spun on the clock: Windows sleep granularity is about 1 ms.
_SPIN_BELOW_SEC = 0.002


def relay_mode(settings: SensorRelaySettings, hardware_trigger: TriggerSettings | None) -> str | None:
    """Which relay action the trigger settings written to the camera call for, or None."""
    if not settings.enabled or hardware_trigger is None:
        return None
    if hardware_trigger.mode == TriggerMode.EXTERNAL and hardware_trigger.external_frame_one_frame:
        return MODE_FORWARD
    if hardware_trigger.mode == TriggerMode.SOFTWARE:
        return MODE_SNAP
    return None


class SensorRelay:
    def __init__(
        self,
        io: DigitalIo,
        settings: SensorRelaySettings,
        mode: str,
        on_edge: Callable[[], None] = lambda: None,
        on_error: Callable[[DeviceError], None] = lambda _error: None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if mode not in MODE_LABELS:
            raise ValueError(f"unknown relay mode {mode!r}")
        self.io = io
        self.settings = settings.normalized()
        self.mode = mode
        self._on_edge = on_edge
        self._on_error = on_error
        self._clock = clock
        self._sleep = sleep
        self._stats_lock = threading.Lock()
        self._stats = SensorRelayStats(mode=mode)
        self._previous_active: bool | None = None
        self._last_edge_at: float | None = None
        self._last_poll_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---- state ---------------------------------------------------------------------------------
    def stats(self) -> SensorRelayStats:
        with self._stats_lock:
            return replace(self._stats, running=self.is_running)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def _update(self, **changes) -> None:
        fixed = changes.pop("set", {})
        with self._stats_lock:
            values = {name: getattr(self._stats, name) + value for name, value in changes.items()}
            values.update(fixed)
            self._stats = replace(self._stats, **values)

    # ---- one poll ------------------------------------------------------------------------------
    def _write_do(self, active: bool) -> None:
        s = self.settings
        self.io.write_bit(s.do_port, s.do_bit, active != s.do_active_low)

    def _wait(self, seconds: float) -> None:
        deadline = self._clock() + max(0.0, seconds)
        remaining = deadline - self._clock()
        if remaining > _SPIN_BELOW_SEC:
            self._sleep(remaining - _SPIN_BELOW_SEC)
        while self._clock() < deadline:
            pass

    def prepare(self) -> None:
        """Open the card and park the DO inactive; raises `DeviceError`."""
        self.io.connect(self.settings)
        if self.mode == MODE_FORWARD:
            self._write_do(False)

    def pulse_once(self) -> None:
        """Drive the DO active for `pulse_ms`, then inactive again (also used by the manual test)."""
        self._write_do(True)
        try:
            self._wait(self.settings.pulse_ms / 1000.0)
        finally:
            self._write_do(False)

    def read_input(self) -> tuple[bool, bool]:
        """(active, raw level) of the Sensor DI; the card must be open."""
        s = self.settings
        raw = self.io.read_bit(s.di_port, s.di_bit)
        return raw != s.di_active_low, raw

    def step(self) -> bool:
        """Poll the DI once and act on an edge. Returns True when an edge was accepted."""
        s = self.settings
        now = self._clock()
        gap_ms = 0.0 if self._last_poll_at is None else (now - self._last_poll_at) * 1000.0
        self._last_poll_at = now
        active, _raw = self.read_input()
        previous, self._previous_active = self._previous_active, active
        with self._stats_lock:
            self._stats = replace(
                self._stats,
                polls=self._stats.polls + 1,
                di_active=active,
                max_poll_gap_ms=max(self._stats.max_poll_gap_ms, gap_ms),
            )
        if previous is None or previous or not active:
            return False
        if self._last_edge_at is not None and (now - self._last_edge_at) * 1000.0 < s.min_interval_ms:
            self._update(ignored_edges=1)
            return False
        self._last_edge_at = now
        if self.mode == MODE_FORWARD:
            self.pulse_once()
            self._update(edges=1, pulses=1)
        else:
            self._update(edges=1)
        self._on_edge()
        return True

    # ---- thread --------------------------------------------------------------------------------
    def start(self) -> None:
        """Open the card on the caller's thread (errors raise), then poll on a relay thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
        self.prepare()
        with self._lock:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="ccd-sensor-relay", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self) -> None:
        poll_sec = self.settings.poll_interval_ms / 1000.0
        try:
            while not self._stop_event.is_set():
                self.step()
                if poll_sec > 0:
                    # Plain sleep between polls (Python 3.11+ uses a high-resolution timer on Windows);
                    # only the DO pulse width is spun, so polling does not keep a core busy.
                    self._sleep(poll_sec)
        except DeviceError as exc:
            self._stop_event.set()
            self._update(set={"error": str(exc)})
            self._on_error(exc)
        finally:
            if self.mode == MODE_FORWARD:
                try:
                    self._write_do(False)
                except DeviceError:
                    pass
