"""Windows sleep timer with a media pause before the machine suspends.

Run from the project root with the Windows-only media extra, so playback can be
paused first:

    uv run --extra sleep-timer python experiments/sleep-timer/sleep-timer-app-win.py

Without it the timer still works, it just leaves playing media alone.
"""

import asyncio
import ctypes
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

# Sun Valley theme: gives the ttk widgets a native Windows 11 look and, unlike
# the classic tk widgets, an actual dark mode. Optional, so the app still runs
# (in the default light theme) if the extra is not installed.
try:
    import sv_ttk
except ImportError:
    sv_ttk = None


def windows_apps_use_dark() -> bool:
    """True if Windows is set to dark app mode, else False.

    Reads the same registry value the Settings app writes under
    Personalize: `AppsUseLightTheme` is 1 for light and 0 for dark. Any
    missing key/value (or a non-Windows host, where winreg does not exist)
    falls back to dark, matching the app's default before this was added.
    """
    try:
        import winreg
    except ImportError:
        return True
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            light, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
    except OSError:
        return True
    return light == 0

# System Media Transport Controls: the same API the hardware media keys use.
# Optional, so the timer is still usable if the extra is not installed. The two
# names sit in separate import statements because they are aliased.
try:
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as MediaManager,
    )
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
    )
except ImportError:
    MediaManager = None

# Queue payload asking the Tk thread to suspend the machine.
SUSPEND = object()


async def pause_playing_media() -> int:
    """Pause every media session that is currently playing.

    Returns how many sessions were paused. Sessions that are already paused or
    stopped are left alone, so this never starts playback the way a
    play/pause media key would.
    """
    manager = await MediaManager.request_async()
    paused = 0
    for session in manager.get_sessions():
        if session.get_playback_info().playback_status != PlaybackStatus.PLAYING:
            continue
        if await session.try_pause_async():
            paused += 1
    return paused


class SleepTimerApp:
    # Grace period that lets the "Sleeping..." label paint before suspending.
    SLEEP_LABEL_DELAY_S = 0.5
    # How often the Tk thread drains updates posted by the countdown thread.
    STATUS_POLL_MS = 100
    # Bound on waiting for a stopped countdown thread to finish.
    JOIN_TIMEOUT_S = 2.0
    # Bound on waiting for the media pause, so a hung WinRT call cannot block
    # the suspend forever.
    MEDIA_PAUSE_TIMEOUT_S = 3.0
    # How often to re-read the Windows app-mode setting, so a change made in
    # Settings while this is open is picked up without a restart.
    THEME_POLL_MS = 2000

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sleep Timer")
        self.root.geometry("300x300")
        self.root.resizable(False, False)

        # Match the Windows "app mode" setting, then keep following it live via
        # _sync_theme. classic tk.* widgets would ignore this; only ttk ones are
        # themed.
        self._closing = False
        self._current_theme = None
        if sv_ttk is not None:
            self._current_theme = self._detect_theme()
            sv_ttk.set_theme(self._current_theme)

        self.timer_thread = None
        self.is_running = False
        self.remaining_seconds = 0
        # Bumped on every start and every reset, so a straggling worker thread
        # can tell its run is over and stop touching shared state.
        self.run_token = 0
        # Set by a reset to wake the countdown thread out of its tick at once.
        self.stop_event = threading.Event()
        # The countdown thread never calls into Tk: such a call blocks until the
        # Tk thread dispatches it, which would stall the join() in _reset_idle.
        # It posts updates here and the Tk thread applies them.
        self.status_queue = queue.Queue()

        # Minutes input field
        ttk.Label(root, text="Time until sleep in mins:", font=("Segoe UI", 10)).pack(
            pady=(15, 5)
        )
        self.minutes_entry = ttk.Entry(
            root, font=("Segoe UI", 12), justify="center", width=10
        )
        self.minutes_entry.insert(0, "30")
        self.minutes_entry.pack(pady=5)

        # Countdown
        self.label_status = ttk.Label(
            root, text="Timer is not launched", font=("Segoe UI", 11, "bold")
        )
        self.label_status.pack(pady=10)

        # Control buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn_start = ttk.Button(
            btn_frame, text="Start", command=self.start_timer, width=10
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(
            btn_frame,
            text="Cancel",
            command=self.stop_timer,
            state=tk.DISABLED,
            width=10,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Closing the window must stop the countdown thread, not orphan it.
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._pump_status()
        self._sync_theme()

    def _detect_theme(self) -> str:
        """Name of the sv_ttk theme matching the Windows app-mode setting."""
        return "dark" if windows_apps_use_dark() else "light"

    def _sync_theme(self):
        """Re-apply the theme if the Windows app-mode setting changed.

        Polled from the Tk thread rather than driven by WM_SETTINGCHANGE: no
        window-procedure subclassing, and it also catches changes made while
        the app was in the background. Restyling only on an actual change
        avoids needless style rebuilds every tick.
        """
        if self._closing or sv_ttk is None:
            return
        theme = self._detect_theme()
        if theme != self._current_theme:
            self._current_theme = theme
            sv_ttk.set_theme(theme)
        self.root.after(self.THEME_POLL_MS, self._sync_theme)

    def start_timer(self):
        try:
            minutes = float(self.minutes_entry.get())
            if minutes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Input a correct number")
            return

        # Round instead of truncating, and never go below 1s. Tiny inputs
        # (e.g. 0.016 min = 0.96s) would otherwise land on 0, skip the
        # countdown loop entirely, and suspend immediately.
        self.remaining_seconds = max(1, round(minutes * 60))
        self.stop_event.clear()
        self.run_token += 1
        token = self.run_token
        self.is_running = True

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.minutes_entry.config(state=tk.DISABLED)

        # Invoke countdown in a separate thread
        self.timer_thread = threading.Thread(
            target=self._run_timer, args=(token,), daemon=True
        )
        self.timer_thread.start()

    def _is_current(self, token: int) -> bool:
        """True while `token` still identifies the run that is active."""
        return token == self.run_token and self.is_running

    def _run_timer(self, token: int):
        while self._is_current(token) and self.remaining_seconds > 0:
            mins, secs = divmod(self.remaining_seconds, 60)
            self.status_queue.put((token, f"Left: {mins:02d}:{secs:02d}"))

            # Wakes the moment a reset sets stop_event, instead of sleeping out
            # the rest of the tick.
            if self.stop_event.wait(1):
                return
            if not self._is_current(token):
                # Cancelled mid-tick. The reset owns the state now, so return
                # instead of decrementing; the clamp below is belt-and-braces
                # for the sliver between the check and the assignment.
                return
            self.remaining_seconds = max(0, self.remaining_seconds - 1)

        if self._is_current(token) and self.remaining_seconds <= 0:
            # One queue item per poll pass, so the label gets a chance to paint
            # before the suspend item is handled.
            self.status_queue.put((token, "Sleeping..."))
            if not self.stop_event.wait(self.SLEEP_LABEL_DELAY_S):
                self.status_queue.put((token, SUSPEND))

    def _pump_status(self):
        """Apply one pending worker update, on the Tk thread.

        One item per pass: Tk only repaints once a callback returns, so
        handling the label and the suspend in the same pass would suspend the
        machine with "Sleeping..." still unpainted.
        """
        try:
            token, payload = self.status_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            if self._is_current(token):
                if payload is SUSPEND:
                    self.sleep_windows()
                else:
                    self.label_status.config(text=payload)

        self.root.after(self.STATUS_POLL_MS, self._pump_status)

    def stop_timer(self):
        self._reset_idle("Timer was stopped")

    def on_close(self):
        """Window close: join the countdown thread, then tear Tk down.

        The widgets are not reset first: they are about to be destroyed, so
        only the teardown half of _reset_idle is meaningful here.
        """
        # Stops the theme poll from rescheduling after the interpreter is gone.
        self._closing = True
        self._stop_worker()
        self.root.destroy()

    def _stop_worker(self):
        """End the current run and wait for its thread to exit."""
        # Bumping the token makes any straggling worker see a stale token and
        # stop writing to shared state.
        self.run_token += 1
        self.is_running = False
        self.stop_event.set()
        self._join_worker()

    def _reset_idle(self, status_text: str):
        """Stop the countdown thread and hand the controls back."""
        self._stop_worker()
        self.remaining_seconds = 0
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.minutes_entry.config(state=tk.NORMAL)
        self.label_status.config(text=status_text)

    def _join_worker(self):
        """Wait for the countdown thread to exit.

        Safe from the Tk thread only because the worker never calls into Tk:
        a thread blocked inside root.after() is waiting for this thread to
        dispatch, so joining it here would stall for the whole timeout.
        """
        if self.timer_thread is not None and self.timer_thread.is_alive():
            self.timer_thread.join(self.JOIN_TIMEOUT_S)

    def _pause_media(self):
        """Ask playing media to pause, giving up rather than stalling forever.

        This runs on the Tk thread, so the WinRT work gets its own throwaway
        thread: a wedged asyncio/WinRT call here would freeze the UI and hold
        up the suspend with it.
        """
        if MediaManager is None:
            print(
                "Media pause unavailable: run with `uv run --extra sleep-timer "
                "python experiments/sleep-timer/sleep-timer-app-win.py`",
                file=sys.stderr,
            )
            return

        finished = threading.Event()

        def worker():
            try:
                asyncio.run(pause_playing_media())
            except (OSError, RuntimeError) as exc:
                # WinRT reports failed calls as OSError. Best effort either way:
                # a pause we could not deliver must not hold up the suspend.
                print(f"Could not pause media: {exc!r}", file=sys.stderr)
            finally:
                finished.set()

        threading.Thread(target=worker, daemon=True).start()
        if not finished.wait(self.MEDIA_PAUSE_TIMEOUT_S):
            print("Media pause timed out; suspending anyway", file=sys.stderr)

    def sleep_windows(self):
        # Pause playback first: SetSuspendState is called below with bForce=1,
        # which skips the PBT_APMQUERYSUSPEND broadcast, so players get no
        # chance to pause themselves before the machine goes down.
        self._pause_media()

        #  Windows API call: SetSuspendState(bHibernate, bForce, bDisableWakeEvent)
        suspended = bool(ctypes.windll.powrprof.SetSuspendState(0, 1, 0))

        # Control reaches here because the call failed, or because the machine
        # suspended and has now woken up. Reset in either case, otherwise a
        # refused suspend leaves the buttons disabled with no way back.
        self._reset_idle("Timer is not launched")
        if not suspended:
            messagebox.showerror("Error", "Could not put the system to sleep")


if __name__ == "__main__":
    root = tk.Tk()
    app = SleepTimerApp(root)
    root.mainloop()
