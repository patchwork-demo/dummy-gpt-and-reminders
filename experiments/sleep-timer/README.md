# sleep-timer — quick guide

A small Tkinter app that puts Windows to sleep after a countdown, pausing
whatever is playing first. Windows-only.

## Setup

Media pausing comes from the optional `sleep-timer` extra:

```sh
uv sync --extra sleep-timer
```

Without it the timer still works — it just leaves playing media alone and prints
a hint to stderr instead.

## Run

From the project root (uses the project `.venv`, Python 3.14):

```sh
uv run --extra sleep-timer python experiments/sleep-timer/sleep-timer-app-win.py
```

Type the minutes until sleep and press **Start**. The label counts down, switches
to `Sleeping...`, and the machine suspends. **Cancel** stops the countdown;
a *sleep* that the system refuses pops up an error instead of leaving the
buttons stuck.

## How it works

- The countdown runs on a worker thread that never touches Tk — such a call
  would block until the Tk thread dispatches it. It posts `(token, text)` items
  to a queue and the Tk thread applies them.
- `_pump_status()` drains **one** item per 100 ms pass, because Tk only repaints
  once a callback returns: handling the label and the suspend in the same pass
  would suspend with `Sleeping...` still unpainted.
- `run_token` + `stop_event` let a Cancel or a window close end the run
  immediately and make a straggling worker notice its run is over.
- At zero the app calls `SetSuspendState(0, 1, 0)` — no hibernate, **forced**,
  wake events enabled.
- Because `bForce=1` skips the `PBT_APMQUERYSUSPEND` broadcast, players get no
  chance to pause themselves, so the app pauses them first over System Media
  Transport Controls (the API behind the hardware media keys). Only sessions that
  are actually **playing** are paused, so nothing is started by accident. The
  pause runs on a throwaway thread with a 3 s bound: a pause that fails or hangs
  never blocks the suspend.

Playback is *not* resumed on wake — press play when you are back.

## Tunables

Class constants on `SleepTimerApp`:

| Constant | Default | Meaning |
|---|---|---|
| `SLEEP_LABEL_DELAY_S` | `0.5` | gap that lets `Sleeping...` paint before suspending |
| `STATUS_POLL_MS` | `100` | how often the Tk thread drains worker updates |
| `JOIN_TIMEOUT_S` | `2.0` | bound on waiting for the countdown thread to exit |
| `MEDIA_PAUSE_TIMEOUT_S` | `3.0` | bound on waiting for the media pause |

## Notes

- Needs Windows 10 1809+ for SMTC. On classic S3 sleep nothing survives the
  suspend anyway; this mainly matters in Modern Standby (S0ix), where the audio
  engine keeps running.
- Requires a desktop session — it is a GUI app, not a service. Closing the window
  stops the countdown thread rather than orphaning it.
- Depends on `winrt-windows-media-control` / `-foundation` (`--extra sleep-timer`);
  the import is optional, so a plain `uv run python ...` still starts the timer.
