---
tags: [macos, launchd, automation, scheduling, reference]
created: 2026-09-04
---

# launchd on macOS

> [!abstract] Why not cron
> **cron skips a job if the Mac is asleep at the scheduled time — permanently.**
> launchd runs it on the next wake. If you close your laptop at 3pm and the job
> was due at 3:45, cron gives you nothing and launchd gives you a late run.
> On macOS, cron is also deprecated: it's a compatibility shim over launchd.

---

## The mental model

launchd is macOS's single init/scheduler system. You describe a job in a **plist**
(an XML property list), drop it in a known directory, and register it. launchd
then owns the lifecycle: schedules it, restarts it, logs its exit codes.

Each job has a **Label** — a unique reverse-DNS identifier like
`com.avrm.optionscollector` — which is how you refer to it in every command.

### Agent vs Daemon

| | Runs as | Location | Runs when |
|---|---|---|---|
| **LaunchAgent** (user) | you | `~/Library/LaunchAgents/` | you're logged in |
| LaunchAgent (all users) | each user | `/Library/LaunchAgents/` | any user logged in |
| LaunchDaemon | root | `/Library/LaunchDaemons/` | at boot, no login needed |

**Use a LaunchAgent** unless you need it running with nobody logged in. Agents
don't need sudo and can't break the boot process.

---

## Where our files live

```
~/Library/LaunchAgents/com.avrm.optionscollector.plist   the job definition
~/bin/options_collector.sh                               the launcher script
~/Library/Logs/options_collector.log                     what the script writes
~/Library/Logs/options_collector.launchd.{out,err}.log   what launchd captures
/Volumes/SD3.2_256/.../algo/data/options/                the data it produces
```

> [!important] Why the launcher is on the internal drive
> The project lives on an exFAT USB volume mounted `noowners,nosuid`. **launchd
> refuses to execute a program whose ownership it cannot verify there** — you get
> exit code 78 (`EX_CONFIG`) and `launchctl kickstart` silently does nothing.
>
> Two fixes, and we used both:
> 1. Invoke the interpreter explicitly: `ProgramArguments` = `["/bin/bash", "<script>"]`
>    rather than relying on the script's own execute bit.
> 2. Keep the launcher on APFS. This also means it still runs — and still **logs**
>    `SKIP: USB volume not mounted` — when the drive is absent. With the script on
>    the volume, an unmounted drive means launchd can't run anything at all, so you
>    get silence instead of a diagnosis.

---

## The plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.avrm.optionscollector</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/avrm/bin/options_collector.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer>
              <key>Hour</key><integer>15</integer>
              <key>Minute</key><integer>45</integer></dict>
        <!-- ... Weekday 2-5 ... -->
    </array>

    <key>RunAtLoad</key><true/>
    <key>StandardOutPath</key><string>/Users/avrm/Library/Logs/....out.log</string>
    <key>StandardErrorPath</key><string>/Users/avrm/Library/Logs/....err.log</string>
    <key>ProcessType</key><string>Background</string>
</dict>
</plist>
```

### Keys worth knowing

| Key | Does |
|---|---|
| `Label` | Unique id. Required. Convention is reverse-DNS |
| `ProgramArguments` | Command + args, as an array. `argv[0]` is the executable |
| `StartCalendarInterval` | Cron-like schedule. One dict, or an array of them |
| `StartInterval` | Simpler alternative: run every N **seconds** |
| `RunAtLoad` | Run once immediately when loaded / at login |
| `KeepAlive` | Restart if it exits. For long-running services, not scheduled jobs |
| `WorkingDirectory` | cd here first (we do it in the script instead) |
| `EnvironmentVariables` | dict of env vars — launchd gives you almost none |
| `ProcessType` | `Background` lowers scheduling priority. Polite for cron-ish jobs |

**`StartCalendarInterval` fields**: `Minute`, `Hour`, `Day` (of month), `Weekday`
(0/7 = Sunday), `Month`. **Omitted means "every"** — the inverse of cron's `*`.
So `{Hour: 15, Minute: 45}` alone fires every day; adding `Weekday` restricts it.
There's no range syntax, so Mon–Fri needs **five separate dicts**.

---

## Commands

```bash
# validate the XML BEFORE loading -- a malformed plist fails obscurely
plutil -lint ~/Library/LaunchAgents/com.avrm.optionscollector.plist

# register / unregister
launchctl load   ~/Library/LaunchAgents/com.avrm.optionscollector.plist
launchctl unload ~/Library/LaunchAgents/com.avrm.optionscollector.plist

# is it registered? -> "PID  lastExitCode  Label"
launchctl list | grep optionscollector

# detailed state, schedule, exit code
launchctl print "gui/$(id -u)/com.avrm.optionscollector"

# force a run right now (great for testing)
launchctl kickstart -k "gui/$(id -u)/com.avrm.optionscollector"
```

`launchctl list` output is three columns: **PID** (`-` when not running),
**last exit code**, **Label**. `0` is what you want.

> [!note] load/unload vs bootstrap/bootout
> `load`/`unload` are the older forms and still work fine. The modern equivalents
> are `launchctl bootstrap gui/$UID <plist>` and
> `launchctl bootout gui/$UID/<label>`. Same effect for a user agent.

---

## Gotchas

> [!warning] The environment is nearly empty
> No `.zshrc`, no `.zprofile`, no aliases. `PATH` is roughly `/usr/bin:/bin:/usr/sbin:/sbin`.
> The working directory is **`/`**, not the script's folder.
>
> Use absolute paths for everything. Call `.venv/bin/python` directly rather than
> "activating" a venv. This is the single most common reason a script that works
> in your terminal fails under launchd.

**Failures are silent** unless you set `StandardErrorPath` or log yourself. Unlike
cron, launchd doesn't even mail you.

**Full Disk Access** may be required to reach `/Volumes` or `~/Documents`.
System Settings → Privacy & Security → Full Disk Access.

**A missed run is coalesced, not queued.** If the Mac sleeps Friday through
Monday, you get **one** run on wake — not one per missed day. For anything
point-in-time (like an options chain) the intervening days are gone for good.

**Editing a plist requires a reload.** `unload` then `load`; launchd does not
re-read the file on its own.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| exit **78** (`EX_CONFIG`) | launchd can't execute the program — ownership/permissions. Classic on exFAT/`noowners` volumes. Invoke `/bin/bash <script>` and keep the script on APFS |
| exit **127** | command not found — a bare command relying on `PATH` |
| exit **126** | found but not executable — missing `chmod +x` |
| `kickstart` does nothing, `runs` stays flat | Job registered but launchd won't spawn it. Same class as 78 |
| Works manually, fails under launchd | Environment. Test with `env -i PATH=/usr/bin:/bin <your script>` |
| Nothing in the log at all | Job never loaded (`launchctl list`), or Full Disk Access missing |

Testing under a stripped environment, which is what launchd actually gives you:

```bash
env -i PATH=/usr/bin:/bin /Users/avrm/bin/options_collector.sh
```

---

## Removing our job

```bash
launchctl unload ~/Library/LaunchAgents/com.avrm.optionscollector.plist
rm ~/Library/LaunchAgents/com.avrm.optionscollector.plist
```

## Related
- [[Strategy Spec v1]]
