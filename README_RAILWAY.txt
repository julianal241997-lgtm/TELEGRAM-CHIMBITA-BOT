README — Railway (CHIMBITA-RAILWAY-FIXED)

1) Railway → New Project → Upload → select this zip.
2) Wait for build (Python auto-detected by requirements.txt).
3) Add Scheduler:
   - Schedules → New Schedule
   - Command: python scan_bot.py
   - Cron: 0 * * * *
   - Enabled: ON
4) Check Deployments → Logs and Telegram. Startup message and scan summary will appear.
