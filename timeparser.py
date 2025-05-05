import pandas as pd
import plotly.express as px
import re
from datetime import datetime

# Paste your log data as a single string
log_data = open("times.txt", "r").read()
log_data = re.sub(r"^(?!\!).*\n", "", log_data)

lines = log_data.strip().splitlines()

# Step 1: Parse the log lines
events = []
stack = {}

for line in lines:
    match = re.match(r"!(START|END)\s+(.*)\s+(\d+\.\d+)", line)
    if not match:
        continue
    action, label, timestamp = match.groups()
    timestamp = float(timestamp)
    dt = datetime.fromtimestamp(timestamp)

    label = label.strip()
    key = label  # You could refine this with more logic if needed

    if action == "START":
        stack.setdefault(key, []).append(dt)
    elif action == "END" and key in stack and stack[key]:
        start_dt = stack[key].pop(0)
        duration_ms = (dt - start_dt).total_seconds() * 1000
        events.append(
            {
                "Task": key,
                "Start": start_dt,
                "Finish": dt,
                "Duration (ms)": round(duration_ms, 3),
            },
        )

# Step 2: Create DataFrame and plot
df = pd.DataFrame(events)
fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", title="Event Timeline", hover_data=["Duration (ms)"])
fig.update_yaxes(autorange="reversed")
fig.show()
