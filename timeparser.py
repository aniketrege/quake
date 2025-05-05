from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import re
from datetime import datetime

# Paste your log data as a single string
log_data = open("times.txt", "r").read()
log_data = re.sub(r"quake\d-\d  \| ", "", log_data)
log_data = re.sub(r"^(?!\!).*\n", "", log_data)

lines = log_data.strip().splitlines()


@dataclass
class Event:
    start: datetime | None = None
    end: datetime | None = None


events_objs = defaultdict(lambda: Event())

for line in lines:
    # if "job" in line:
    #     continue
    match = re.match(r"!(START|END)\s+(.*)\s+(\d+\.\d+)", line)
    if not match:
        continue
    action, label, timestamp = match.groups()
    timestamp = float(timestamp)
    dt = datetime.fromtimestamp(timestamp)
    if action == "START":
        events_objs[label].start = dt
    elif action == "END":
        events_objs[label].end = dt
events = []
for label, event in events_objs.items():
    if not event.start or not event.end:
        continue
    else:
        duration_ms = (event.end - event.start).total_seconds() * 1000
        events.append((label, event.start, event.end, duration_ms))
events.sort(key=lambda e: e[2], reverse=True)

# Convert to DataFrame
df = pd.DataFrame(events, columns=["Task", "Start", "Finish", "Duration_ms"])

# Create timeline with hover data
fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", hover_data={"Duration_ms": ":.0f"})
fig.update_layout(title="Task Timeline")

# Show the figure
fig.show()

for event in events:
    print(event)
