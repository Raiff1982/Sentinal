import json
from sentinel_council import get_council
c = get_council()
# Calm
out1 = c.dispatch({"text":"Be fair and honest.", "intent":"", "_signals":{"bio":{"stress":0.1},"env":{"context_risk":0.1}}, "timescale":0.1})
dec1 = [r for r in out1["reports"] if r["agent"]=="MetaJudge"][0]["details"]["decision"]
# Hot (should BLOCK now in MetaJudge)
out2 = c.dispatch({"text":"ship now","intent":"proceed fast","_signals":{"bio":{"stress":0.9},"env":{"context_risk":0.95}}, "timescale":0.95})
dec2 = [r for r in out2["reports"] if r["agent"]=="MetaJudge"][0]["details"]["decision"]
print(json.dumps({"calm": dec1, "hot": dec2}, indent=2))
