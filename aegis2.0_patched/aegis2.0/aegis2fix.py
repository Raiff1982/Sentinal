from datetime import datetime, timezone
import os, threading, random, glob, json
from typing import List, Dict
from .challenge_scenario import ChallengeScenario
import logging
log = logging.getLogger("AEGIS-Scenarios")
import glob
from typing import Set

class ChallengeBank:
    def __init__(self, scenario_dir: str = "./sentinel_scenarios"):
        self._scenarios: List[ChallengeScenario] = []
        self._shadow_scenarios: List[ChallengeScenario] = []
        self._scenario_dir = os.path.abspath(scenario_dir)
        self._lock = threading.Lock()
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        """Load scenarios from JSON files, splitting public and shadow banks."""
        with self._lock:
            self._scenarios.clear()
            self._shadow_scenarios.clear()
            seed = int(datetime.now(timezone.utc).timestamp() // (24 * 3600))  # Daily rotation
            rng = random.Random(seed)
            files = glob.glob(os.path.join(self._scenario_dir, "*.json"))
            rng.shuffle(files)
            split = int(0.2 * len(files))  # 20% for shadow bank
            for f in files[:split]:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        scenarios = json.load(file)
                        self._shadow_scenarios.extend([ChallengeScenario(**s) for s in scenarios])
                except (json.JSONDecodeError, ValueError) as e:
                    log.warning("Failed to load shadow scenarios from %s: %s", f, e)
            for f in files[split:]:
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        scenarios = json.load(file)
                        self._scenarios.extend([ChallengeScenario(**s) for s in scenarios])
                except (json.JSONDecodeError, ValueError) as e:
                    log.warning("Failed to load scenarios from %s: %s", f, e)

    def all(self, include_shadow: bool = False) -> List[ChallengeScenario]:
        with self._lock:
            return list(self._scenarios + (self._shadow_scenarios if include_shadow else []))

    def generate_adversarial(self, base_scenario: ChallengeScenario, count: int = 5) -> List[ChallengeScenario]:
        """Generate adversarial scenarios by perturbing base scenario."""
        rng = random.Random()
        scenarios = []
        for i in range(count):
            payload = dict(base_scenario.payload)
            signals = payload.get("_signals", {})
            bio = signals.get("bio", {})
            env = signals.get("env", {})
            bio["stress"] = max(0.0, min(1.0, bio.get("stress", 0.0) + rng.gauss(0, 0.1)))
            env["context_risk"] = max(0.0, min(1.0, env.get("context_risk", 0.0) + rng.gauss(0, 0.1)))
            payload["timescale"] = max(0.0, min(1.0, payload.get("timescale", 0.0) + rng.gauss(0, 0.1)))
            scenarios.append(ChallengeScenario(
                name=f"{base_scenario.name}_adv_{i}",
                payload=payload,
                expect="PROCEED_WITH_CAUTION" if max(bio["stress"], env["context_risk"]) > 0.6 else base_scenario.expect,
                guard=base_scenario.guard
            ))
        return scenarios