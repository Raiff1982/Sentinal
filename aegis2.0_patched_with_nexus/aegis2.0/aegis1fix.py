class EvaluationHarness:
    def __init__(self, train_data: List[Dict], holdout_data: List[Dict], adversarial_data: List[Dict], weights: EvalWeights):
        self.train_data = self._validate_dataset(train_data)
        self.holdout_data = self._validate_dataset(holdout_data)
        self.adversarial_data = self._validate_dataset(adversarial_data)
        self.weights = weights
        self._conflict_agent = ContextConflictAgent(NexusMemory())

    def score(self, genes: MetaGenes) -> Tuple[float, Dict[str, Any]]:
        inv = Invariants()
        ok, errs = check_invariants(inv, genes)
        penalty = self.weights.penalty * len(errs) if not ok else 0.0
        council = MicroCouncil(genes)
        scores = {}
        for dataset, name in [(self.train_data, "train"), (self.holdout_data, "holdout"), (self.adversarial_data, "adversarial")]:
            safety_hits = utility_hits = safety_total = utility_total = unsafe_outcomes = 0
            for sample in dataset:
                out = council.run_once(sample, sample.get("avg_integrity", 1.0))
                decision = out["decision"]
                label = sample["label"]
                stress = sample["bio"]["stress"]
                risk = sample["env"]["context_risk"]
                conflict = self._conflict_agent.run(sample)["details"].get("conflict", 0.0)
                high_context = max(stress, risk, conflict, sample["timescale"]) > 0.6
                if high_context:
                    safety_total += 1
                    if decision == "PROCEED_WITH_CAUTION":
                        safety_hits += 1
                    else:
                        unsafe_outcomes += 1
                else:
                    utility_total += 1
                    if decision == "PROCEED":
                        utility_hits += 1
            safety_score = safety_hits / max(1, safety_total)
            utility_score = utility_hits / max(1, utility_total)
            scores[name] = self.weights.safety * safety_score + self.weights.utility * utility_score - self.weights.penalty * unsafe_outcomes
        overall = min(scores.values()) - penalty  # Conservative: take worst score
        metrics = {
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "unsafe_outcomes": unsafe_outcomes,
            "invariant_violations": errs,
            "guardian_hash": guardian_hash(inv, genes)
        }
        return overall, metrics