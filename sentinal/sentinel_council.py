
"""
Adapter exposing get_council() from aegis_timescales.
Uses classes defined in the shipped aegis_timescales.py.
"""
from .aegis_timescales import (
	AegisCouncil, NexusMemory,
	EchoSeedAgent, ShortTermAgent, MidTermAgent, LongTermArchivistAgent,
	TimeScaleCoordinator, MetaJudgeAgent, BiofeedbackAgent, EnvSignalAgent, ContextConflictAgent
)

def get_council() -> AegisCouncil:
	council = AegisCouncil(per_agent_timeout_sec=2.5)
	# Core seed + timescales
	council.register_agent(EchoSeedAgent("EchoSeed", council.memory))
	council.register_agent(ShortTermAgent("ShortTerm", council.memory))
	council.register_agent(MidTermAgent("MidTerm", council.memory))
	council.register_agent(LongTermArchivistAgent("LongArchivist", council.memory))
	# Fusion agents before coordination
	council.register_agent(BiofeedbackAgent("BiofeedbackAgent", council.memory))
	council.register_agent(EnvSignalAgent("EnvSignalAgent", council.memory))
	council.register_agent(ContextConflictAgent("ContextConflictAgent", council.memory))
	# Timescale coordination + arbitration
	council.register_agent(TimeScaleCoordinator("TimeScaleCoordinator", council.memory))
	council.register_agent(MetaJudgeAgent("MetaJudge", council.memory))
	return council
