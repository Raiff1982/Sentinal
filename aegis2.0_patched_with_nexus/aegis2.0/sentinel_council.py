"""Adapter exposing get_council() from aegis_timescales and attaching NexusSignalEngine."""
from aegis_timescales import (
    AegisCouncil, NexusMemory,
    EchoSeedAgent, ShortTermAgent, MidTermAgent, LongTermArchivistAgent,
    TimeScaleCoordinator, MetaJudgeAgent, BiofeedbackAgent, EnvSignalAgent, ContextConflictAgent
)
from aegis_nexus import NexusSignalEngine

def get_council() -> AegisCouncil:
    council = AegisCouncil(per_agent_timeout_sec=2.5)
    # Attach Nexus Signal Engine to shared memory so agents can use it
    if not hasattr(council.memory, 'nexus'):
        council.memory.nexus = NexusSignalEngine(root='./nexus')
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
