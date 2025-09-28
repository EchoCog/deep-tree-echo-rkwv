"""
EchoSpace: Agent-Arena-Relation Architecture
Implementation of Marduk's recursive blueprint for fractal agency
"""

from .core import Agent, Arena, AgentArenaRelation
from .namespaces import NamespaceManager
from .memory import EchoMemorySystem
from .marduk import ActualMarduk, VirtualMarduk
from .sandbox import MardukSandbox
from .consensus import ConsensusManager
from .workflows import WorkflowOrchestrator

__version__ = "1.0.0"
__all__ = [
    "Agent",
    "Arena", 
    "AgentArenaRelation",
    "NamespaceManager",
    "EchoMemorySystem",
    "ActualMarduk",
    "VirtualMarduk", 
    "MardukSandbox",
    "ConsensusManager",
    "WorkflowOrchestrator"
]