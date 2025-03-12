from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# TODO: Create DevelopmentalDomain enum (LANGUAGE, EMOTIONAL, COGNITIVE, SOCIAL, etc.)
# TODO: Define DevelopmentalStage enum with hierarchical stages:
#   - PRENATAL (initialization)
#   - INFANT (basic patterns)
#   - TODDLER (simple associations)
#   - CHILD (language fundamentals)
#   - ADOLESCENT (complex reasoning)
#   - ADULT (mature cognition)

# TODO: Implement MilestoneStatus enum (NOT_STARTED, IN_PROGRESS, ACHIEVED)
# TODO: Create DevelopmentalMilestone model:
#   - milestone_id: str
#   - name: str
#   - description: str
#   - domain: DevelopmentalDomain
#   - prerequisites: List[str]
#   - min_stage: DevelopmentalStage
#   - metrics: Dict[str, float] (thresholds for achievement)

# TODO: Define MilestoneProgress model to track individual milestone:
#   - milestone_id: str
#   - status: MilestoneStatus
#   - current_metrics: Dict[str, float]
#   - started_at: Optional[datetime]
#   - achieved_at: Optional[datetime]
#   - progress_percentage: float

# TODO: Implement DomainProgress model for domain-specific tracking:
#   - domain: DevelopmentalDomain
#   - current_stage: DevelopmentalStage
#   - milestone_progress: Dict[str, MilestoneProgress]
#   - aggregate_score: float

# TODO: Create DevelopmentalProgression model for overall tracking:
#   - domain_progress: Dict[DevelopmentalDomain, DomainProgress]
#   - overall_stage: DevelopmentalStage
#   - developmental_age: timedelta
#   - started_at: datetime
#   - developmental_velocity: Dict[DevelopmentalDomain, float]

# TODO: Define Stage transition requirements
# TODO: Add validation for milestone prerequisites
# TODO: Implement helper methods for progress calculation