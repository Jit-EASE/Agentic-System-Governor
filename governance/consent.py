
from dataclasses import dataclass
from typing import Tuple

from config import CHILD_AGE_THRESHOLD
from .audit import log_governance_event


@dataclass
class UserProfile:
    user_id: str
    age: int
    country: str = "IE"


def evaluate_consent(user: UserProfile, consent_granted: bool) -> Tuple[bool, bool]:
    """
    Returns:
        allowed_to_run_model (bool)
        is_child (bool)
    """
    is_child = user.age < CHILD_AGE_THRESHOLD

    if not consent_granted:
        log_governance_event(
            "CONSENT_DENIED",
            {
                "user_id": user.user_id,
                "age": user.age,
                "is_child": is_child,
            },
        )
        return False, is_child

    if is_child:
        log_governance_event(
            "CHILD_PROFILE_FLAGGED",
            {
                "user_id": user.user_id,
                "age": user.age,
            },
        )

    return True, is_child
