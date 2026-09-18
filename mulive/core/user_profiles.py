from dataclasses import dataclass
from pathlib import Path
import re

import yaml


USER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    display_name: str
    grade: int | None = None

    def to_client_dict(self) -> dict:
        data = {
            "id": self.user_id,
            "display_name": self.display_name,
        }
        if self.grade is not None:
            data["grade"] = self.grade
        return data


class UserDirectory:
    def __init__(
        self,
        profiles: tuple[UserProfile, ...],
        default_user: str,
    ):
        if not profiles:
            raise ValueError("At least one user profile is required")
        self.profiles = profiles
        self._by_id = {profile.user_id: profile for profile in profiles}
        if len(self._by_id) != len(profiles):
            raise ValueError("Duplicate user profile ids")
        if default_user not in self._by_id:
            raise ValueError(f"Unknown default user: {default_user!r}")
        self.default_user = default_user

    def resolve(self, user_id: str | None = None) -> UserProfile:
        if not user_id:
            return self._by_id[self.default_user]
        profile = self._by_id.get(user_id)
        if profile is None:
            raise KeyError(user_id)
        return profile

    def to_client_dict(self) -> dict:
        return {
            "users": [profile.to_client_dict() for profile in self.profiles],
            "default_user": self.default_user,
        }


def load_user_directory(path: Path) -> UserDirectory:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError(f"{path}: user profile schema must be 1")

    users = data.get("users")
    if not isinstance(users, dict):
        raise ValueError(f"{path}: users must be a mapping")

    profiles = []
    for user_id, profile_data in users.items():
        if not isinstance(user_id, str) or not USER_ID_RE.fullmatch(user_id):
            raise ValueError(f"{path}: invalid user id {user_id!r}")
        if not isinstance(profile_data, dict):
            raise ValueError(f"{path}: profile {user_id!r} must be a mapping")
        display_name = " ".join(
            str(profile_data.get("display_name") or user_id).split()
        )
        grade = profile_data.get("grade")
        if grade is not None and not isinstance(grade, int):
            raise ValueError(f"{path}: grade for {user_id!r} must be an integer")
        profiles.append(UserProfile(user_id, display_name, grade))

    return UserDirectory(
        tuple(profiles),
        str(data.get("default_user") or ""),
    )
