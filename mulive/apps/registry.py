import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mulive.apps.app_config import UI_API_VERSION


APP_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class AppDefinition:
    id: str
    title: str
    description: str
    capabilities: tuple[str, ...]
    bundle_dir: Path
    assets_dir: Path | None
    ui_module: str | None
    ui_stylesheet: str | None
    transport_config: dict[str, Any]
    config: dict[str, Any]
    session_runner_factory: Callable[[dict[str, Any]], Callable]
    enabled: bool = True

    def create_session_runner(self):
        """Create the per-connection session entry point."""
        return self.session_runner_factory(deepcopy(self.config))

    def public_metadata(self, *, include_ui: bool = False) -> dict[str, Any]:
        metadata = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "available": True,
        }
        if include_ui:
            metadata.update(
                {
                    "app_name": self.title,
                    "ui_api_version": UI_API_VERSION,
                    "ui_module": self.ui_module,
                    "ui_stylesheet": self.ui_stylesheet,
                }
            )
        return metadata


@dataclass(frozen=True)
class UnavailableApp:
    id: str
    title: str
    bundle_path: str
    reason: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": "This app is currently unavailable.",
            "capabilities": [],
            "available": False,
        }


class AppRegistry:
    def __init__(self):
        self._apps: dict[str, AppDefinition] = {}
        self._entries: list[AppDefinition | UnavailableApp] = []

    def register(self, app: AppDefinition) -> None:
        if not APP_ID_RE.fullmatch(app.id):
            raise ValueError(f"Invalid app id: {app.id!r}")
        if app.id in self._apps:
            raise ValueError(f"Duplicate app id: {app.id}")
        self._apps[app.id] = app
        self._entries.append(app)

    def record_unavailable(
        self,
        *,
        app_id: str,
        title: str,
        bundle_path: str,
        reason: str,
    ) -> None:
        self._entries.append(
            UnavailableApp(
                id=app_id,
                title=title,
                bundle_path=bundle_path,
                reason=reason,
            )
        )

    def get(self, app_id: str) -> AppDefinition | None:
        app = self._apps.get(app_id)
        return app if app is not None and app.enabled else None

    def require(self, app_id: str) -> AppDefinition:
        app = self.get(app_id)
        if app is None:
            raise KeyError(app_id)
        return app

    def enabled_apps(self) -> tuple[AppDefinition, ...]:
        return tuple(app for app in self._apps.values() if app.enabled)

    def unavailable_apps(self) -> tuple[UnavailableApp, ...]:
        return tuple(
            entry for entry in self._entries if isinstance(entry, UnavailableApp)
        )

    def public_apps(self) -> list[dict[str, Any]]:
        return [
            entry.public_metadata()
            for entry in self._entries
            if isinstance(entry, UnavailableApp) or entry.enabled
        ]
