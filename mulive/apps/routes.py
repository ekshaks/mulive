from pathlib import Path

from aiohttp import web


class AppRoutes:
    """Attach dashboard and app-specific HTTP routes to an aiohttp app."""

    def __init__(
        self,
        *,
        registry,
        user_directory,
        client_html_path: Path,
        dashboard_html_path: Path,
        accept_offer,
        base_transport_config: dict,
    ):
        self.registry = registry
        if user_directory is None:
            raise ValueError("App dashboard requires a user directory")
        self.user_directory = user_directory
        self.client_html_path = Path(client_html_path)
        self.dashboard_html_path = Path(dashboard_html_path)
        self.accept_offer = accept_offer
        self.base_transport_config = dict(base_transport_config)

    def register(self, app: web.Application) -> None:
        app.router.add_get("/", self.dashboard)
        app.router.add_get("/api/apps", self.list_apps)
        app.router.add_get("/api/users", self.list_users)
        app.router.add_get("/api/apps/{app_id}", self.app_config)
        app.router.add_get("/apps/{app_id}", self.app_page)
        app.router.add_post("/apps/{app_id}/offer", self.app_offer)

        for definition in self.registry.enabled_apps():
            if definition.assets_dir is None:
                continue
            app.router.add_static(
                f"/app-assets/{definition.id}/",
                path=definition.assets_dir,
                name=f"app_assets_{definition.id}",
                follow_symlinks=False,
            )

    async def dashboard(self, request):
        return web.FileResponse(self.dashboard_html_path)

    async def list_apps(self, request):
        return web.json_response(self.registry.public_apps())

    async def list_users(self, request):
        return web.json_response(self.user_directory.to_client_dict())

    async def app_config(self, request):
        definition = self._requested_app(request)
        return web.json_response(definition.public_metadata(include_ui=True))

    async def app_page(self, request):
        self._requested_app(request)
        return web.FileResponse(self.client_html_path)

    async def app_offer(self, request):
        definition = self._requested_app(request)
        try:
            profile = self.user_directory.resolve(request.query.get("user_id"))
        except KeyError:
            raise web.HTTPBadRequest(text="Unknown user profile")

        session_runner = definition.create_session_runner()

        async def run_user_session(session):
            session.user_id = profile.user_id
            await session_runner(session)

        config = {**self.base_transport_config, **definition.transport_config}
        return await self.accept_offer(request, run_user_session, config)

    def _requested_app(self, request):
        app_id = request.match_info.get("app_id", "")
        definition = self.registry.get(app_id)
        if definition is None:
            raise web.HTTPNotFound(text=f"Unknown or disabled app: {app_id}")
        return definition
