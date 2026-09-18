from typing import Callable, Dict, Set, Any, Awaitable
from pathlib import Path
import ssl
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import MediaStreamError

from ..core.utils import rx_Subject as Subject # for input audio/video subjects
from ..apps.routes import AppRoutes
from ..core.auth import AppAuthentication
from .setup_tracks import pc_session_setup
from .._resources import packaged_path

DEFAULT_CLIENT_HTML_PATH = packaged_path("client", "client.html")
DEFAULT_DASHBOARD_HTML_PATH = packaged_path("client", "dashboard.html")
DEFAULT_LOGIN_HTML_PATH = packaged_path("client", "login.html")

class Server:
    def __init__(
        self,
        run_session: Callable = None,
        client_html_path: Path = DEFAULT_CLIENT_HTML_PATH,
        config: Dict = {},
        app_assets_dir: Path = None,
        app_registry=None,
        user_directory=None,
        dashboard_html_path: Path = DEFAULT_DASHBOARD_HTML_PATH,
        authentication: AppAuthentication | None = None,
    ):
        """Initialize the WebRTC server with a session runner.
        
        Args:
            run_session: Async function that owns one SessionContext lifecycle.
        """

        if (run_session is None) == (app_registry is None):
            raise ValueError("Provide either run_session or app_registry")
        self.run_session = run_session
        self.app_registry = app_registry
        self.user_directory = user_directory
        self.authentication = authentication
        self.pcs: Set[RTCPeerConnection] = set()
        middlewares = [authentication.middleware] if authentication is not None else []
        self.app = web.Application(middlewares=middlewares)
        self.config = config
        self.app_assets_dir = self._resolve_app_assets_dir(app_assets_dir)
        self.dashboard_html_path = Path(dashboard_html_path)
        self._setup_routes(Path(client_html_path))
        self.app.on_shutdown.append(self.on_shutdown)

    @staticmethod
    def _resolve_app_assets_dir(app_assets_dir):
        if app_assets_dir is None:
            return None
        path = Path(app_assets_dir).resolve()
        if not path.is_dir():
            raise ValueError(f"App assets directory does not exist: {path}")
        return path

    
    def _setup_routes(self, client_html_path):
        """Set up the web application routes."""
        print('setting up routes..')
        if self.authentication is not None:
            self.authentication.register_routes(self.app)
        if self.app_registry is None:
            self.app.router.add_post("/offer", self.offer_handler)

            async def client_config(request):
                return web.json_response(self.config.get("client_config", {}))

            self.app.router.add_get("/client-config", client_config)

            if self.app_assets_dir is not None:
                self.app.router.add_static(
                    "/app-assets/",
                    path=self.app_assets_dir,
                    name="app_assets",
                    follow_symlinks=False,
                )
        else:
            AppRoutes(
                registry=self.app_registry,
                user_directory=self.user_directory,
                client_html_path=client_html_path,
                dashboard_html_path=self.dashboard_html_path,
                accept_offer=self._accept_offer,
                base_transport_config=self.config,
            ).register(self.app)
        
        client_dir = client_html_path.parent  # base directory containing index.html, js/, assets/, etc.

        async def serve_client_file(request):
            requested_path = request.match_info.get('path', '')
            if requested_path == '':
                return web.FileResponse(client_html_path)

            # Resolve full path and prevent directory traversal
            full_path = (client_dir / requested_path).resolve()
            if not full_path.is_relative_to(client_dir) or not full_path.exists():
                raise web.HTTPNotFound()
            return web.FileResponse(full_path)

        # catch-all route
        self.app.router.add_get("/{path:.*}", serve_client_file)

    async def offer_handler(self, request):
        """Handle WebRTC offer and set up media processing pipeline."""
        return await self._accept_offer(request, self.run_session, self.config)

    async def _accept_offer(self, request, run_session, config):
        params = await request.json()
        # Reject new offers when the server is at its concurrency cap.
        # Kokoro + faster-whisper are CPU-bound and a small AWS box cannot
        # multiplex many voice sessions in real time. Default is None
        # (no cap — previous behaviour); deployments opt in via config,
        # e.g. quickstart's --max-concurrent-sessions. The check runs after
        # the await above so check -> create -> add has no interleaving
        # point on the event loop (no double-admit race).
        max_sessions = config.get("max_concurrent_sessions")
        if max_sessions is not None and len(self.pcs) >= max_sessions:
            print(f"Rejecting offer: at capacity ({len(self.pcs)}/{max_sessions})")
            raise web.HTTPServiceUnavailable(
                text=f"Server busy: {len(self.pcs)} of {max_sessions} sessions active."
            )
        pc = pc_session_setup(
            run_session,
            config,
            on_peer_close=self.pcs.discard,
        )
        self.pcs.add(pc)
 
        
        try:
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return web.json_response({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            })
        except Exception as e:
            print(f"Error in offer handler: {e}")
            self.pcs.discard(pc)
            await pc.close()
            raise web.HTTPInternalServerError(text=str(e))

    async def on_shutdown(self, app=None):
        """Handle application shutdown."""
        print("Shutting down...")
        # Close all peer connections
        for pc in list(self.pcs):
            await pc.close()
        self.pcs.clear()
    
    def run(self, host="0.0.0.0", port=9000):
        ssl_context = None
        ssl_keyfile = self.config.get("ssl_keyfile")
        ssl_certfile = self.config.get("ssl_certfile")
        if ssl_keyfile and ssl_certfile:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
            print(f"Serving HTTPS on {host}:{port}")

        web.run_app(self.app, host=host, port=port, ssl_context=ssl_context)
