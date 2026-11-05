"""Agent views: pages your agent serves, shown to the workspace inside Genfleet.

Declare each view in ``genfleet.toml`` and name the app that serves them::

    [agent]
    name = "sales-assistant"
    views_app = "sales_views:views"     # module:symbol, next to the manifest

    [[views]]
    id = "team"                # lowercase letters, digits and '-'
    title = "Sales team"       # the tab's label; defaults to the id
    path = "/team"             # the page; everything it loads lives under it
    min_role = "admin"         # owner | admin | member (default admin)

Then serve them with :class:`Views`::

    from genfleet.sdk.views import Viewer, Views

    views = Views()
    views.static("/team", "static/team")        # index.html, app.js, app.css

    @views.route("/team/api/roster")
    async def roster(request, viewer: Viewer):
        return {"people": [...], "editor": viewer.user_id}

    @views.route("/team/api/roster", methods=["POST"], min_role="owner")
    async def save(request, viewer: Viewer):
        ...

How a view runs:

- The platform frames the view's page in its app, on an origin of the
  view's own. The page itself (``static``) loads without a viewer: keep data
  out of it.
- The page loads the platform script first, ``<script src="/__gf/view.js">``,
  and calls ``genfleet.fetch(url, init)`` for its data: same origin only, and
  each request carries the viewer's short-lived credential.
- A :meth:`Views.route` handler gets the signed-in :class:`Viewer` (the
  member's user id, workspace role and the view). The platform sets it and
  removes any copy a browser sent. A route refuses a request with no viewer
  (401) or a role below its ``min_role`` (403).
- **No inline script or style.** Every view is served with a Content
  Security Policy of ``default-src 'self'`` and no inline allowance, so a
  single HTML file with ``<script>`` or ``<style>`` blocks won't run. Ship
  your script and style as files next to the page. Nothing loads from
  another origin either (no CDN scripts, fonts or images), and forms post
  only to the view's own origin.
- Changes that belong to the workspace (its content files) are written
  through the platform, not kept in the agent.

Try it locally with :func:`dev_server` (``python -m genfleet.sdk.views``),
which fakes the viewer and sends the same security headers.

Requires the ``serve`` extra: ``pip install 'genfleet-sdk[serve]'``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

#: Headers the platform sets with the signed-in viewer. A browser's copies are removed before your code runs.
VIEWER_ID_HEADER = "x-genfleet-viewer-id"
VIEWER_ROLE_HEADER = "x-genfleet-viewer-role"
VIEW_ID_HEADER = "x-genfleet-view"

#: The platform's script every view page loads first.
CLIENT_SCRIPT_PATH = "/__gf/view.js"

MinRole = Literal["owner", "admin", "member"]
MIN_ROLES: tuple[MinRole, ...] = ("owner", "admin", "member")
_MEMBER_ROLES = frozenset({"owner", "admin", "developer", "billing", "member", "viewer"})

#: The security headers the platform sends with every view response, as the dev server sends them.
SECURITY_HEADERS = {
    "content-security-policy": (
        "default-src 'self'; connect-src 'self'; form-action 'self'; base-uri 'none'; object-src 'none'"
    ),
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cross-origin-opener-policy": "same-origin",
}


def role_meets(role: str, min_role: MinRole) -> bool:
    """Whether a workspace ``role`` reaches ``min_role``: ``owner``; ``admin`` (owners and admins); ``member`` (any member)."""
    if min_role == "owner":
        return role == "owner"
    if min_role == "admin":
        return role in ("owner", "admin")
    return role in _MEMBER_ROLES


@dataclass(frozen=True)
class Viewer:
    """The workspace member looking at a view."""

    user_id: str
    #: Their role in the workspace: ``owner``, ``admin``, ``developer``, ``billing``, ``member`` or ``viewer``.
    role: str
    #: The view they opened (``[[views]] id``).
    view: str

    def at_least(self, min_role: MinRole) -> bool:
        return role_meets(self.role, min_role)


def viewer_from_headers(headers: Mapping[str, str]) -> Viewer | None:
    """The viewer the platform stamped on a request, or ``None`` when there is none."""
    lowered = {k.lower(): v for k, v in headers.items()}
    user_id, role, view = (lowered.get(h, "").strip() for h in (VIEWER_ID_HEADER, VIEWER_ROLE_HEADER, VIEW_ID_HEADER))
    if not user_id or not role or not view:
        return None
    return Viewer(user_id=user_id, role=role, view=view)


Handler = Callable[..., Awaitable[Any]]


def _starlette() -> Any:
    try:
        import starlette.applications as applications
        import starlette.responses as responses
        import starlette.routing as routing
        import starlette.staticfiles as staticfiles
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "genfleet.sdk.views requires the serve extra. Install it with: pip install 'genfleet-sdk[serve]'"
        ) from exc
    return applications, responses, routing, staticfiles


class Views:
    """The ASGI app that serves an agent's views. Name it in ``[agent] views_app``."""

    def __init__(self) -> None:
        self._routes: list[Any] = []
        self._app: Any = None

    def static(self, path: str, directory: str | Path, *, html: bool = True) -> None:
        """Serve the files in ``directory`` under ``path`` (``index.html`` for the page itself).

        Static files load without a viewer: put your page, script and style
        here, and your data behind :meth:`route`. A relative ``directory`` is
        resolved from the module that calls this.
        """
        _, _, routing, staticfiles = _starlette()
        folder = Path(directory)
        if not folder.is_absolute():
            caller = Path(inspect.stack()[1].filename).resolve().parent
            folder = caller / folder
        self._routes.append(routing.Mount(_route_path(path), app=staticfiles.StaticFiles(directory=folder, html=html)))
        self._app = None

    def route(self, path: str, *, methods: list[str] | None = None, min_role: MinRole = "member") -> Callable[[Handler], Handler]:
        """A data route: ``async def handler(request, viewer)``, returning a dict or list (JSON) or a response.

        Refuses a request without a signed-in viewer (401) and a viewer below
        ``min_role`` (403). The view's own ``[[views]] min_role`` is checked
        by the platform before the request arrives; this one can only narrow it.
        """
        if min_role not in MIN_ROLES:
            raise ValueError(f"min_role must be one of {', '.join(MIN_ROLES)}")
        route_path = _route_path(path)
        _, responses, routing, _ = _starlette()

        def decorate(handler: Handler) -> Handler:
            async def endpoint(request: Any) -> Any:
                viewer = viewer_from_headers(request.headers)
                if viewer is None:
                    return responses.JSONResponse({"error": "sign_in_required"}, status_code=401)
                if not viewer.at_least(min_role):
                    return responses.JSONResponse({"error": "role_refused"}, status_code=403)
                result = await handler(request, viewer)
                if isinstance(result, (dict, list)):
                    return responses.JSONResponse(result)
                return result

            self._routes.append(routing.Route(route_path, endpoint, methods=methods or ["GET"]))
            self._app = None
            return handler

        return decorate

    def asgi(self) -> Any:
        if self._app is None:
            applications, _, routing, _ = _starlette()
            # Routes before static mounts, so `/team/api/...` is never shadowed by `/team`.
            mounts = [r for r in self._routes if isinstance(r, routing.Mount)]
            ordered = [r for r in self._routes if not isinstance(r, routing.Mount)] + mounts
            self._app = applications.Starlette(routes=ordered)
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.asgi()(scope, receive, send)


def _route_path(path: str) -> str:
    if not path.startswith("/") or "//" in path or ".." in path.split("/"):
        raise ValueError(f"view path {path!r} must start with '/' and have no empty or '..' segments")
    return path.rstrip("/") or "/"


#: The dev server's stand-in for the platform script: same API, no handshake.
DEV_CLIENT_SCRIPT = (
    "(function(){window.genfleet=Object.freeze({"
    "fetch:function(i,o){return fetch(i,o)},ready:function(){return Promise.resolve()},"
    "navigate:function(p){console.log('genfleet.navigate',p)}})})();"
)


def dev_app(views: Views, viewer: Viewer | None) -> Any:
    """``views`` as the platform would serve it, for local runs.

    Every request gets ``viewer`` (``None``: none, like the page's first
    load), any viewer header the browser sent is replaced, ``/__gf/view.js``
    is a stand-in for the platform script, and every response carries the
    platform's security headers, so an inline script fails here as it would
    on the platform.
    """
    stamped = viewer

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await views(scope, receive, send)
            return
        if scope["path"] == CLIENT_SCRIPT_PATH:
            await _send(send, 200, DEV_CLIENT_SCRIPT.encode(), "application/javascript")
            return
        drop = {VIEWER_ID_HEADER.encode(), VIEWER_ROLE_HEADER.encode(), VIEW_ID_HEADER.encode()}
        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() not in drop]
        if stamped is not None:
            headers += [
                (VIEWER_ID_HEADER.encode(), stamped.user_id.encode()),
                (VIEWER_ROLE_HEADER.encode(), stamped.role.encode()),
                (VIEW_ID_HEADER.encode(), stamped.view.encode()),
            ]

        async def secured(message: Any) -> None:
            if message["type"] == "http.response.start":
                kept = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"content-security-policy"]
                message = {**message, "headers": kept + [(k.encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]}
            await send(message)

        await views({**scope, "headers": headers}, receive, secured)

    return app


async def _send(send: Any, status: int, body: bytes, content_type: str) -> None:
    headers = [(b"content-type", content_type.encode())] + [(k.encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


def dev_server(views: Views, *, viewer: Viewer | None = None, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve ``views`` locally with a fake viewer (default: an admin on the first view). Blocks."""
    import uvicorn

    uvicorn.run(dev_app(views, viewer or Viewer(user_id="dev-user", role="admin", view="dev")), host=host, port=port)


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import importlib
    import sys

    parser = argparse.ArgumentParser(prog="python -m genfleet.sdk.views", description="Serve an agent's views locally.")
    parser.add_argument("app", help="module:symbol of the Views app, e.g. sales_views:views")
    parser.add_argument("--role", default="admin", help="the fake viewer's workspace role")
    parser.add_argument("--view", default="dev", help="the fake viewer's view id")
    parser.add_argument("--user", default="dev-user", help="the fake viewer's user id")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    module_name, _, symbol = args.app.partition(":")
    sys.path.insert(0, ".")
    views = getattr(importlib.import_module(module_name), symbol)
    print(json.dumps({"serving": args.app, "url": f"http://127.0.0.1:{args.port}/", "role": args.role}))
    dev_server(views, viewer=Viewer(user_id=args.user, role=args.role, view=args.view), port=args.port)


if __name__ == "__main__":  # pragma: no cover
    _main()
