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

- The platform frames the view's page in its app, sandboxed: the page has
  no cookies and no storage. The page itself (``static``) loads without a
  viewer: keep data out of it.
- The page loads the platform script first, ``<script src="/__gf/view.js">``,
  and calls ``genfleet.fetch(url, init)`` for its data: the view's own URLs
  only, and each request carries the viewer's short-lived credential.
- A :meth:`Views.route` handler gets the signed-in :class:`Viewer` (the
  member's user id, workspace role and the view). The platform sets it and
  removes any copy a browser sent. A route refuses a request with no viewer
  (401) or a role below its ``min_role`` (403).
- **No inline script or style.** Every view is served with a Content
  Security Policy with no inline allowance, so a single HTML file with
  ``<script>`` or ``<style>`` blocks won't run. Ship your script and style
  as files next to the page. Nothing loads from another site either (no CDN
  scripts, fonts or images), and forms post only to the view itself.
- Changes that belong to the workspace (its content files) are written
  through the platform, not kept in the agent.

Try it locally with :func:`dev_server` (``python -m genfleet.sdk.views``),
which fakes the viewer and sends the same security headers.

Requires the ``serve`` extra: ``pip install 'genfleet-sdk[serve]'``.
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
from collections.abc import Awaitable, Callable, Collection, Mapping
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

#: The iframe sandbox a view runs in (no ``allow-same-origin``: an opaque origin,
#: so no cookies and no ``localStorage``/``sessionStorage``), also sent as CSP.
SANDBOX_FLAGS = "allow-scripts allow-forms allow-downloads"


def security_headers(origin: str) -> dict[str, str]:
    """The security headers the platform sends with every view response, for a view served from ``origin``.

    The CSP names the origin (``'self'`` means nothing in an opaque origin) and
    has no inline allowance; ``frame-ancestors`` (the app) is left out locally.
    """
    return {
        "content-security-policy": (
            f"default-src {origin}; connect-src {origin}; form-action {origin}; base-uri 'none'; "
            f"object-src 'none'; sandbox {SANDBOX_FLAGS}"
        ),
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "cross-origin-opener-policy": "same-origin",
    }


#: The only headers of yours the platform passes to the browser. ``set-cookie``,
#: CORS, framing and your own CSP are dropped; a ``location`` only when it stays on the view.
PASSED_RESPONSE_HEADERS = ("content-type", "content-language", "etag", "last-modified", "location", "content-disposition")


def cache_control_for(content_type: str | None) -> str:
    """``private, no-store`` for HTML and JSON (and untyped), ``private, no-cache`` otherwise, as the platform sets it."""
    kind = (content_type or "").lower()
    return "private, no-store" if kind == "" or kind.startswith("text/html") or "json" in kind else "private, no-cache"


_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
#: Types a request without a viewer may get back (the platform's backstop): the page and its files.
_ANONYMOUS_TYPES = ("text/html", "text/css", "text/javascript", "application/javascript", "image/", "font/", "application/font", "application/wasm")


def _location_ok(value: str) -> bool:
    """A redirect that stays on the view: relative or a path; no scheme, ``//``, backslash or whitespace."""
    if not value or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in value) or "\\" in value:
        return False
    return not _SCHEME.match(value) and not value.startswith("//")


def _anonymous_type_ok(content_type: str | None) -> bool:
    kind = (content_type or "").split(";")[0].strip().lower()
    return bool(kind) and any(kind == t or (t.endswith("/") and kind.startswith(t)) for t in _ANONYMOUS_TYPES)


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

    def __init__(self, root: str | Path | None = None) -> None:
        """``root``: where relative ``static`` directories are found, usually
        ``Path(__file__).parent`` of the module that builds the app (the
        sandbox's working directory is not the agent's folder)."""
        self._root = Path(root).resolve() if root is not None else None
        self._routes: list[Any] = []
        self._mounts: list[tuple[str, Any]] = []
        self._app: Any = None

    def static(self, path: str, directory: str | Path, *, html: bool = True) -> None:
        """Serve the files in ``directory`` under ``path`` (``index.html`` for the page itself).

        Static files load without a viewer: put your page, script and style
        here, and your data behind :meth:`route`. A relative ``directory`` is
        resolved against the ``root`` given to :class:`Views` (a relative one
        without a ``root`` is an error).

        The page lives at ``path + "/"``: the bare ``path`` answers with a
        relative redirect there, so the page's relative URLs (``app.js``,
        ``genfleet.fetch("api/...")``) resolve inside the view.
        """
        _, _, _, staticfiles = _starlette()
        folder = Path(directory)
        if not folder.is_absolute():
            if self._root is None:
                raise ValueError("a relative static directory needs Views(root=Path(__file__).parent)")
            folder = self._root / folder
        self._mounts.append((_route_path(path), staticfiles.StaticFiles(directory=folder, html=html)))
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
            applications, responses, routing, _ = _starlette()

            def to_slash(path: str) -> Any:
                # Relative, so it works under whatever prefix the platform
                # mounts the app at; Starlette's own slash redirect is absolute.
                target = (path.rsplit("/", 1)[-1] or ".") + "/"

                async def redirect(request: Any) -> Any:
                    return responses.Response(status_code=307, headers={"location": target}, media_type="text/plain")

                return redirect

            # Data routes first, then each mount's bare-path redirect, then the
            # mounts, longest path first, so `/team/api/...` and `/team/admin`
            # are never shadowed by `/team`, whatever order they were added in.
            mounts = sorted(self._mounts, key=lambda m: len(m[0]), reverse=True)
            ordered = [
                *self._routes,
                *(routing.Route(p, to_slash(p), methods=["GET", "HEAD"]) for p, _ in mounts if p != "/"),
                *(routing.Mount(p if p != "/" else "", app=a) for p, a in mounts),
            ]
            app = applications.Starlette(routes=ordered)
            # No absolute redirects to a URL the platform would drop.
            app.router.redirect_slashes = False
            self._app = app
        return self._app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.asgi()(scope, receive, send)


def _route_path(path: str) -> str:
    if not path.startswith("/") or "//" in path or ".." in path.split("/"):
        raise ValueError(f"view path {path!r} must start with '/' and have no empty or '..' segments")
    return path.rstrip("/") or "/"


#: The dev server's stand-in for the platform script: same API, no handshake.
#: Static and the same for every request: it carries **no token**. The dev
#: token reaches only the view's own HTML pages (a nonce'd inline script that
#: sets ``__gfDev``), which no other site can read, and ``genfleet.fetch``
#: sends it only to the dev server's own origin.
DEV_CLIENT_SCRIPT = (
    "(function(){window.genfleet=Object.freeze({"
    "fetch:function(i,o){o=o||{};var d=window.__gfDev||{},u=new URL(typeof i==='string'?i:i.url,location.href),"
    "h=new Headers(o.headers||{});if(d.token&&u.origin===d.origin)h.set('Authorization','Bearer '+d.token);"
    "return fetch(u.href,Object.assign({},o,{headers:h,mode:'cors',credentials:'omit'}))},"
    "ready:function(){return Promise.resolve()},"
    "navigate:function(p){console.log('genfleet.navigate',p)}})})();"
)


def _dev_token_script(token: str, origin: str) -> str:
    # JSON in a script element: `<`, `>` and `&` escaped so the value can't close the tag.
    value = json.dumps({"token": token, "origin": origin}).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"window.__gfDev={value};"


_SAFE_METHODS = frozenset({"GET", "HEAD"})
_VIEWER_HEADER_NAMES = {VIEWER_ID_HEADER.encode(), VIEWER_ROLE_HEADER.encode(), VIEW_ID_HEADER.encode()}


def dev_app(
    views: Views, viewer: Viewer | None, *, token: str | None = None, allowed_hosts: Collection[str] | None = None,
) -> Any:
    """``views`` as the platform would serve it, for local runs and tests.

    - The page runs sandboxed (CSP ``sandbox``, no ``allow-same-origin``), so it
      has an opaque origin as in the app: cookies and storage throw, and its
      fetches carry ``Origin: null``, answered with CORS for ``null``.
    - A request with ``Authorization: Bearer <token>`` gets ``viewer``. The
      token is random per run (``token``, else ``secrets.token_urlsafe``) and
      only the view's own HTML pages carry it (a nonce'd inline script; HTML
      gets no CORS, so no other site can read it). Any page on any site can
      send an ``Origin: null`` fetch here, so any other bearer is anonymous.
    - A ``Host`` not in ``allowed_hosts`` is a 421 (DNS rebinding);
      ``dev_server`` passes ``127.0.0.1``, ``localhost`` and its bound host,
      with its port. ``None`` (in-process tests) checks nothing.
      The bearer is removed, and any viewer header the browser sent is replaced.
    - ``POST``/``PUT``/``PATCH``/``DELETE`` without a bearer: 401.
    - Without a bearer only the page and its files come back: any other type is
      a 401 (redirects and 304s pass, without a body).
    - Only ``PASSED_RESPONSE_HEADERS`` of yours reach the browser (a ``location``
      only when it stays on the view), then the security headers and
      ``cache-control``, once each.
    - ``/__gf/view.js`` is a stand-in for the platform script.
    """

    dev_token = token or secrets.token_urlsafe(24)
    expected = f"Bearer {dev_token}".encode()
    hosts = {h.lower() for h in allowed_hosts} if allowed_hosts is not None else None

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await views(scope, receive, send)
            return
        incoming = scope.get("headers", [])
        host = next((v.decode("latin-1").lower() for k, v in incoming if k.lower() == b"host"), "")
        if hosts is not None and host not in hosts:
            await _send(send, 421, b'{"error":"misdirected_request"}', "application/json", {})
            return
        origin = _origin_of(scope)
        cors = _cors(incoming)
        security = {**security_headers(origin), **cors}
        if scope["path"] == CLIENT_SCRIPT_PATH:
            await _send(send, 200, DEV_CLIENT_SCRIPT.encode(), "application/javascript", security)
            return
        method = scope.get("method", "GET")
        if method == "OPTIONS":
            await _send(send, 204, b"", "text/plain", {**security, "access-control-allow-methods": "GET, HEAD, POST, PUT, PATCH, DELETE",
                                                        "access-control-allow-headers": "authorization, content-type", "access-control-max-age": "600"})
            return
        presented = next((v.strip() for k, v in incoming if k.lower() == b"authorization"), b"")
        signed_in = hmac.compare_digest(presented, expected)
        if not signed_in and method not in _SAFE_METHODS:
            await _send(send, 401, b'{"error":"token_required"}', "application/json", security)
            return
        headers = [(k, v) for k, v in incoming if k.lower() not in _VIEWER_HEADER_NAMES and k.lower() != b"authorization"]
        if signed_in and viewer is not None:
            headers += [
                (VIEWER_ID_HEADER.encode(), viewer.user_id.encode()),
                (VIEWER_ROLE_HEADER.encode(), viewer.role.encode()),
                (VIEW_ID_HEADER.encode(), viewer.view.encode()),
            ]
        state: dict[str, Any] = {"drop_body": False, "refused": False, "html": None, "start": None}
        nonce = secrets.token_urlsafe(16)

        async def secured(message: Any) -> None:
            if message["type"] == "http.response.start":
                status = message["status"]
                agent = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
                if status in (204, 304) or 300 <= status < 400:
                    state["drop_body"] = True
                elif not signed_in and not _anonymous_type_ok(agent.get("content-type")):
                    state["refused"] = True
                    await _send(send, 401, b'{"error":"sign_in_required"}', "application/json", security)
                    return
                kept = {
                    name: agent[name]
                    for name in PASSED_RESPONSE_HEADERS
                    if name in agent and (name != "location" or _location_ok(agent[name]))
                }
                out = {**kept, **security, "cache-control": cache_control_for(agent.get("content-type"))}
                if (agent.get("content-type") or "").lower().startswith("text/html") and not state["drop_body"]:
                    # The view's own page: the dev token goes in here, by a nonce'd
                    # inline script, and the page gets no CORS (no other site reads it).
                    out.pop("access-control-allow-origin", None)
                    out["content-security-policy"] = out["content-security-policy"].replace(
                        "base-uri", f"script-src {origin} 'nonce-{nonce}'; base-uri", 1)
                    state["html"] = bytearray()
                    state["start"] = {**message, "headers": [(k.encode(), v.encode()) for k, v in out.items()]}
                    return
                message = {**message, "headers": [(k.encode(), v.encode()) for k, v in out.items()]}
            elif message["type"] == "http.response.body":
                if state["refused"]:
                    return
                if state["drop_body"]:
                    message = {**message, "body": b""}
                elif state["html"] is not None:
                    state["html"] += message.get("body", b"")
                    if message.get("more_body"):
                        return
                    tag = f'<script nonce="{nonce}">{_dev_token_script(dev_token, origin)}</script>'.encode()
                    body = bytes(state["html"])
                    at = body.lower().find(b"<head>")
                    body = body[: at + 6] + tag + body[at + 6 :] if at >= 0 else tag + body
                    await send(state["start"])
                    message = {"type": "http.response.body", "body": body}
            await send(message)

        await views({**scope, "headers": headers}, receive, secured)

    return app


def _origin_of(scope: Any) -> str:
    host = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k.lower() == b"host"), "127.0.0.1")
    return f"{scope.get('scheme', 'http')}://{host}"


def _cors(incoming: Any) -> dict[str, str]:
    origin = next((v for k, v in incoming if k.lower() == b"origin"), None)
    return {"access-control-allow-origin": "null", "vary": "origin"} if origin == b"null" else {"vary": "origin"}


async def _send(send: Any, status: int, body: bytes, content_type: str, security: dict[str, str]) -> None:
    headers = {"content-type": content_type, **security, "cache-control": cache_control_for(content_type)}
    await send({"type": "http.response.start", "status": status, "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})
    await send({"type": "http.response.body", "body": body})


def dev_server(views: Views, *, viewer: Viewer | None = None, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve ``views`` locally (``dev_app``) on ``host:port``. Blocks.

    ``viewer`` is stamped on token-bearing requests; by default user
    ``dev-user``, role ``admin``, view ``dev``.
    """
    import uvicorn

    allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"{host}:{port}"}
    app = dev_app(views, viewer or Viewer(user_id="dev-user", role="admin", view="dev"), allowed_hosts=allowed)
    uvicorn.run(app, host=host, port=port)


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
