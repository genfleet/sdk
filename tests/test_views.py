"""Agent views: the manifest's `[[views]]`, the `Views` app and the dev server."""

from __future__ import annotations

import httpx
import pytest

from genfleet.sdk.manifest import ManifestError, load_agent_manifest
from genfleet.sdk.views import Viewer, Views, dev_app, role_meets, security_headers, viewer_from_headers
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.responses import Response

MANIFEST = """
[agent]
name = "sa"
views_app = "sa_views:views"

[[views]]
id = "team"
title = "Sales team"
path = "/team"

[[views]]
id = "ops"
path = "/ops"
min_role = "owner"
"""


def write(tmp_path, text):
    (tmp_path / "genfleet.toml").write_text(text)
    return tmp_path


def test_manifest_reads_views(tmp_path):
    m = load_agent_manifest(write(tmp_path, MANIFEST))
    assert m.views_app == "sa_views:views"
    assert [(v.id, v.title, v.path, v.min_role) for v in m.views] == [("team", "Sales team", "/team", "admin"), ("ops", None, "/ops", "owner")]


@pytest.mark.parametrize("bad", [
    '[[views]]\nid = "Team"\npath = "/t"\n',
    '[[views]]\nid = "t"\npath = "t"\n',
    '[[views]]\nid = "t"\npath = "/a/../b"\n',
    '[[views]]\nid = "t"\npath = "/__gf/x"\n',
    '[[views]]\nid = "t"\npath = "/t"\nmin_role = "root"\n',
    '[[views]]\nid = "t"\npath = "/t"\nroles = ["owner"]\n',
    '[[views]]\nid = "t"\npath = "/a"\n\n[[views]]\nid = "t"\npath = "/b"\n',
])
def test_manifest_refuses_bad_views(tmp_path, bad):
    with pytest.raises(ManifestError):
        load_agent_manifest(write(tmp_path, '[agent]\nname = "sa"\nviews_app = "m:v"\n\n' + bad))


def test_views_need_an_app(tmp_path):
    with pytest.raises(ManifestError, match="views_app"):
        load_agent_manifest(write(tmp_path, '[agent]\nname = "sa"\n\n[[views]]\nid = "t"\npath = "/t"\n'))
    with pytest.raises(ManifestError, match="module:symbol"):
        load_agent_manifest(write(tmp_path, '[agent]\nname = "sa"\nviews_app = "nope"\n'))


def test_viewer_headers_and_role_floors():
    v = viewer_from_headers({"X-Genfleet-Viewer-Id": "u-1", "x-genfleet-viewer-role": "admin", "x-genfleet-view": "team"})
    assert v == Viewer("u-1", "admin", "team")
    assert viewer_from_headers({"x-genfleet-viewer-id": "u-1"}) is None
    assert [r for r in ("owner", "admin", "member", "billing") if role_meets(r, "admin")] == ["owner", "admin"]
    assert v.at_least("member") and not v.at_least("owner")


def make_views(tmp_path) -> Views:
    (tmp_path / "team").mkdir(exist_ok=True)
    (tmp_path / "team" / "index.html").write_text("<script src='/__gf/view.js'></script><script src='app.js'></script>")
    views = Views()
    views.static("/team", tmp_path / "team")

    @views.route("/team/api/me")
    async def me(request, viewer: Viewer):
        return {"user": viewer.user_id, "role": viewer.role}

    @views.route("/team/api/save", methods=["POST"], min_role="owner")
    async def save(request, viewer: Viewer):
        return {"saved": True}

    return views


async def call(app, method, path, **kw):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://v") as c:
        return await c.request(method, path, **kw)


async def test_routes_need_a_viewer_and_the_role(tmp_path):
    views = make_views(tmp_path)
    assert (await call(views, "GET", "/team/api/me")).status_code == 401
    as_admin = {"x-genfleet-viewer-id": "u-1", "x-genfleet-viewer-role": "admin", "x-genfleet-view": "team"}
    res = await call(views, "GET", "/team/api/me", headers=as_admin)
    assert res.json() == {"user": "u-1", "role": "admin"}
    assert (await call(views, "POST", "/team/api/save", headers=as_admin)).status_code == 403
    assert (await call(views, "POST", "/team/api/save", headers={**as_admin, "x-genfleet-viewer-role": "owner"})).status_code == 200


async def test_static_page_loads_without_a_viewer(tmp_path):
    res = await call(make_views(tmp_path), "GET", "/team/")
    assert res.status_code == 200 and "/__gf/view.js" in res.text


async def test_dev_server_stamps_the_viewer_only_for_a_token_and_replaces_spoofs(tmp_path):
    app = dev_app(make_views(tmp_path), Viewer("dev", "member", "team"), token="dev")
    anonymous = await call(app, "GET", "/team/api/me", headers={"x-genfleet-viewer-id": "u", "x-genfleet-viewer-role": "owner", "x-genfleet-view": "team"})
    assert anonymous.status_code == 401
    res = await call(app, "GET", "/team/api/me", headers={"authorization": "Bearer dev", "x-genfleet-viewer-role": "owner"})
    assert res.json() == {"user": "dev", "role": "member"}
    assert (await call(app, "POST", "/team/api/save")).status_code == 401
    script = await call(app, "GET", "/__gf/view.js")
    assert script.status_code == 200 and 'var T="dev"' in script.text


async def test_dev_server_sends_the_platform_headers_once_and_drops_the_agents(tmp_path):
    views = make_views(tmp_path)

    @views.route("/team/api/cookie")
    async def cookie(request, viewer):
        return Response("x", media_type="text/plain", headers={"set-cookie": "a=b; Domain=views.test", "access-control-allow-origin": "*",
                                      "content-security-policy": "default-src *", "location": "https://evil.example/"})

    res = await call(dev_app(views, Viewer("dev", "admin", "team"), token="dev"), "GET", "/team/api/cookie", headers={"authorization": "Bearer dev"})
    for dropped in ("set-cookie", "access-control-allow-origin", "location"):
        assert dropped not in res.headers
    assert res.headers.get_list("content-security-policy") == [security_headers("http://v")["content-security-policy"]]
    assert "unsafe-inline" not in res.headers["content-security-policy"]
    assert res.headers["cache-control"] == "private, no-cache"
    page = await call(dev_app(views, None), "GET", "/team/")
    assert page.headers["cache-control"] == "private, no-store"


async def test_the_bare_view_path_redirects_relatively_under_a_mount(tmp_path):
    host = Starlette(routes=[Mount("/views", app=make_views(tmp_path))])
    bare = await call(host, "GET", "/views/team")
    assert bare.status_code == 307 and bare.headers["location"] == "team/"
    assert httpx.URL("http://v/views/team").join(bare.headers["location"]).path == "/views/team/"
    page = await call(host, "GET", "/views/team/")
    assert page.status_code == 200 and "app.js" in page.text
    # The page's relative URLs land inside the view.
    assert httpx.URL("http://v/views/team/").join("api/me").path == "/views/team/api/me"
    assert (await call(host, "GET", "/views/team/api/me", headers={"x-genfleet-viewer-id": "u", "x-genfleet-viewer-role": "admin", "x-genfleet-view": "team"})).status_code == 200


async def test_no_path_traversal_out_of_a_static_folder(tmp_path):
    (tmp_path / "secret.txt").write_text("nope")
    views = make_views(tmp_path)
    for path in ("/team/../secret.txt", "/team/%2e%2e/secret.txt", "/team/..%2fsecret.txt"):
        res = await call(views, "GET", path)
        assert res.status_code == 404 and "nope" not in res.text, path


def test_static_mounts_are_ordered_longest_first(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    views = Views()
    views.static("/team", tmp_path / "a")
    views.static("/team/admin", tmp_path / "b")
    paths = [getattr(r, "path", None) for r in views.asgi().routes]
    assert paths.index("/team/admin") < paths.index("/team")


def test_a_relative_static_folder_needs_a_root(tmp_path):
    with pytest.raises(ValueError, match="root"):
        Views().static("/team", "static/team")
    (tmp_path / "static").mkdir()
    Views(root=tmp_path).static("/team", "static")


def test_view_paths_are_checked():
    with pytest.raises(ValueError):
        Views().route("team")
    with pytest.raises(ValueError):
        Views().route("/a/../b")
    with pytest.raises(ValueError):
        Views().route("/a", min_role="root")  # type: ignore[arg-type]



async def test_dev_server_sandboxes_the_page_and_answers_its_opaque_origin(tmp_path):
    app = dev_app(make_views(tmp_path), Viewer("dev", "admin", "team"), token="dev")
    page = await call(app, "GET", "/team/")
    csp = page.headers["content-security-policy"]
    assert "sandbox allow-scripts allow-forms allow-downloads" in csp and "allow-same-origin" not in csp
    assert "default-src http://v" in csp and "'self'" not in csp
    pre = await call(app, "OPTIONS", "/team/api/me", headers={"origin": "null", "access-control-request-method": "GET"})
    assert pre.status_code == 204 and pre.headers["access-control-allow-origin"] == "null"
    res = await call(app, "GET", "/team/api/me", headers={"origin": "null", "authorization": "Bearer dev"})
    assert res.headers["access-control-allow-origin"] == "null" and "access-control-allow-credentials" not in res.headers


async def test_dev_server_copies_the_platform_backstop_and_redirect_rule(tmp_path):
    views = make_views(tmp_path)

    @views.route("/team/api/go")
    async def go(request, viewer):
        return Response("moved", status_code=302, headers={"location": request.query_params.get("to", "next")}, media_type="text/plain")

    app = dev_app(views, Viewer("dev", "admin", "team"), token="dev")
    bare = await call(app, "GET", "/team")
    assert bare.status_code == 307 and bare.headers["location"] == "team/" and bare.content == b""
    signed = {"authorization": "Bearer dev"}
    moved = await call(app, "GET", "/team/api/go", headers=signed)
    assert moved.status_code == 302 and moved.headers["location"] == "next" and moved.content == b""
    for bad in ("https://evil.example/", "//evil.example", "/\tx"):
        res = await call(app, "GET", "/team/api/go", params={"to": bad}, headers=signed)
        assert "location" not in res.headers, bad


async def test_dev_server_refuses_data_to_a_request_without_a_viewer(tmp_path):
    views = make_views(tmp_path)

    @views.route("/team/api/open")
    async def leaky(request, viewer):  # pragma: no cover - the route refuses first
        return {"x": 1}

    app = dev_app(views, Viewer("dev", "admin", "team"), token="dev")
    assert (await call(app, "GET", "/team/api/open")).status_code == 401


async def test_dev_server_treats_any_other_bearer_as_anonymous(tmp_path):
    # Any page on any site can send an Origin: null fetch to the dev server.
    app = dev_app(make_views(tmp_path), Viewer("dev", "admin", "team"))
    for bearer in ("Bearer x", "Bearer dev", "bearer ", "Basic abc"):
        res = await call(app, "GET", "/team/api/me", headers={"origin": "null", "authorization": bearer})
        assert res.status_code == 401, bearer
        assert (await call(app, "POST", "/team/api/save", headers={"origin": "null", "authorization": bearer})).status_code == 401
    script = (await call(app, "GET", "/__gf/view.js")).text
    token = script.split('var T="', 1)[1].split('"', 1)[0]
    assert len(token) >= 24
    assert (await call(app, "GET", "/team/api/me", headers={"authorization": f"Bearer {token}"})).status_code == 200


async def test_dev_server_never_hands_its_script_to_another_page(tmp_path):
    app = dev_app(make_views(tmp_path), Viewer("dev", "admin", "team"), token="dev")
    res = await call(app, "GET", "/__gf/view.js", headers={"origin": "null"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers  # a sandboxed page can't fetch() it to read the token
    assert '"http://v"' in res.text  # the token goes only to the dev server's own origin
