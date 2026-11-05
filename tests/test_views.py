"""Agent views: the manifest's `[[views]]`, the `Views` app and the dev server."""

from __future__ import annotations

import httpx
import pytest

from genfleet.sdk.manifest import ManifestError, load_agent_manifest
from genfleet.sdk.views import SECURITY_HEADERS, Viewer, Views, dev_app, role_meets, viewer_from_headers

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


async def test_dev_server_fakes_the_viewer_replaces_spoofs_and_sends_the_platform_csp(tmp_path):
    app = dev_app(make_views(tmp_path), Viewer("dev", "member", "team"))
    res = await call(app, "GET", "/team/api/me", headers={"x-genfleet-viewer-role": "owner"})
    assert res.json() == {"user": "dev", "role": "member"}
    assert res.headers["content-security-policy"] == SECURITY_HEADERS["content-security-policy"]
    assert "unsafe-inline" not in res.headers["content-security-policy"]
    script = await call(app, "GET", "/__gf/view.js")
    assert script.status_code == 200 and "genfleet" in script.text
    assert (await call(dev_app(make_views(tmp_path), None), "GET", "/team/api/me")).status_code == 401


def test_view_paths_are_checked():
    with pytest.raises(ValueError):
        Views().route("team")
    with pytest.raises(ValueError):
        Views().route("/a/../b")
    with pytest.raises(ValueError):
        Views().route("/a", min_role="root")  # type: ignore[arg-type]
