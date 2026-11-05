# Agent views

An agent can serve its own pages, such as a team roster editor or a live
pipeline, and the members of the workspace that installed it open them from
the agent's page in Genfleet. No second login: the platform tells your code
who is looking.

Views are available to agents published by Genfleet for now.

## Declare them

In `genfleet.toml`:

```toml
[agent]
name = "sales-assistant"
views_app = "sales_views:views"   # module:symbol, next to this file

[[views]]
id = "team"                        # lowercase letters, digits and '-'
title = "Sales team"               # the tab label; defaults to the id
path = "/team"                     # the page; everything it loads lives under it
min_role = "admin"                 # owner | admin | member (default admin)
```

At most 8 views. `min_role` is the lowest workspace role that may open the
view: `owner`, `admin` (owners and admins) or `member` (every member). A
request outside every declared `path` is refused.

## Serve them

```python
from pathlib import Path

from genfleet.sdk.views import Viewer, Views

views = Views(root=Path(__file__).parent)
views.static("/team", "static/team")      # index.html, app.js, app.css

@views.route("/team/api/roster")
async def roster(request, viewer: Viewer):
    return {"people": load_roster(), "you": viewer.user_id}

@views.route("/team/api/roster", methods=["POST"], min_role="owner")
async def save(request, viewer: Viewer):
    body = await request.json()
    ...
    return {"saved": True}
```

- `static` files load without a viewer. They are your page, script and
  style; keep data out of them. A relative folder is found under `root`.
- The page is `path + "/"` (`/team/`); the bare `/team` redirects there, so
  the page's relative URLs (`app.js`, `api/roster`) stay inside the view.
- A `route` handler gets the signed-in `Viewer` (`user_id`, `role`, `view`).
  It answers 401 when there is no viewer and 403 when the role is below the
  route's `min_role`. Return a dict or list for JSON, or any Starlette
  response.
- The viewer comes in the `x-genfleet-viewer-id`, `x-genfleet-viewer-role`
  and `x-genfleet-view` headers. The platform sets them and removes any copy
  a browser sent. `viewer_from_headers(request.headers)` reads them if you use
  your own ASGI app (FastAPI or Starlette) instead of `Views`.

## The page

Load the platform script first, then fetch your data through it:

```html
<!doctype html>
<link rel="stylesheet" href="app.css">
<script src="/__gf/view.js"></script>
<script src="app.js"></script>
```

```js
// app.js
const res = await genfleet.fetch("api/roster");
const roster = await res.json();
```

`genfleet.fetch(url, init)` works like `fetch` for your own view's URLs and
adds the viewer's short-lived credential; it refuses other origins.
`genfleet.ready()` resolves once the page is signed in.
`genfleet.navigate(path)` asks Genfleet to open one of its own pages.

**No inline script or style.** Every view is served with a Content Security
Policy of `default-src 'self'` with no inline allowance. A single HTML file
with `<script>` or `<style>` blocks won't run: ship script and style as
separate files next to the page. Nothing loads from another origin either
(no CDN scripts, fonts or images), and forms post only to the view itself.
Responses are not cached, and the page can't set cookies that Genfleet
keeps.

## Try it locally

```sh
pip install 'genfleet-sdk[serve]'
python -m genfleet.sdk.views sales_views:views --role admin --view team
```

It serves your views on `http://127.0.0.1:8765/` (open
`http://127.0.0.1:8765/team/`) as Genfleet does: a request made with
`genfleet.fetch` gets the fake viewer and one without gets none; a write
without it is refused; only `content-type`, `content-language`, `etag`,
`last-modified`, a relative `location` and `content-disposition` of your
response headers pass (no cookies, no CORS); and every response carries the
same security headers, so an inline script fails locally as it would in
Genfleet. `/__gf/view.js` is a stand-in that needs no sign-in. In code: `dev_server(views, viewer=Viewer(...))`
or `dev_app(...)` for tests.
