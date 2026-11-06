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
- Genfleet serves every agent's views from one site, each agent under a path
  of its own; your code sees paths from `/` as written above. So **use
  relative URLs** for your own files and routes (`app.js`, `api/roster`,
  `../pipeline/`). An absolute path such as `/team/app.js` would point
  outside your agent. The one absolute path is the platform script,
  `/__gf/view.js`. A redirect to a path of yours (`/team/`) is fixed up for
  you.
- A `route` handler gets the signed-in `Viewer` (`user_id`, `role`, `view`).
  It answers 401 when there is no viewer and 403 when the role is below the
  route's `min_role`. Return a dict or list for JSON, or any Starlette
  response.
- The viewer comes in the `x-genfleet-viewer-id`, `x-genfleet-viewer-role`
  and `x-genfleet-view` headers. The platform sets them and removes any copy
  a browser sent. `viewer_from_headers(request.headers)` reads them if you use
  your own ASGI app (FastAPI or Starlette) instead of `Views`.

## Save workspace content

A view can read and save content files of the workspace that installed the
agent, such as its routing rules. List them in the view:

```toml
[[views]]
id = "team"
path = "/team"
editable = ["assignment_rules.toml"]   # exact content paths, at most 16
```

The platform serves them at `__gf/content` under the view's path, so from
the page at `/team/` the URL is the relative `__gf/content`:

```js
const res = await genfleet.fetch('__gf/content?path=assignment_rules.toml')
const { basedOn, text } = await res.json()      // text is null when the file is missing

const saved = await genfleet.fetch(
  `__gf/content?path=assignment_rules.toml&basedOn=${basedOn}`,
  { method: 'PUT', body: newText },
)
// 200 {basedOn, status}: a new content version, made by the viewer
// 409 {error: "content_stale"}: someone saved since you read; read again
```

- Saving is for workspace owners and admins, whoever else may open the view.
- A save is a new version of the workspace's content, with the viewer
  recorded as its author. Validate the text with your own routes before you
  save it: the agent loads the new version when it starts again.
- At most 256 KiB of UTF-8 text per file. `basedOn` is required on a save.
- Your code never sees these requests, and the local dev server does not
  serve them.

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
adds the viewer's short-lived credential; it refuses anything else. A view is
one page: use client-side routing rather than full-page links between
pages.
`genfleet.ready()` resolves once the page is signed in.
`genfleet.navigate(path)` asks Genfleet to open one of its own pages.

**The page has no cookies and no storage.** It runs in a sandboxed frame
with an opaque origin: `document.cookie`, `localStorage`, `sessionStorage`
and IndexedDB throw or are empty, and your responses can't set cookies.
Keep state in the page's memory, or save it through your routes. Your
fetches are cross-origin requests from that opaque origin; `genfleet.fetch`
handles that.

**No inline script or style.** Every view is served with a Content Security
Policy with no inline allowance. A single HTML file with `<script>` or
`<style>` blocks won't run: ship script and style as separate files next to
the page. Nothing loads from another site either (no CDN scripts, fonts or
images), and forms post only to the view itself. HTML and JSON responses
are never cached; scripts, styles and images are revalidated on each load.

## Try it locally

```sh
pip install 'genfleet-sdk[serve]'
python -m genfleet.sdk.views sales_views:views --role admin --view team
```

It serves your views on `http://127.0.0.1:8765/` (open
`http://127.0.0.1:8765/team/`) as Genfleet does:

- the page is sandboxed, so cookies and storage fail locally too;
- a request made with `genfleet.fetch` gets the fake viewer and one without
  gets none; a write without it is refused, and so is anything but your page
  and its files (redirects and 304s pass);
- only `content-type`, `content-language`, `etag`, `last-modified`, a
  `location` that stays on the view and `content-disposition` of your
  response headers pass (no cookies, no CORS of yours);
- every response carries the same security headers, so an inline script
  fails locally as it would in Genfleet.

`/__gf/view.js` is a stand-in that needs no sign-in. In code: `dev_server(views, viewer=Viewer(...))`
or `dev_app(...)` for tests.
