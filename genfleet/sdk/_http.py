"""The one HTTP opener for SDK calls that carry a bearer token.

Plain ``urllib.request.urlopen`` follows 301/302/303/307/308 redirects to any
host and re-sends the request's headers, ``Authorization`` included. The
platform endpoints the SDK calls never redirect, so every such call goes
through :data:`OPENER`, which refuses redirects: a 3xx answer is raised as an
``urllib.error.HTTPError`` with its status and nothing is sent to
``Location``. Callers turn that into their own error with
:data:`REDIRECT_REFUSED` (see :func:`is_redirect`).
"""

from __future__ import annotations

import urllib.request

__all__ = ["OPENER", "REDIRECT_REFUSED", "is_redirect"]

#: The message a caller puts on the error it raises for a refused redirect.
REDIRECT_REFUSED = "the platform answered with a redirect, which the SDK does not follow"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so the bearer token never goes where ``Location`` points."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(_NoRedirect)


def is_redirect(status: int) -> bool:
    """Whether ``status`` is a redirect the opener refused."""
    return 300 <= status < 400
