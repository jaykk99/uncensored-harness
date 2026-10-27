"""Tests for app/integrations.py — native direct-API integrations.

The httpx client factory (integrations._client) is stubbed with canned
responses; no network is touched. Env credentials are monkeypatched.
"""

from __future__ import annotations

import base64
import json

import pytest

from app import integrations as NI
from app.integrations import Fal, GitHub, GitLab, Gmail, Vercel


# ---------------------------------------------------------------------------
# Fake httpx surface
# ---------------------------------------------------------------------------

class FakeResp:
    def __init__(self, json_data=None, status_code=200, text=""):
        self._json = json_data
        self.status_code = status_code
        self.text = text or (json.dumps(json_data) if json_data is not None else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeClient:
    def __init__(self, routes):
        # routes: {(method, path_suffix): FakeResp | callable}
        self._routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _dispatch(self, method, url, kw):
        self.calls.append((method, url, kw))
        for (m, suffix), resp in self._routes.items():
            if m == method and url.endswith(suffix):
                return resp(url, kw) if callable(resp) else resp
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        return self._dispatch("GET", url, kw)

    def post(self, url, **kw):
        return self._dispatch("POST", url, kw)

    def delete(self, url, **kw):
        return self._dispatch("DELETE", url, kw)


@pytest.fixture
def fake_http(monkeypatch):
    """Swap integrations._client for a stub. Returns (routes, clients)."""
    routes: dict = {}
    clients: list[FakeClient] = []

    def factory(timeout, **kw):
        c = FakeClient(routes)
        clients.append(c)
        return c

    monkeypatch.setattr(NI, "_client", factory)
    return routes, clients


_CRED_VARS = (
    "GITHUB_TOKEN", "GITLAB_TOKEN", "GITLAB_HOST",
    "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN",
    "VERCEL_TOKEN", "VERCEL_TEAM_ID", "FAL_KEY",
)


@pytest.fixture(autouse=True)
def clean_credential_env(monkeypatch):
    """Clear every integration credential before each test.

    The *_not_connected_message tests below assert the "not connected"
    path specifically, which only holds when these vars are genuinely
    unset -- without this, a GITHUB_TOKEN (or similar) already present in
    whatever environment runs the suite (CI, a dev shell with `gh` set up,
    this repo's own sandbox) silently flips that test to the "connected"
    branch and it fails with an unrelated-looking network-call assertion.
    Individual tests still opt in to a credential via gh_env/gl_env/etc.
    """
    for name in _CRED_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gh_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-test")


@pytest.fixture
def gl_env(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "gl-test")


@pytest.fixture
def gmail_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "csec")
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "rtok")
    NI.Gmail._token_cache = {}


@pytest.fixture
def vercel_env(monkeypatch):
    monkeypatch.setenv("VERCEL_TOKEN", "vc-test")


@pytest.fixture
def fal_env(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "fal-test")


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

def test_github_not_connected_message(fake_http):
    assert NI.github_action("repos", {}) == \
        "GitHub is not connected — set GITHUB_TOKEN."


def test_github_repos_mapping(fake_http, gh_env):
    routes, _ = fake_http
    routes[("GET", "/user/repos")] = FakeResp([{
        "full_name": "jaykk99/x", "private": True, "description": "d",
        "updated_at": "2026-10-20T00:00:00Z", "html_url": "https://github.com/jaykk99/x"}])
    repos = GitHub.repos()
    assert repos[0] == {"name": "jaykk99/x", "private": True, "description": "d",
                        "updated": "2026-10-20T00:00:00Z",
                        "url": "https://github.com/jaykk99/x"}


def test_github_create_issue(fake_http, gh_env):
    routes, _ = fake_http
    routes[("POST", "/repos/o/r/issues")] = FakeResp(
        {"number": 42, "html_url": "https://github.com/o/r/issues/42"})
    out = NI.github_action("create_issue",
                           {"owner_repo": "o/r", "title": "bug", "body": "b"})
    assert out == {"number": 42, "url": "https://github.com/o/r/issues/42"}


def test_github_create_issue_requires_title(fake_http, gh_env):
    with pytest.raises(ValueError, match="title"):
        NI.github_action("create_issue", {"owner_repo": "o/r", "title": "  "})


def test_github_dispatch(fake_http, gh_env):
    routes, _ = fake_http
    routes[("POST", "/repos/o/r/actions/workflows/ci.yml/dispatches")] = FakeResp({})
    out = GitHub.dispatch("o/r", "ci.yml", "main", {"x": "1"})
    assert out == {"ok": True, "workflow": "ci.yml", "ref": "main"}


def test_github_404_raises_clear_error(fake_http, gh_env):
    routes, _ = fake_http
    routes[("GET", "/repos/nope/nope")] = FakeResp(None, 404, "Not Found")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        GitHub.repo("nope/nope")


def test_github_unknown_action(fake_http, gh_env):
    out = NI.github_action("frobnicate", {})
    assert "unknown github action" in out and "repos" in out


def test_github_auth_header(fake_http, gh_env):
    routes, clients = fake_http
    routes[("GET", "/user/repos")] = FakeResp([])
    GitHub.repos()
    _m, _u, kw = clients[0].calls[0]
    assert kw["headers"]["Authorization"] == "Bearer gh-test"
    assert "2022-11-28" in kw["headers"]["X-GitHub-Api-Version"]


# ---------------------------------------------------------------------------
# GitLab
# ---------------------------------------------------------------------------

def test_gitlab_not_connected_message(fake_http):
    assert NI.gitlab_action("projects", {}) == \
        "GitLab is not connected — set GITLAB_TOKEN."


def test_gitlab_projects_and_pipelines(fake_http, gl_env):
    routes, _ = fake_http
    routes[("GET", "/api/v4/projects")] = FakeResp([{
        "id": 7, "path_with_namespace": "jaykk99/site", "visibility": "public",
        "last_activity_at": "2026-10-19T00:00:00Z",
        "web_url": "https://gitlab.com/jaykk99/site"}])
    routes[("GET", "/api/v4/projects/7/pipelines")] = FakeResp([{
        "id": 99, "status": "success", "ref": "main", "sha": "abcdef123456",
        "created_at": "2026-10-19T01:00:00Z",
        "web_url": "https://gitlab.com/jaykk99/site/-/pipelines/99"}])
    (proj,) = NI.gitlab_action("projects", {})
    assert proj["name"] == "jaykk99/site"
    (pipe,) = NI.gitlab_action("pipelines", {"project_id": "7"})
    assert pipe == {"id": 99, "status": "success", "ref": "main", "sha": "abcdef12",
                    "created": "2026-10-19T01:00:00Z",
                    "url": "https://gitlab.com/jaykk99/site/-/pipelines/99"}


def test_gitlab_custom_host(monkeypatch, fake_http, gl_env):
    monkeypatch.setenv("GITLAB_HOST", "https://git.example.com")
    routes, clients = fake_http
    routes[("GET", "/api/v4/projects")] = FakeResp([])
    GitLab.projects()
    _m, url, kw = clients[0].calls[0]
    assert url.startswith("https://git.example.com/api/v4/projects")
    assert kw["headers"]["PRIVATE-TOKEN"] == "gl-test"


# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------

def test_gmail_not_connected_message(fake_http):
    assert "one-time Google" in NI.gmail_action("search", {"query": "x"})


def _oauth_routes(routes):
    routes[("POST", "https://oauth2.googleapis.com/token")] = FakeResp(
        {"access_token": "ya-test", "expires_in": 3600})


def test_gmail_token_refresh_and_cache(fake_http, gmail_env):
    routes, clients = fake_http
    _oauth_routes(routes)
    t1 = Gmail._access_token()
    t2 = Gmail._access_token()
    assert t1 == t2 == "ya-test"
    # One HTTP call total — the second read came from cache.
    assert len(clients) == 1


def test_gmail_send_builds_rfc2822(fake_http, gmail_env):
    routes, clients = fake_http
    _oauth_routes(routes)
    routes[("POST", "/gmail/v1/users/me/messages/send")] = FakeResp({"id": "m1"})

    def capture(url, kw):
        raw = kw["json"]["raw"]
        decoded = base64.urlsafe_b64decode(raw).decode()
        assert "To: a@b.com" in decoded
        assert "Subject: hi" in decoded
        return FakeResp({"id": "m1"})

    routes[("POST", "/gmail/v1/users/me/messages/send")] = capture
    out = NI.gmail_action("send", {"to": "a@b.com", "subject": "hi",
                                   "body": "hello"})
    assert out == {"id": "m1", "to": "a@b.com"}


def test_gmail_send_rejects_bad_address(fake_http, gmail_env):
    with pytest.raises(ValueError, match="valid 'to' address"):
        NI.gmail_action("send", {"to": "not-an-email", "subject": "s"})


def test_gmail_oauth_failure_is_honest(fake_http, gmail_env):
    routes, _ = fake_http
    routes[("POST", "https://oauth2.googleapis.com/token")] = FakeResp(
        None, 400, "invalid_grant")
    with pytest.raises(RuntimeError, match="one-time Google connect"):
        Gmail._access_token()


# ---------------------------------------------------------------------------
# Vercel
# ---------------------------------------------------------------------------

def test_vercel_not_connected_message(fake_http):
    assert NI.vercel_action("projects", {}) == \
        "Vercel is not connected — set VERCEL_TOKEN."


def _vercel_project_routes(routes, repo_id="123"):
    routes[("GET", "/v9/projects")] = FakeResp({"projects": [{
        "id": "prj_1", "name": "mysite", "framework": "nextjs",
        "updatedAt": 123, "link": {}}]})
    routes[("GET", "/v9/projects/prj_1")] = FakeResp({
        "id": "prj_1", "link": {"type": "github", "repoId": repo_id}})


def test_vercel_redeploy_from_git_link(fake_http, vercel_env):
    routes, clients = fake_http
    _vercel_project_routes(routes)

    def capture(url, kw):
        body = kw["json"]
        assert body["gitSource"] == {"type": "github", "repoId": "123",
                                     "ref": "main"}
        return FakeResp({"id": "dpl_1", "url": "mysite-x.vercel.app",
                         "readyState": "QUEUED"})

    routes[("POST", "/v13/deployments")] = capture
    out = NI.vercel_action("redeploy", {"project": "mysite"})
    assert out == {"id": "dpl_1", "url": "mysite-x.vercel.app",
                   "state": "QUEUED"}


def test_vercel_redeploy_unknown_project(fake_http, vercel_env):
    routes, _ = fake_http
    routes[("GET", "/v9/projects")] = FakeResp({"projects": []})
    with pytest.raises(ValueError, match="no such vercel project"):
        NI.vercel_action("redeploy", {"project": "ghost"})


def test_vercel_set_env_upserts(fake_http, vercel_env):
    routes, clients = fake_http
    _vercel_project_routes(routes)
    routes[("GET", "/v9/projects/prj_1/env")] = FakeResp({"envs": [{
        "id": "env_old", "key": "API_KEY", "target": ["production"]}]})
    routes[("DELETE", "/v9/projects/prj_1/env/env_old")] = FakeResp({})
    routes[("POST", "/v10/projects/prj_1/env")] = FakeResp({"id": "env_new"})
    out = NI.vercel_action("set_env", {"project": "mysite", "key": "API_KEY",
                                       "value": "newval"})
    assert out["id"] == "env_new"
    methods = [m for c in clients for m, _u, _k in c.calls]
    assert "DELETE" in methods  # old duplicate removed first
    assert "POST" in methods    # then the fresh value is created


def test_vercel_team_id_forwarded(monkeypatch, fake_http, vercel_env):
    monkeypatch.setenv("VERCEL_TEAM_ID", "team_9")
    routes, clients = fake_http
    routes[("GET", "/v9/projects")] = FakeResp({"projects": []})
    Vercel.projects()
    _m, _u, kw = clients[0].calls[0]
    assert kw["params"]["teamId"] == "team_9"


# ---------------------------------------------------------------------------
# Fal
# ---------------------------------------------------------------------------

def test_fal_not_connected_message(fake_http):
    assert NI.fal_action("generate", {"prompt": "x"}) == \
        "Fal is not connected — set FAL_KEY."


def test_fal_generate_and_result(fake_http, fal_env):
    routes, _ = fake_http
    routes[("POST", "https://queue.fal.run/fal-ai/flux-2-dev")] = FakeResp(
        {"request_id": "req_1"})
    states = [{"status": "IN_QUEUE"}, {"status": "COMPLETED"}]
    routes[("GET", "/requests/req_1/status")] = lambda u, k: FakeResp(states.pop(0))
    routes[("GET", "/requests/req_1")] = FakeResp({"images": [{
        "url": "https://img.example/a.png", "width": 1024, "height": 576}]})

    gen = NI.fal_action("generate", {"prompt": "a red barn"})
    assert gen["request_id"] == "req_1"
    res = NI.fal_action("result", {"request_id": "req_1", "wait_s": 30})
    assert res["images"][0]["url"] == "https://img.example/a.png"


def test_fal_result_timeout_is_honest(monkeypatch, fake_http, fal_env):
    routes, _ = fake_http
    routes[("GET", "/requests/req_9/status")] = FakeResp({"status": "IN_QUEUE"})
    monkeypatch.setattr(NI.time, "sleep", lambda s: None)
    # wait_s small; loop exits on the real clock, status never completes.
    out = None
    try:
        NI.fal_action("result", {"request_id": "req_9", "wait_s": 5})
    except RuntimeError as e:
        out = str(e)
    assert out and "not done after 5s" in out


# ---------------------------------------------------------------------------
# Registry + router arg validation
# ---------------------------------------------------------------------------

def test_integration_status_shape():
    st = NI.integration_status()
    assert set(st) == {"github", "gitlab", "gmail", "vercel", "fal"}
    assert all(isinstance(v, bool) for v in st.values())


def test_routers_reject_missing_args(fake_http, gh_env, vercel_env, fal_env):
    with pytest.raises(ValueError, match="missing required arg"):
        NI.github_action("repo", {})
    with pytest.raises(ValueError, match="missing required arg"):
        NI.vercel_action("set_env", {"project": "p", "key": "k"})
    with pytest.raises(ValueError, match="missing required arg"):
        NI.fal_action("status", {})


def test_tool_wrapper_formats_errors(fake_http, gh_env):
    from app import tools as T
    out = T._integration_call(NI.github_action, "repo", {})
    assert out.startswith("integration error: missing required arg")
    out2 = T._integration_call(NI.github_action, "bogus", {})
    assert "unknown github action" in out2
    # Dict results come back as JSON text.
    routes, _ = fake_http
    routes[("GET", "/user/repos")] = FakeResp([])
    out3 = T._integration_call(NI.github_action, "repos", {})
    assert out3.strip() == "[]"
