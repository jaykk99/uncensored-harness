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


def test_github_create_issue_blank_title_direct():
    # The router's _req() rejects a blank title before create_issue is
    # reached, so the classmethod's own guard needs a direct call.
    with pytest.raises(ValueError, match="issue title is required"):
        GitHub.create_issue("o/r", "   ")
    with pytest.raises(ValueError, match="issue title is required"):
        GitHub.create_issue("o/r", "")


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


# ---------------------------------------------------------------------------
# GitHub — repo / file / issues / prs / workflow_runs
# ---------------------------------------------------------------------------

def test_github_repo_mapping(fake_http, gh_env):
    routes, _ = fake_http
    routes[("GET", "/repos/jaykk99/x")] = FakeResp({
        "full_name": "jaykk99/x", "private": False,
        "description": "d", "default_branch": "main",
        "stargazers_count": 3, "forks_count": 1,
        "open_issues_count": 2, "html_url": "https://github.com/jaykk99/x"})
    r = GitHub.repo("jaykk99/x")
    assert r["stars"] == 3 and r["default_branch"] == "main"
    assert r["open_issues"] == 2


def test_github_file_base64_decoded(fake_http, gh_env):
    routes, clients = fake_http
    content = base64.b64encode(b"hello file").decode()
    routes[("GET", "/repos/jaykk99/x/contents/a/b.txt")] = FakeResp({
        "path": "a/b.txt", "sha": "abc", "size": 10,
        "content": content, "encoding": "base64"})
    r = GitHub.file("jaykk99/x", "/a/b.txt")
    assert r["content"] == "hello file" and r["sha"] == "abc"
    # ref is passed as a query param when given.
    routes[("GET", "/repos/jaykk99/x/contents/a/b.txt")] = FakeResp({
        "path": "a/b.txt", "sha": "abc", "size": 10,
        "content": "plain", "encoding": "none"})
    r = GitHub.file("jaykk99/x", "a/b.txt", ref="dev")
    assert r["content"] == "plain"
    assert clients[-1].calls[-1][2]["params"] == {"ref": "dev"}


def test_github_issues_and_prs(fake_http, gh_env):
    routes, _ = fake_http
    routes[("GET", "/repos/jaykk99/x/issues")] = FakeResp([
        {"number": 1, "title": "bug", "state": "open",
         "labels": [{"name": "a"}], "html_url": "u1"},
        {"number": 2, "title": "pr-ish", "state": "open",
         "labels": [], "html_url": "u2", "pull_request": {}}])
    issues = GitHub.issues("jaykk99/x")
    assert issues[0]["labels"] == ["a"] and issues[0]["is_pr"] is False
    assert issues[1]["is_pr"] is True
    routes[("GET", "/repos/jaykk99/x/pulls")] = FakeResp([
        {"number": 5, "title": "feat", "state": "open",
         "head": {"ref": "f"}, "base": {"ref": "main"},
         "draft": True, "html_url": "u5"}])
    (pr,) = GitHub.prs("jaykk99/x")
    assert pr["head"] == "f" and pr["draft"] is True


def test_github_workflow_runs_named_and_all(fake_http, gh_env):
    routes, _ = fake_http
    run_row = {"id": 7, "name": "ci", "status": "completed",
               "conclusion": "success", "head_branch": "main",
               "event": "push", "created_at": "2026-01-01",
               "html_url": "u"}
    routes[("GET", "/actions/workflows/ci.yml/runs")] = FakeResp(
        {"workflow_runs": [run_row]})
    routes[("GET", "/actions/runs")] = FakeResp({"workflow_runs": [run_row]})
    named = GitHub.workflow_runs("jaykk99/x", workflow="ci.yml")
    assert named[0]["id"] == 7 and named[0]["conclusion"] == "success"
    all_runs = GitHub.workflow_runs("jaykk99/x")
    assert len(all_runs) == 1


def test_github_post_empty_body_returns_empty_dict(fake_http, gh_env):
    routes, _ = fake_http
    routes[("POST", "/dispatches")] = FakeResp(None, text="")
    out = GitHub.dispatch("jaykk99/x", "ci.yml")
    assert out == {"ok": True, "workflow": "ci.yml", "ref": "main"}


def test_raise_for_retryable_status(fake_http, gh_env):
    routes, _ = fake_http
    routes[("GET", "/user/repos")] = FakeResp(None, status_code=429,
                                              text="slow down")
    with pytest.raises(RuntimeError, match="transient — retry"):
        GitHub.repos()


def test_github_repos_limit_clamped(fake_http, gh_env):
    routes, clients = fake_http
    routes[("GET", "/user/repos")] = FakeResp([])
    GitHub.repos(limit=500)
    assert clients[-1].calls[-1][2]["params"]["per_page"] == 100
    GitHub.repos(limit=0)
    assert clients[-1].calls[-1][2]["params"]["per_page"] == 1


# ---------------------------------------------------------------------------
# GitLab — project / issues
# ---------------------------------------------------------------------------

def test_gitlab_project_and_issues(fake_http, gl_env):
    routes, clients = fake_http
    routes[("GET", "/api/v4/projects/42")] = FakeResp({
        "id": 42, "path_with_namespace": "grp/proj", "description": "d",
        "default_branch": "main", "star_count": 4, "forks_count": 2,
        "open_issues_count": 1, "web_url": "https://gitlab.com/grp/proj"})
    p = GitLab.project(42)
    assert p["stars"] == 4 and p["name"] == "grp/proj"
    routes[("GET", "/api/v4/projects/42/issues")] = FakeResp([
        {"iid": 3, "title": "t", "state": "opened",
         "labels": ["x"], "web_url": "u"}])
    (i,) = GitLab.issues(42)
    assert i["iid"] == 3 and i["labels"] == ["x"]
    routes[("GET", "/api/v4/projects")] = FakeResp([])
    GitLab.projects(search="foo", limit=5)
    params = clients[-1].calls[-1][2]["params"]
    assert params["search"] == "foo" and params["per_page"] == 5


def test_gitlab_pipelines_sha_shortened(fake_http, gl_env):
    routes, _ = fake_http
    routes[("GET", "/api/v4/projects/1/pipelines")] = FakeResp([
        {"id": 9, "status": "success", "ref": "main",
         "sha": "abcdef0123456789", "created_at": "t", "web_url": "u"}])
    (pl,) = GitLab.pipelines(1)
    assert pl["sha"] == "abcdef01"


# ---------------------------------------------------------------------------
# Gmail — search / get / body extraction / token cache
# ---------------------------------------------------------------------------

def _gmail_token_route(routes):
    routes[("POST", "oauth2.googleapis.com/token")] = FakeResp({
        "access_token": "tok-1", "expires_in": 3600})


def test_gmail_search_fills_snippets(fake_http, gmail_env):
    routes, _ = fake_http
    _gmail_token_route(routes)
    routes[("GET", "/gmail/v1/users/me/messages")] = FakeResp({
        "messages": [{"id": "m1", "threadId": "t1"},
                     {"id": "m2", "threadId": "t2"}]})
    routes[("GET", "/gmail/v1/users/me/messages/m1")] = FakeResp({
        "id": "m1", "threadId": "t1", "snippet": "snip one",
        "payload": {"headers": [
            {"name": "Subject", "value": "Hi"},
            {"name": "From", "value": "a@b.c"},
            {"name": "Date", "value": "Mon"}]}})
    routes[("GET", "/gmail/v1/users/me/messages/m2")] = FakeResp(
        None, status_code=404, text="gone")
    out = Gmail.search("from:x", limit=10)
    assert len(out) == 2
    assert out[0]["snippet"] == "snip one" and out[0]["subject"] == "Hi"
    # A failed per-message fetch leaves the snippet blank, not an error.
    assert out[1]["snippet"] == ""


def test_gmail_get_full_body_plain_and_html(fake_http, gmail_env):
    routes, _ = fake_http
    _gmail_token_route(routes)
    plain = base64.urlsafe_b64encode(b"plain body").decode()
    html = base64.urlsafe_b64encode(b"<p>html body</p>").decode()
    routes[("GET", "/gmail/v1/users/me/messages/m9")] = FakeResp({
        "id": "m9", "threadId": "t", "snippet": "s",
        "payload": {"headers": [{"name": "Subject", "value": "Sub"}],
                    "parts": [
                        {"mimeType": "text/html",
                         "body": {"data": html}},
                        {"mimeType": "text/plain",
                         "body": {"data": plain}}]}})
    out = Gmail.get("m9")
    assert out["body"] == "plain body"
    assert out["subject"] == "Sub"


def test_gmail_get_metadata_has_no_body(fake_http, gmail_env):
    routes, _ = fake_http
    _gmail_token_route(routes)
    routes[("GET", "/gmail/v1/users/me/messages/m1")] = FakeResp({
        "id": "m1", "payload": {"headers": []}})
    out = Gmail.get("m1", fmt="metadata")
    assert out["body"] == ""


def test_gmail_extract_body_variants():
    plain = base64.urlsafe_b64encode(b"p").decode()
    html = base64.urlsafe_b64encode(b"<b>h</b> x").decode()
    assert Gmail._extract_body({"mimeType": "text/plain",
                                "body": {"data": plain}}) == "p"
    assert Gmail._extract_body({"mimeType": "text/html",
                                "body": {"data": html}}) == " h  x"
    nested = {"parts": [{"mimeType": "text/plain",
                         "body": {"data": plain}}]}
    assert Gmail._extract_body(nested) == "p"
    assert Gmail._extract_body({}) == ""


def test_gmail_token_cached_between_calls(fake_http, gmail_env):
    routes, clients = fake_http
    _gmail_token_route(routes)
    routes[("POST", "/gmail/v1/users/me/messages/send")] = FakeResp(
        {"id": "sent-1"})
    Gmail.send("a@b.c", "s", "b")
    Gmail.send("a@b.c", "s2", "b2")
    token_calls = [c for c in clients[0].calls
                   if "oauth2" in c[1]] if clients else []
    # One token fetch total across both sends (the second reuses the cache).
    all_calls = [c for cl in clients for c in cl.calls]
    assert sum(1 for c in all_calls if "oauth2" in c[1]) == 1


def test_gmail_send_default_subject(fake_http, gmail_env):
    routes, _ = fake_http
    _gmail_token_route(routes)
    routes[("POST", "/gmail/v1/users/me/messages/send")] = FakeResp(
        {"id": "sent-2"})
    out = Gmail.send("a@b.c", "", "body")
    assert out["id"] == "sent-2" and out["to"] == "a@b.c"


# ---------------------------------------------------------------------------
# Vercel — projects / deployments / env_vars / redeploy errors
# ---------------------------------------------------------------------------

def _vercel_projects_route(routes, name="site", pid="prj_1"):
    routes[("GET", "/v9/projects")] = FakeResp({"projects": [
        {"id": pid, "name": name, "framework": "nextjs",
         "updatedAt": 123}]})
    return pid


def test_vercel_projects_mapping(fake_http, vercel_env):
    routes, _ = fake_http
    _vercel_projects_route(routes)
    (p,) = Vercel.projects()
    assert p["name"] == "site" and p["url"] == "https://site.vercel.app"
    assert p["framework"] == "nextjs"


def test_vercel_deployments_with_project_filter(fake_http, vercel_env):
    routes, clients = fake_http
    _vercel_projects_route(routes)
    routes[("GET", "/v6/deployments")] = FakeResp({"deployments": [
        {"uid": "d1", "state": "READY", "url": "site.vercel.app",
         "createdAt": 5, "meta": {"githubCommitMessage": "fix"}}]})
    (d,) = Vercel.deployments(project="site")
    assert d["id"] == "d1" and d["meta"] == "fix"
    params = clients[-1].calls[-1][2]["params"]
    assert params["projectId"] == "prj_1"
    Vercel.deployments()
    assert "projectId" not in clients[-1].calls[-1][2]["params"]


def test_vercel_redeploy_no_github_link(fake_http, vercel_env):
    routes, _ = fake_http
    _vercel_projects_route(routes)
    routes[("GET", "/v9/projects/prj_1")] = FakeResp({"link": {"type": "other"}})
    with pytest.raises(RuntimeError, match="no linked GitHub repo"):
        Vercel.redeploy("site")


def test_vercel_env_vars_and_unknown_project(fake_http, vercel_env):
    routes, _ = fake_http
    _vercel_projects_route(routes)
    routes[("GET", "/v9/projects/prj_1/env")] = FakeResp({"envs": [
        {"key": "K", "target": ["production"], "id": "env_1"}]})
    (e,) = Vercel.env_vars("site")
    assert e["key"] == "K" and e["id"] == "env_1"
    with pytest.raises(ValueError, match="no such vercel project"):
        Vercel.env_vars("ghost")
    with pytest.raises(ValueError, match="no such vercel project"):
        Vercel.set_env("ghost", "K", "v")


def test_vercel_set_env_new_key_no_delete(fake_http, vercel_env):
    routes, clients = fake_http
    _vercel_projects_route(routes)
    routes[("GET", "/v9/projects/prj_1/env")] = FakeResp({"envs": []})
    routes[("POST", "/v10/projects/prj_1/env")] = FakeResp({"id": "env_9"})
    out = Vercel.set_env("site", "NEW_KEY", "v")
    assert out == {"key": "NEW_KEY", "target": "production", "id": "env_9"}
    methods = [c[0] for cl in clients for c in cl.calls]
    assert "DELETE" not in methods


def test_vercel_params_team_id(monkeypatch, fake_http, vercel_env):
    monkeypatch.setenv("VERCEL_TEAM_ID", "team_1")
    routes, clients = fake_http
    routes[("GET", "/v9/projects")] = FakeResp({"projects": []})
    Vercel.projects()
    assert clients[-1].calls[-1][2]["params"]["teamId"] == "team_1"


# ---------------------------------------------------------------------------
# Fal — generate validation / status / result states
# ---------------------------------------------------------------------------

def test_fal_generate_requires_prompt(fake_http, fal_env):
    with pytest.raises(ValueError, match="prompt is required"):
        Fal.generate("   ")


def test_fal_generate_clamps_num_images(fake_http, fal_env):
    routes, clients = fake_http
    routes[("POST", "queue.fal.run/fal-ai/flux-2-dev")] = FakeResp(
        {"request_id": "r1"})
    Fal.generate("a cat", num_images=99)
    body = clients[-1].calls[-1][2]["json"]
    assert body["num_images"] == 4
    Fal.generate("a cat", num_images=0)
    assert clients[-1].calls[-1][2]["json"]["num_images"] == 1


def test_fal_status_passthrough(fake_http, fal_env):
    routes, _ = fake_http
    routes[("GET", "/requests/r1/status")] = FakeResp(
        {"status": "IN_PROGRESS"})
    assert Fal.status("r1") == {"status": "IN_PROGRESS"}


def test_fal_result_failed_is_honest(fake_http, fal_env):
    routes, _ = fake_http
    routes[("GET", "/requests/r1/status")] = FakeResp({"status": "FAILED"})
    with pytest.raises(RuntimeError, match="not done"):
        Fal.result("r1", wait_s=5)


def test_fal_result_polls_then_returns_images(monkeypatch, fake_http, fal_env):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    routes, _ = fake_http
    states = [{"status": "IN_PROGRESS"}, {"status": "COMPLETED"}]
    routes[("GET", "/requests/r1/status")] = lambda url, kw: FakeResp(states.pop(0))
    routes[("GET", "/requests/r1")] = FakeResp({"images": [
        {"url": "https://img/1.png", "width": 100, "height": 50}]})
    out = Fal.result("r1", wait_s=5)
    assert out["images"][0]["url"] == "https://img/1.png"


def test_fal_custom_model(monkeypatch, fake_http, fal_env):
    monkeypatch.setenv("FAL_IMAGE_MODEL", "fal-ai/flux-pro")
    routes, _ = fake_http
    routes[("POST", "queue.fal.run/fal-ai/flux-pro")] = FakeResp(
        {"request_id": "rx"})
    out = Fal.generate("x")
    assert out["model"] == "fal-ai/flux-pro"


# ---------------------------------------------------------------------------
# Routers — unknown actions, arg checks, full action coverage
# ---------------------------------------------------------------------------

def test_routers_unknown_actions(fake_http, gh_env, gl_env, gmail_env,
                                 vercel_env, fal_env):
    assert NI.github_action("bogus", {}).startswith("unknown github action")
    assert NI.gitlab_action("bogus", {}).startswith("unknown gitlab action")
    assert NI.gmail_action("bogus", {}).startswith("unknown gmail action")
    assert NI.vercel_action("bogus", {}).startswith("unknown vercel action")
    assert NI.fal_action("bogus", {}).startswith("unknown fal action")


def test_routers_not_connected_messages(fake_http):
    assert "GITLAB_TOKEN" in NI.gitlab_action("projects", {})
    assert "Google" in NI.gmail_action("search", {})
    assert "VERCEL_TOKEN" in NI.vercel_action("projects", {})
    assert "FAL_KEY" in NI.fal_action("generate", {})


def test_github_router_full_actions(fake_http, gh_env):
    routes, _ = fake_http
    routes[("GET", "/repos/jaykk99/x")] = FakeResp({
        "full_name": "jaykk99/x", "private": True, "description": None,
        "default_branch": "main", "stargazers_count": 0, "forks_count": 0,
        "open_issues_count": 0, "html_url": "u"})
    assert NI.github_action("repo", {"owner_repo": "jaykk99/x"})["name"] == "jaykk99/x"
    routes[("GET", "/repos/jaykk99/x/issues")] = FakeResp([])
    assert NI.github_action("issues", {"owner_repo": "jaykk99/x"}) == []
    routes[("GET", "/repos/jaykk99/x/pulls")] = FakeResp([])
    assert NI.github_action("prs", {"owner_repo": "jaykk99/x"}) == []
    routes[("GET", "/actions/runs")] = FakeResp({"workflow_runs": []})
    assert NI.github_action("workflow_runs", {"owner_repo": "jaykk99/x"}) == []
    routes[("POST", "/dispatches")] = FakeResp(None, text="")
    assert NI.github_action("dispatch",
                            {"owner_repo": "jaykk99/x", "workflow": "ci.yml"})["ok"] is True
    routes[("GET", "/repos/jaykk99/x/contents/f.txt")] = FakeResp({
        "path": "f.txt", "sha": "s", "size": 1, "content": "x",
        "encoding": "none"})
    assert NI.github_action("file", {"owner_repo": "jaykk99/x",
                                     "path": "f.txt"})["content"] == "x"


def test_gitlab_router_actions(fake_http, gl_env):
    routes, _ = fake_http
    routes[("GET", "/api/v4/projects")] = FakeResp([])
    assert NI.gitlab_action("projects", {}) == []
    routes[("GET", "/api/v4/projects/1")] = FakeResp({
        "id": 1, "path_with_namespace": "a/b", "description": None,
        "default_branch": "main", "star_count": 0, "forks_count": 0,
        "open_issues_count": 0, "web_url": "u"})
    assert NI.gitlab_action("project", {"project_id": "1"})["id"] == 1
    routes[("GET", "/api/v4/projects/1/issues")] = FakeResp([])
    assert NI.gitlab_action("issues", {"project_id": "1"}) == []
    routes[("GET", "/api/v4/projects/1/pipelines")] = FakeResp([])
    assert NI.gitlab_action("pipelines", {"project_id": "1"}) == []


def test_gmail_router_actions(fake_http, gmail_env):
    routes, _ = fake_http
    _gmail_token_route(routes)
    routes[("POST", "/gmail/v1/users/me/messages/send")] = FakeResp({"id": "s1"})
    assert NI.gmail_action("send", {"to": "a@b.c", "subject": "hi"})["id"] == "s1"
    routes[("GET", "/gmail/v1/users/me/messages")] = FakeResp({"messages": []})
    assert NI.gmail_action("search", {"query": "x"}) == []
    routes[("GET", "/gmail/v1/users/me/messages/m1")] = FakeResp({
        "id": "m1", "payload": {"headers": []}})
    assert NI.gmail_action("get", {"message_id": "m1"})["id"] == "m1"


def test_vercel_router_actions(fake_http, vercel_env):
    routes, _ = fake_http
    _vercel_projects_route(routes)
    assert NI.vercel_action("projects", {})[0]["name"] == "site"
    routes[("GET", "/v6/deployments")] = FakeResp({"deployments": []})
    assert NI.vercel_action("deployments", {}) == []
    routes[("GET", "/v9/projects/prj_1")] = FakeResp(
        {"link": {"type": "github", "repoId": 5}})
    routes[("POST", "/v13/deployments")] = FakeResp(
        {"id": "d9", "url": "u", "readyState": "READY"})
    out = NI.vercel_action("redeploy", {"project": "site"})
    assert out["id"] == "d9"
    routes[("GET", "/v9/projects/prj_1/env")] = FakeResp({"envs": []})
    assert NI.vercel_action("env_vars", {"project": "site"}) == []
    routes[("POST", "/v10/projects/prj_1/env")] = FakeResp({"id": "e1"})
    assert NI.vercel_action("set_env", {"project": "site", "key": "K",
                                        "value": "v"})["id"] == "e1"


def test_fal_router_actions(fake_http, fal_env):
    routes, _ = fake_http
    routes[("POST", "queue.fal.run/fal-ai/flux-2-dev")] = FakeResp(
        {"request_id": "r1"})
    assert NI.fal_action("generate", {"prompt": "x"})["request_id"] == "r1"
    routes[("GET", "/requests/r1/status")] = FakeResp({"status": "COMPLETED"})
    assert NI.fal_action("status", {"request_id": "r1"})["status"] == "COMPLETED"
    routes[("GET", "/requests/r1")] = FakeResp({"images": []})
    assert NI.fal_action("result", {"request_id": "r1"}) == {"images": []}


def test_req_lists_all_missing():
    with pytest.raises(ValueError, match="missing required arg"):
        NI._req({}, "a", "b")
    with pytest.raises(ValueError, match="a, b"):
        NI._req({"a": "  "}, "a", "b")


def test_integration_status_all_off(monkeypatch):
    for var in ("GITHUB_TOKEN", "GITLAB_TOKEN", "GOOGLE_OAUTH_CLIENT_ID",
                "GOOGLE_OAUTH_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN",
                "VERCEL_TOKEN", "FAL_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert NI.integration_status() == {
        "github": False, "gitlab": False, "gmail": False,
        "vercel": False, "fal": False}


def test_env_trims_whitespace(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "  tok  ")
    assert NI._env("GITHUB_TOKEN") == "tok"
    assert GitHub.available() is True
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert GitHub.available() is False
