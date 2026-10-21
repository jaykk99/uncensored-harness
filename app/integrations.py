"""Native integrations — direct REST APIs, no Composio middleman.

Each integration below talks to the provider's own API with httpx and an
env-var credential. When the credential is missing the integration reports
itself unavailable with an honest message instead of failing obscurely.

Integrations (env vars):
  GitHub  GITHUB_TOKEN            — repos, issues, PRs, Actions dispatch
  GitLab  GITLAB_TOKEN            — projects, issues, pipelines
            GITLAB_HOST (default https://gitlab.com)
  Gmail   GOOGLE_OAUTH_CLIENT_ID + GOOGLE_OAUTH_CLIENT_SECRET
            + GOOGLE_REFRESH_TOKEN — send, search, read (OAuth refresh flow)
  Vercel  VERCEL_TOKEN            — projects, deployments, redeploy, env vars
            VERCEL_TEAM_ID (optional)
  Fal     FAL_KEY                 — image generation (queue API)
            FAL_IMAGE_MODEL (default fal-ai/flux-2-dev)

Google OAuth note: this needs Jay's one-time Google connect — the refresh
token comes from the standard OAuth consent flow (offline access) and is
stored as GOOGLE_REFRESH_TOKEN. Nothing here fabricates or bypasses auth.

Guards: these tools act on Jay's own accounts only. Sending mail or
dispatching workflows is gated by the same message-approval flow as
everything else in the harness (the agent shows the action first).
"""

from __future__ import annotations

import base64
import logging
import os
import time
from email.message import EmailMessage
from typing import Any

import httpx

from .models import _client, RETRYABLE_STATUS

logger = logging.getLogger(__name__)

_API_TIMEOUT = 30


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _raise_for(r: httpx.Response, what: str) -> None:
    if r.status_code in RETRYABLE_STATUS:
        raise RuntimeError(f"{what}: HTTP {r.status_code} (transient — retry)")
    if r.status_code >= 400:
        raise RuntimeError(f"{what}: HTTP {r.status_code}: {r.text[:300]}")


# ===========================================================================
# GitHub — direct REST (docs.github.com/rest)
# ===========================================================================

class GitHub:
    """Jay's GitHub via the REST API. Token: GITHUB_TOKEN."""

    base = "https://api.github.com"

    @staticmethod
    def available() -> bool:
        return bool(_env("GITHUB_TOKEN"))

    @staticmethod
    def _headers() -> dict:
        return {
            "Authorization": f"Bearer {_env('GITHUB_TOKEN')}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "uncensored-harness/1.0",
        }

    @classmethod
    def _get(cls, path: str, params: dict | None = None) -> Any:
        with _client(_API_TIMEOUT) as c:
            r = c.get(cls.base + path, headers=cls._headers(), params=params)
        _raise_for(r, f"github GET {path}")
        return r.json()

    @classmethod
    def _post(cls, path: str, body: dict) -> Any:
        with _client(_API_TIMEOUT) as c:
            r = c.post(cls.base + path, headers=cls._headers(), json=body)
        _raise_for(r, f"github POST {path}")
        return r.json() if r.text else {}

    # -- repos -----------------------------------------------------------
    @classmethod
    def repos(cls, limit: int = 30) -> list[dict]:
        """List Jay's repos (newest first)."""
        data = cls._get("/user/repos",
                        {"per_page": max(1, min(limit, 100)),
                         "sort": "updated", "direction": "desc"})
        return [{"name": r["full_name"], "private": r["private"],
                 "description": r.get("description"),
                 "updated": r.get("updated_at"),
                 "url": r.get("html_url")} for r in data]

    @classmethod
    def repo(cls, owner_repo: str) -> dict:
        r = cls._get(f"/repos/{owner_repo}")
        return {"name": r["full_name"], "private": r["private"],
                "description": r.get("description"),
                "default_branch": r.get("default_branch"),
                "stars": r.get("stargazers_count"), "forks": r.get("forks_count"),
                "open_issues": r.get("open_issues_count"),
                "url": r.get("html_url")}

    @classmethod
    def file(cls, owner_repo: str, path: str, ref: str = "") -> dict:
        params = {"ref": ref} if ref else None
        r = cls._get(f"/repos/{owner_repo}/contents/{path.lstrip('/')}",
                     params)
        content = r.get("content", "")
        if r.get("encoding") == "base64":
            import base64 as _b
            content = _b.b64decode(content).decode("utf-8", "replace")
        return {"path": r.get("path"), "sha": r.get("sha"),
                "size": r.get("size"), "content": content[:20000]}

    # -- issues ----------------------------------------------------------
    @classmethod
    def issues(cls, owner_repo: str, state: str = "open",
               limit: int = 20) -> list[dict]:
        data = cls._get(f"/repos/{owner_repo}/issues",
                        {"state": state, "per_page": max(1, min(limit, 100))})
        return [{"number": i["number"], "title": i["title"],
                 "state": i["state"], "labels": [l["name"] for l in i["labels"]],
                 "url": i["html_url"], "is_pr": "pull_request" in i}
                for i in data]

    @classmethod
    def create_issue(cls, owner_repo: str, title: str,
                     body: str = "", labels: list[str] | None = None) -> dict:
        if not title.strip():
            raise ValueError("issue title is required")
        r = cls._post(f"/repos/{owner_repo}/issues",
                      {"title": title, "body": body,
                       "labels": labels or []})
        return {"number": r["number"], "url": r["html_url"]}

    # -- pull requests ---------------------------------------------------
    @classmethod
    def prs(cls, owner_repo: str, state: str = "open",
            limit: int = 20) -> list[dict]:
        data = cls._get(f"/repos/{owner_repo}/pulls",
                        {"state": state, "per_page": max(1, min(limit, 100))})
        return [{"number": p["number"], "title": p["title"],
                 "state": p["state"], "head": p["head"]["ref"],
                 "base": p["base"]["ref"], "draft": p["draft"],
                 "url": p["html_url"]} for p in data]

    # -- actions ---------------------------------------------------------
    @classmethod
    def workflow_runs(cls, owner_repo: str, workflow: str = "",
                      limit: int = 10) -> list[dict]:
        path = (f"/repos/{owner_repo}/actions/workflows/{workflow}/runs"
                if workflow else f"/repos/{owner_repo}/actions/runs")
        data = cls._get(path, {"per_page": max(1, min(limit, 100))})
        return [{"id": w["id"], "name": w["name"],
                 "status": w["status"], "conclusion": w.get("conclusion"),
                 "branch": w["head_branch"], "event": w["event"],
                 "created": w["created_at"], "url": w["html_url"]}
                for w in data.get("workflow_runs", [])]

    @classmethod
    def dispatch(cls, owner_repo: str, workflow: str, ref: str = "main",
                 inputs: dict | None = None) -> dict:
        """Trigger a workflow_dispatch event (honest 404 when the workflow
        has no workflow_dispatch trigger — GitHub rejects it)."""
        cls._post(f"/repos/{owner_repo}/actions/workflows/{workflow}/dispatches",
                  {"ref": ref, "inputs": inputs or {}})
        return {"ok": True, "workflow": workflow, "ref": ref}


# ===========================================================================
# GitLab — direct REST (docs.gitlab.com/api)
# ===========================================================================

class GitLab:
    """Jay's GitLab via the REST API. Token: GITLAB_TOKEN."""

    @staticmethod
    def available() -> bool:
        return bool(_env("GITLAB_TOKEN"))

    @classmethod
    def _base(cls) -> str:
        return _env("GITLAB_HOST") or "https://gitlab.com"

    @classmethod
    def _headers(cls) -> dict:
        return {"PRIVATE-TOKEN": _env("GITLAB_TOKEN"),
                "User-Agent": "uncensored-harness/1.0"}

    @classmethod
    def _get(cls, path: str, params: dict | None = None) -> Any:
        with _client(_API_TIMEOUT) as c:
            r = c.get(cls._base() + "/api/v4" + path,
                      headers=cls._headers(), params=params)
        _raise_for(r, f"gitlab GET {path}")
        return r.json()

    @classmethod
    def projects(cls, search: str = "", limit: int = 20) -> list[dict]:
        params: dict = {"per_page": max(1, min(limit, 100)),
                        "order_by": "last_activity_at"}
        if search:
            params["search"] = search
        data = cls._get("/projects", params)
        return [{"id": p["id"], "name": p["path_with_namespace"],
                 "visibility": p["visibility"],
                 "last_activity": p.get("last_activity_at"),
                 "url": p.get("web_url")} for p in data]

    @classmethod
    def project(cls, project_id: str | int) -> dict:
        p = cls._get(f"/projects/{project_id}")
        return {"id": p["id"], "name": p["path_with_namespace"],
                "description": p.get("description"),
                "default_branch": p.get("default_branch"),
                "stars": p.get("star_count"), "forks": p.get("forks_count"),
                "open_issues": p.get("open_issues_count"),
                "url": p.get("web_url")}

    @classmethod
    def issues(cls, project_id: str | int, state: str = "opened",
               limit: int = 20) -> list[dict]:
        data = cls._get(f"/projects/{project_id}/issues",
                        {"state": state, "per_page": max(1, min(limit, 100))})
        return [{"iid": i["iid"], "title": i["title"], "state": i["state"],
                 "labels": i.get("labels", []), "url": i.get("web_url")}
                for i in data]

    @classmethod
    def pipelines(cls, project_id: str | int,
                  limit: int = 10) -> list[dict]:
        data = cls._get(f"/projects/{project_id}/pipelines",
                        {"per_page": max(1, min(limit, 100))})
        return [{"id": p["id"], "status": p["status"], "ref": p["ref"],
                 "sha": (p.get("sha") or "")[:8],
                 "created": p.get("created_at"), "url": p.get("web_url")}
                for p in data]


# ===========================================================================
# Gmail — Google OAuth (needs Jay's one-time connect)
# ===========================================================================

class Gmail:
    """Jay's Gmail via the Gmail REST API.

    Auth: standard OAuth2 refresh flow. Needs GOOGLE_OAUTH_CLIENT_ID,
    GOOGLE_OAUTH_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN (from Jay's
    one-time consent with offline access). No token = honest "not
    connected" instead of a confusing 401.
    """

    _token_cache: dict = {}

    @staticmethod
    def available() -> bool:
        return bool(_env("GOOGLE_OAUTH_CLIENT_ID")
                    and _env("GOOGLE_OAUTH_CLIENT_SECRET")
                    and _env("GOOGLE_REFRESH_TOKEN"))

    @classmethod
    def _access_token(cls) -> str:
        cached = cls._token_cache.get("token", "")
        if cached and cls._token_cache.get("exp", 0) > time.time() + 60:
            return cached
        with _client(_API_TIMEOUT) as c:
            r = c.post("https://oauth2.googleapis.com/token", data={
                "client_id": _env("GOOGLE_OAUTH_CLIENT_ID"),
                "client_secret": _env("GOOGLE_OAUTH_CLIENT_SECRET"),
                "refresh_token": _env("GOOGLE_REFRESH_TOKEN"),
                "grant_type": "refresh_token",
            })
        if r.status_code >= 400:
            raise RuntimeError(
                "gmail oauth refresh failed "
                f"(HTTP {r.status_code}) — Jay needs to re-run the "
                "one-time Google connect to mint a fresh refresh token")
        data = r.json()
        cls._token_cache = {"token": data["access_token"],
                            "exp": time.time() + data.get("expires_in", 3600)}
        return data["access_token"]

    @classmethod
    def _headers(cls) -> dict:
        return {"Authorization": f"Bearer {cls._access_token()}",
                "User-Agent": "uncensored-harness/1.0"}

    @classmethod
    def send(cls, to: str, subject: str, body: str) -> dict:
        """Send a plain-text email from Jay's Gmail. (Approval-gated by the
        harness's message-approval flow — the agent shows the draft first.)"""
        if not to or "@" not in to:
            raise ValueError("valid 'to' address required")
        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject or "(no subject)"
        msg.set_content(body or "")
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        with _client(_API_TIMEOUT) as c:
            r = c.post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                       headers=cls._headers(), json={"raw": raw})
        _raise_for(r, "gmail send")
        return {"id": r.json().get("id"), "to": to}

    @classmethod
    def search(cls, query: str, limit: int = 10) -> list[dict]:
        """Search messages (Gmail query syntax). Returns id/thread/snippet."""
        with _client(_API_TIMEOUT) as c:
            r = c.get("https://gmail.googleapis.com/gmail/v1/users/me/messages",
                      headers=cls._headers(),
                      params={"q": query, "maxResults": max(1, min(limit, 50))})
        _raise_for(r, "gmail search")
        out = []
        for m in r.json().get("messages", []):
            out.append({"id": m["id"], "thread": m.get("threadId"),
                        "snippet": ""})
        # Fill snippets (one extra call each — bounded by limit).
        for entry in out:
            try:
                full = cls.get(entry["id"], fmt="metadata")
                entry["snippet"] = full.get("snippet", "")
                entry["subject"] = full.get("subject", "")
                entry["from"] = full.get("from", "")
            except Exception:
                pass
        return out

    @classmethod
    def get(cls, message_id: str, fmt: str = "full") -> dict:
        with _client(_API_TIMEOUT) as c:
            r = c.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
                      headers=cls._headers(),
                      params={"format": "metadata" if fmt == "metadata" else "full",
                              "metadataHeaders": ["Subject", "From", "Date"]})
        _raise_for(r, "gmail get")
        data = r.json()
        headers = {h["name"].lower(): h["value"]
                   for h in data.get("payload", {}).get("headers", [])}
        body = ""
        if fmt == "full":
            body = cls._extract_body(data.get("payload", {}))[:20000]
        return {"id": data["id"], "thread": data.get("threadId"),
                "snippet": data.get("snippet", ""),
                "subject": headers.get("subject", ""),
                "from": headers.get("from", ""),
                "date": headers.get("date", ""), "body": body}

    @staticmethod
    def _extract_body(payload: dict) -> str:
        """Walk MIME parts for the first text/plain (fallback text/html)."""
        html = ""
        stack = [payload]
        while stack:
            part = stack.pop()
            mime = part.get("mimeType", "")
            data = (part.get("body") or {}).get("data", "")
            if data and mime == "text/plain":
                return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
            if data and mime == "text/html" and not html:
                html = base64.urlsafe_b64decode(data).decode("utf-8", "replace")
            stack.extend(part.get("parts", []))
        if html:
            import re as _re
            return _re.sub(r"<[^>]+>", " ", html)
        return ""


# ===========================================================================
# Vercel — direct REST (vercel.com/docs/rest-api)
# ===========================================================================

class Vercel:
    """Jay's Vercel via the REST API. Token: VERCEL_TOKEN (+ VERCEL_TEAM_ID)."""

    base = "https://api.vercel.com"

    @staticmethod
    def available() -> bool:
        return bool(_env("VERCEL_TOKEN"))

    @classmethod
    def _headers(cls) -> dict:
        return {"Authorization": f"Bearer {_env('VERCEL_TOKEN')}",
                "User-Agent": "uncensored-harness/1.0"}

    @classmethod
    def _params(cls, extra: dict | None = None) -> dict:
        p = dict(extra or {})
        team = _env("VERCEL_TEAM_ID")
        if team:
            p["teamId"] = team
        return p

    @classmethod
    def _get(cls, path: str, params: dict | None = None) -> Any:
        with _client(_API_TIMEOUT) as c:
            r = c.get(cls.base + path, headers=cls._headers(),
                      params=cls._params(params))
        _raise_for(r, f"vercel GET {path}")
        return r.json()

    @classmethod
    def _post(cls, path: str, body: dict) -> Any:
        with _client(_API_TIMEOUT) as c:
            r = c.post(cls.base + path, headers=cls._headers(),
                       params=cls._params(), json=body)
        _raise_for(r, f"vercel POST {path}")
        return r.json() if r.text else {}

    @classmethod
    def projects(cls, limit: int = 20) -> list[dict]:
        data = cls._get("/v9/projects", {"limit": max(1, min(limit, 100))})
        return [{"id": p["id"], "name": p["name"],
                 "framework": p.get("framework"),
                 "updated": p.get("updatedAt"),
                 "url": f"https://{p['name']}.vercel.app"}
                for p in data.get("projects", [])]

    @classmethod
    def deployments(cls, project: str = "", limit: int = 10) -> list[dict]:
        params: dict = {"limit": max(1, min(limit, 100))}
        if project:
            # Resolve name -> id for the filter.
            for p in cls.projects(limit=100):
                if p["name"] == project:
                    params["projectId"] = p["id"]
                    break
        data = cls._get("/v6/deployments", params)
        return [{"id": d["uid"], "state": d["state"], "url": d.get("url"),
                 "created": d.get("createdAt"),
                 "meta": (d.get("meta") or {}).get("githubCommitMessage", "")}
                for d in data.get("deployments", [])]

    @classmethod
    def redeploy(cls, project: str, ref: str = "main") -> dict:
        """Create a fresh deployment from the project's linked git repo."""
        target = None
        for p in cls.projects(limit=100):
            if p["name"] == project:
                target = p
                break
        if target is None:
            raise ValueError(f"no such vercel project: {project}")
        detail = cls._get(f"/v9/projects/{target['id']}")
        link = (detail.get("link") or {})
        if link.get("type") != "github" or not link.get("repoId"):
            raise RuntimeError(
                f"project {project} has no linked GitHub repo — "
                "connect it in the Vercel dashboard first")
        data = cls._post("/v13/deployments", {
            "name": project,
            "gitSource": {"type": "github",
                          "repoId": link["repoId"], "ref": ref},
        })
        return {"id": data.get("id"), "url": data.get("url"),
                "state": data.get("readyState")}

    @classmethod
    def env_vars(cls, project: str) -> list[dict]:
        target_id = None
        for p in cls.projects(limit=100):
            if p["name"] == project:
                target_id = p["id"]
                break
        if target_id is None:
            raise ValueError(f"no such vercel project: {project}")
        data = cls._get(f"/v9/projects/{target_id}/env")
        # Values are never returned decrypted — keys/targets only.
        return [{"key": e["key"], "target": e.get("target"),
                 "id": e["id"]} for e in data.get("envs", [])]

    @classmethod
    def set_env(cls, project: str, key: str, value: str,
                target: str = "production") -> dict:
        target_id = None
        for p in cls.projects(limit=100):
            if p["name"] == project:
                target_id = p["id"]
                break
        if target_id is None:
            raise ValueError(f"no such vercel project: {project}")
        # Upsert: delete existing same-key/target first (Vercel 400s on dup).
        for e in cls.env_vars(project):
            if e["key"] == key and target in (e["target"] or []):
                with _client(_API_TIMEOUT) as c:
                    c.delete(f"{cls.base}/v9/projects/{target_id}/env/{e['id']}",
                             headers=cls._headers(), params=cls._params())
        data = cls._post(f"/v10/projects/{target_id}/env",
                         {"key": key, "value": value, "type": "encrypted",
                          "target": [target]})
        return {"key": key, "target": target, "id": data.get("id")}


# ===========================================================================
# Fal — image generation queue API (docs.fal.ai)
# ===========================================================================

class Fal:
    """Fal.ai image generation via the queue API. Key: FAL_KEY."""

    @staticmethod
    def available() -> bool:
        return bool(_env("FAL_KEY"))

    @classmethod
    def _model(cls) -> str:
        return _env("FAL_IMAGE_MODEL") or "fal-ai/flux-2-dev"

    @classmethod
    def _headers(cls) -> dict:
        return {"Authorization": f"Key {_env('FAL_KEY')}",
                "Content-Type": "application/json",
                "User-Agent": "uncensored-harness/1.0"}

    @classmethod
    def generate(cls, prompt: str, image_size: str = "landscape_16_9",
                 num_images: int = 1) -> dict:
        """Submit an image job. Returns a request_id to poll with status()."""
        if not prompt.strip():
            raise ValueError("prompt is required")
        url = f"https://queue.fal.run/{cls._model()}"
        with _client(60) as c:
            r = c.post(url, headers=cls._headers(),
                       json={"prompt": prompt, "image_size": image_size,
                             "num_images": max(1, min(num_images, 4))})
        _raise_for(r, "fal submit")
        return {"request_id": r.json().get("request_id"), "model": cls._model()}

    @classmethod
    def status(cls, request_id: str) -> dict:
        url = f"https://queue.fal.run/{cls._model()}/requests/{request_id}/status"
        with _client(_API_TIMEOUT) as c:
            r = c.get(url, headers=cls._headers())
        _raise_for(r, "fal status")
        return r.json()

    @classmethod
    def result(cls, request_id: str, wait_s: int = 120) -> dict:
        """Poll until COMPLETED (or timeout), then return image URLs."""
        deadline = time.time() + max(5, wait_s)
        last: dict = {}
        while time.time() < deadline:
            last = cls.status(request_id)
            if last.get("status") in ("COMPLETED", "FAILED"):
                break
            time.sleep(3)
        if last.get("status") != "COMPLETED":
            raise RuntimeError(
                f"fal job {request_id} not done after {wait_s}s "
                f"(status={last.get('status')})")
        url = f"https://queue.fal.run/{cls._model()}/requests/{request_id}"
        with _client(_API_TIMEOUT) as c:
            r = c.get(url, headers=cls._headers())
        _raise_for(r, "fal result")
        images = r.json().get("images", [])
        return {"images": [{"url": i.get("url"),
                            "width": i.get("width"),
                            "height": i.get("height")} for i in images]}


# ===========================================================================
# Registry + status for /api/health
# ===========================================================================

INTEGRATIONS: dict[str, Any] = {
    "github": GitHub,
    "gitlab": GitLab,
    "gmail": Gmail,
    "vercel": Vercel,
    "fal": Fal,
}


def integration_status() -> dict:
    """Which native integrations are live (credential present)."""
    return {name: cls.available() for name, cls in INTEGRATIONS.items()}


# Action routers: one tool per integration, "action" picks the method.
# Each entry: action -> (callable, required_args, description).

def _req(args: dict, *names: str) -> None:
    missing = [n for n in names if not str(args.get(n, "")).strip()]
    if missing:
        raise ValueError(f"missing required arg(s): {', '.join(missing)}")


def github_action(action: str, args: dict) -> Any:
    if not GitHub.available():
        return "GitHub is not connected — set GITHUB_TOKEN."
    a = (action or "").lower()
    if a == "repos":
        return GitHub.repos(int(args.get("limit", 30) or 30))
    if a == "repo":
        _req(args, "owner_repo"); return GitHub.repo(args["owner_repo"])
    if a == "file":
        _req(args, "owner_repo", "path")
        return GitHub.file(args["owner_repo"], args["path"], args.get("ref", ""))
    if a == "issues":
        _req(args, "owner_repo")
        return GitHub.issues(args["owner_repo"], args.get("state", "open"),
                             int(args.get("limit", 20) or 20))
    if a == "create_issue":
        _req(args, "owner_repo", "title")
        return GitHub.create_issue(args["owner_repo"], args["title"],
                                   args.get("body", ""),
                                   args.get("labels") or [])
    if a == "prs":
        _req(args, "owner_repo")
        return GitHub.prs(args["owner_repo"], args.get("state", "open"),
                          int(args.get("limit", 20) or 20))
    if a == "workflow_runs":
        _req(args, "owner_repo")
        return GitHub.workflow_runs(args["owner_repo"], args.get("workflow", ""),
                                    int(args.get("limit", 10) or 10))
    if a == "dispatch":
        _req(args, "owner_repo", "workflow")
        return GitHub.dispatch(args["owner_repo"], args["workflow"],
                               args.get("ref", "main") or "main",
                               args.get("inputs") or {})
    return (f"unknown github action: {action}. Available: repos, repo, file, "
            "issues, create_issue, prs, workflow_runs, dispatch")


def gitlab_action(action: str, args: dict) -> Any:
    if not GitLab.available():
        return "GitLab is not connected — set GITLAB_TOKEN."
    a = (action or "").lower()
    if a == "projects":
        return GitLab.projects(args.get("search", ""),
                               int(args.get("limit", 20) or 20))
    if a == "project":
        _req(args, "project_id"); return GitLab.project(args["project_id"])
    if a == "issues":
        _req(args, "project_id")
        return GitLab.issues(args["project_id"], args.get("state", "opened"),
                             int(args.get("limit", 20) or 20))
    if a == "pipelines":
        _req(args, "project_id")
        return GitLab.pipelines(args["project_id"],
                                int(args.get("limit", 10) or 10))
    return ("unknown gitlab action: " + str(action) +
            ". Available: projects, project, issues, pipelines")


def gmail_action(action: str, args: dict) -> Any:
    if not Gmail.available():
        return ("Gmail is not connected — Jay needs the one-time Google "
                "connect (GOOGLE_OAUTH_CLIENT_ID/SECRET + GOOGLE_REFRESH_TOKEN).")
    a = (action or "").lower()
    if a == "send":
        _req(args, "to", "subject")
        # NOTE: the harness's message-approval gate applies — the agent must
        # show Jay the draft and get his OK before calling this.
        return Gmail.send(args["to"], args["subject"], args.get("body", ""))
    if a == "search":
        _req(args, "query")
        return Gmail.search(args["query"], int(args.get("limit", 10) or 10))
    if a == "get":
        _req(args, "message_id"); return Gmail.get(args["message_id"])
    return ("unknown gmail action: " + str(action) +
            ". Available: send, search, get")


def vercel_action(action: str, args: dict) -> Any:
    if not Vercel.available():
        return "Vercel is not connected — set VERCEL_TOKEN."
    a = (action or "").lower()
    if a == "projects":
        return Vercel.projects(int(args.get("limit", 20) or 20))
    if a == "deployments":
        return Vercel.deployments(args.get("project", ""),
                                  int(args.get("limit", 10) or 10))
    if a == "redeploy":
        _req(args, "project")
        return Vercel.redeploy(args["project"], args.get("ref", "main") or "main")
    if a == "env_vars":
        _req(args, "project"); return Vercel.env_vars(args["project"])
    if a == "set_env":
        _req(args, "project", "key", "value")
        return Vercel.set_env(args["project"], args["key"], args["value"],
                              args.get("target", "production") or "production")
    return ("unknown vercel action: " + str(action) +
            ". Available: projects, deployments, redeploy, env_vars, set_env")


def fal_action(action: str, args: dict) -> Any:
    if not Fal.available():
        return "Fal is not connected — set FAL_KEY."
    a = (action or "").lower()
    if a == "generate":
        _req(args, "prompt")
        # generate() only submits; result() polls. Two-step so the agent
        # can do other work while the GPU job runs.
        return Fal.generate(args["prompt"],
                            args.get("image_size", "landscape_16_9"),
                            int(args.get("num_images", 1) or 1))
    if a == "status":
        _req(args, "request_id"); return Fal.status(args["request_id"])
    if a == "result":
        _req(args, "request_id")
        return Fal.result(args["request_id"],
                          int(args.get("wait_s", 120) or 120))
    return ("unknown fal action: " + str(action) +
            ". Available: generate, status, result")
