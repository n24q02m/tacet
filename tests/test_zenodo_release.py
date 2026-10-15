import hashlib
import importlib.util
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "zenodo_release.py"
spec = importlib.util.spec_from_file_location("zenodo_release", SCRIPT)
zr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zr)

PDF = b"%PDF-test-bytes"
MD5 = hashlib.md5(PDF).hexdigest()


def make_fake(*, draft_listing, published_version="v11"):
    state = {"latest": 100}
    calls = []

    def fake(method, url, token, body=None, raw=None):
        calls.append((method, url))
        api = zr.API
        if url == f"{api}/records/{zr.CONCEPT_RECORD}/versions/latest":
            return {"id": state["latest"]}
        if method == "GET" and url == f"{api}/deposit/depositions/100":
            return {
                "metadata": {"version": published_version},
                "files": [{"filename": "paper.pdf"}],
            }
        if method == "GET" and url.startswith(f"{api}/deposit/depositions?status=draft"):
            return draft_listing
        if method == "GET" and url == f"{api}/deposit/depositions/200":
            return {
                "id": 200,
                "links": {"bucket": "https://bucket.invalid/b"},
                "files": [{"filename": "paper.pdf"}],
            }
        if method == "POST" and url == f"{api}/deposit/depositions/100/actions/newversion":
            return {"links": {"latest_draft": f"{api}/deposit/depositions/200"}}
        if method == "PUT" and url == f"{api}/deposit/depositions/200":
            return {}
        if method == "PUT" and url.startswith("https://bucket.invalid/b/"):
            return None
        if method == "GET" and url == f"{api}/deposit/depositions/200/files":
            return [{"id": "f1", "filename": "paper.pdf", "checksum": f"md5:{MD5}"}]
        if method == "POST" and url == f"{api}/deposit/depositions/200/actions/publish":
            state["latest"] = 200
            return {"doi": "10.5281/zenodo.200", "conceptdoi": "10.5281/zenodo.20621240"}
        raise AssertionError(f"unexpected call {method} {url}")

    return fake, calls


OPEN_DRAFT = [
    {"id": 200, "conceptrecid": zr.CONCEPT_RECORD, "state": "unsubmitted", "submitted": False}
]


def run(monkeypatch, tmp_path, fake, *flags, version="v12"):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(PDF)
    monkeypatch.setenv("ZENODO_TOKEN", "test-token")
    monkeypatch.setattr(zr, "_call", fake)
    monkeypatch.setattr(
        sys, "argv", ["zenodo_release.py", "--version", version, "--pdf", str(pdf), *flags]
    )
    return zr.main()


def published(calls):
    return any(m == "POST" and u.endswith("/actions/publish") for m, u in calls)


def test_publish_without_open_draft_refuses_and_does_not_publish(monkeypatch, tmp_path):
    fake, calls = make_fake(draft_listing=[])
    assert run(monkeypatch, tmp_path, fake, "--publish") == 2
    assert not published(calls)


def test_publish_with_open_draft_finalizes(monkeypatch, tmp_path):
    fake, calls = make_fake(draft_listing=OPEN_DRAFT)
    assert run(monkeypatch, tmp_path, fake, "--publish") == 0
    assert published(calls)


def test_draft_mode_stops_before_publish(monkeypatch, tmp_path):
    fake, calls = make_fake(draft_listing=OPEN_DRAFT)
    assert run(monkeypatch, tmp_path, fake) == 0
    assert not published(calls)


def test_duplicate_version_label_is_rejected(monkeypatch, tmp_path):
    fake, calls = make_fake(draft_listing=[])
    with pytest.raises(zr.ZenodoError):
        run(monkeypatch, tmp_path, fake, "--publish", version="v11")
    assert not published(calls)


def test_listing_entry_already_done_is_not_an_open_draft(monkeypatch, tmp_path):
    done = [{"id": 200, "conceptrecid": zr.CONCEPT_RECORD, "state": "done", "submitted": True}]
    fake, calls = make_fake(draft_listing=done)
    assert run(monkeypatch, tmp_path, fake, "--publish") == 2
    assert not published(calls)
