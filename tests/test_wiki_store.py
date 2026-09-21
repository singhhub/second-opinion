"""Tests for patient wiki file storage helpers (backend/wiki_store.py)."""

import pytest

from backend import wiki_store


def test_write_and_read_wiki_page_round_trip(tmp_path):
    patient_id = "patient-a"

    wiki_store.write_wiki_page(patient_id, "allergies.md", "# Allergies\n- Penicillin", root=tmp_path)
    content = wiki_store.read_wiki_page(patient_id, "allergies.md", root=tmp_path)

    assert content == "# Allergies\n- Penicillin"


def test_read_wiki_page_returns_none_when_missing(tmp_path):
    assert wiki_store.read_wiki_page("patient-a", "allergies.md", root=tmp_path) is None


def test_write_wiki_page_creates_parent_directories(tmp_path):
    wiki_store.write_wiki_page("patient-a", "by-system/cardiac.md", "# Cardiac", root=tmp_path)

    path = wiki_store.wiki_dir("patient-a", tmp_path) / "by-system" / "cardiac.md"
    assert path.exists()
    assert path.read_text() == "# Cardiac"


def test_page_path_traversal_is_rejected(tmp_path):
    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.write_wiki_page("patient-a", "../../etc/passwd", "malicious", root=tmp_path)


def test_page_path_cannot_reach_another_patients_files(tmp_path):
    wiki_store.write_wiki_page("patient-a", "allergies.md", "patient A data", root=tmp_path)

    with pytest.raises(wiki_store.UnsafePagePathError):
        wiki_store.read_wiki_page("patient-a", "../patient-b/allergies.md", root=tmp_path)


def test_list_wiki_pages_returns_relative_paths_sorted(tmp_path):
    wiki_store.write_wiki_page("patient-a", "medications.md", "meds", root=tmp_path)
    wiki_store.write_wiki_page("patient-a", "allergies.md", "allergies", root=tmp_path)
    wiki_store.write_wiki_page("patient-a", "by-system/cardiac.md", "cardiac", root=tmp_path)

    pages = wiki_store.list_wiki_pages("patient-a", root=tmp_path)

    assert pages == ["allergies.md", "by-system/cardiac.md", "medications.md"]


def test_write_raw_source_creates_file(tmp_path):
    path = wiki_store.write_raw_source("patient-a", "source-1", "extracted text", root=tmp_path)

    assert path.exists()
    assert path.read_text() == "extracted text"
    assert path.parent.name == "raw"


def test_patient_dirs_never_overlap_across_patients(tmp_path):
    assert wiki_store.patient_dir("patient-a", tmp_path) != wiki_store.patient_dir("patient-b", tmp_path)
    assert wiki_store.wiki_dir("patient-a", tmp_path) != wiki_store.wiki_dir("patient-b", tmp_path)
