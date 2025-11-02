def test_guess_from_extension(tmp_path):
    from docint.utils import mimetype

    file = tmp_path / "sample.txt"
    file.write_text("content")
    assert mimetype._guess_from_extension(file) == "text/plain"


def test_get_mimetype_falls_back(tmp_path, monkeypatch):
    from docint.utils import mimetype

    file = tmp_path / "binary.bin"
    file.write_bytes(b"\x00\x01")

    class DummyMagic:
        def from_file(self, _: str) -> str:
            raise RuntimeError("fail")

    monkeypatch.setattr(mimetype, "_MAGIC", DummyMagic())
    assert mimetype.get_mimetype(file) == "application/octet-stream"
