"""Off Linux the platform already blocks every method: scan names no GPU on top
(core040 R-6)."""
import multibench as mtb
from multibench.engine import envs



def test_no_gpu_sentence_off_linux(monkeypatch, capsys):
    monkeypatch.setattr(envs, "host_system", lambda: "macOS")
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    table = mtb.scan("D11", "vertical", methods=["moETM"])
    assert "NVIDIA GPU" not in table["reason"].iloc[0]
    assert "NVIDIA GPU" not in capsys.readouterr().out


def test_the_gpu_sentence_stays_on_a_linux_host_without_one(monkeypatch):
    monkeypatch.setattr(envs, "host_has_gpu", lambda: False)
    table = mtb.scan("D11", "vertical", methods=["moETM"], verbose=False)
    assert "NVIDIA GPU" in table["reason"].iloc[0]
