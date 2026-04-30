import base64
import json

import pytest

from scripts.continual_learning_offload import package_update
from scripts.update_loader import SignedUpdateLoader


@pytest.fixture()
def deployment_keys(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    priv_path = tmp_path / "deploy.key"
    pub_path = tmp_path / "deploy.pub"
    priv_path.write_bytes(private.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()))
    pub_path.write_bytes(public.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv_path, pub_path


def test_signed_update_installs_atomically(tmp_path, deployment_keys):
    priv, pub = deployment_keys
    engine = tmp_path / "candidate.engine"
    engine.write_bytes(b"engine-v1")
    package = package_update(engine, priv, "1.0.0", {"accepted": True, "baseline_map": 0.7, "candidate_map": 0.69}, tmp_path / "updates" / "pkg.json")
    loader = SignedUpdateLoader(tmp_path / "updates", pub, tmp_path / "models" / "current.engine", tmp_path / "models" / "current.version")
    assert loader.install_package(package) is True
    assert (tmp_path / "models" / "current.engine").read_bytes() == b"engine-v1"
    assert (tmp_path / "models" / "current.version").read_text(encoding="utf-8") == "1.0.0"


def test_tampered_engine_rejected(tmp_path, deployment_keys):
    priv, pub = deployment_keys
    engine = tmp_path / "candidate.engine"
    engine.write_bytes(b"engine-v1")
    package_path = package_update(engine, priv, "1.0.0", {"accepted": True}, tmp_path / "updates" / "pkg.json")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["engine_b64"] = base64.b64encode(b"tampered").decode("ascii")
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = SignedUpdateLoader(tmp_path / "updates", pub, tmp_path / "models" / "current.engine", tmp_path / "models" / "current.version")
    with pytest.raises(Exception):
        loader.install_package(package_path)
    assert not (tmp_path / "models" / "current.engine").exists()


def test_version_guard_rejects_old_package(tmp_path, deployment_keys):
    priv, pub = deployment_keys
    engine = tmp_path / "candidate.engine"
    engine.write_bytes(b"engine-v1")
    package = package_update(engine, priv, "1.0.0", {"accepted": True}, tmp_path / "updates" / "pkg.json")
    version_path = tmp_path / "models" / "current.version"
    version_path.parent.mkdir(parents=True)
    version_path.write_text("2.0.0", encoding="utf-8")
    loader = SignedUpdateLoader(tmp_path / "updates", pub, tmp_path / "models" / "current.engine", version_path)
    assert loader.install_package(package) is False
