import base64
import json
import time

import numpy as np
import pytest

from cairn_edge.src.advanced.continual_learning import HardExampleCollector, HardExampleCollectorConfig, OperatorSignature, SignatureVerifier
from cairn_edge.src.advanced.models import Detection


@pytest.fixture()
def operator_keys(tmp_path):
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    keys = {}
    for operator_id in ["op1", "op2"]:
        private = Ed25519PrivateKey.generate()
        public = private.public_key()
        (key_dir / f"{operator_id}.pub").write_bytes(public.public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo))
        keys[operator_id] = private
    return key_dir, keys


def sign(operator_id, label, detection_dict, private_key):
    msg = SignatureVerifier.message(operator_id, label, detection_dict)
    return base64.b64encode(private_key.sign(msg)).decode("ascii")


def make_collector(tmp_path, key_dir, max_examples=5000):
    cfg = HardExampleCollectorConfig(
        root_dir=str(tmp_path / "hard"),
        metadata_jsonl=str(tmp_path / "hard" / "metadata.jsonl"),
        public_keys_dir=str(key_dir),
        max_examples=max_examples,
        min_free_bytes=0,
        quorum_size=2,
    )
    return HardExampleCollector(cfg)


def test_quorum_logic_marks_ground_truth(tmp_path, operator_keys):
    key_dir, keys = operator_keys
    collector = make_collector(tmp_path, key_dir)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detection = Detection(bbox=(0, 0, 32, 32), confidence=0.9, class_id="uas")
    detection_dict = detection.serializable_dict()
    sigs = [
        OperatorSignature(operator_id="op1", label="uas", signature=sign("op1", "uas", detection_dict, keys["op1"])),
        OperatorSignature(operator_id="op2", label="uas", signature=sign("op2", "uas", detection_dict, keys["op2"])),
    ]
    example = collector._save_now(frame, detection, sigs)
    assert example.ground_truth is True
    collector.stop()


def test_bad_signature_is_rejected(tmp_path, operator_keys):
    key_dir, keys = operator_keys
    collector = make_collector(tmp_path, key_dir)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detection = Detection(bbox=(0, 0, 32, 32), confidence=0.9, class_id="uas")
    bad = OperatorSignature(operator_id="op1", label="uas", signature=base64.b64encode(b"bad").decode("ascii"))
    example = collector._save_now(frame, detection, [bad])
    assert example.ground_truth is False
    assert collector.security_events()
    collector.stop()


def test_ring_buffer_eviction(tmp_path, operator_keys):
    key_dir, _ = operator_keys
    collector = make_collector(tmp_path, key_dir, max_examples=2)
    for i in range(4):
        path = collector.images_dir / f"hard_{i}.jpg"
        path.write_bytes(b"x")
        time.sleep(0.002)
    collector._evict_fifo()
    assert len(list(collector.images_dir.glob("hard_*.jpg"))) == 2
    collector.stop()


def test_storage_full_blocks_save(tmp_path, operator_keys):
    key_dir, _ = operator_keys
    cfg = HardExampleCollectorConfig(
        root_dir=str(tmp_path / "hard"),
        metadata_jsonl=str(tmp_path / "hard" / "metadata.jsonl"),
        public_keys_dir=str(key_dir),
        min_free_bytes=10**18,
    )
    collector = HardExampleCollector(cfg)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detection = Detection(bbox=(0, 0, 32, 32), confidence=0.5, class_id="uas")
    assert collector.maybe_save(frame, detection) is False
    assert collector.health().status == "degraded"
    collector.stop()
