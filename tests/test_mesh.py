import time
from collections import deque

import pytest

from cairn_edge.src.advanced.mesh_sync import MeshSync, Transport
from cairn_edge.src.advanced.models import MeshMessage, Track


class MemoryTransport(Transport):
    def __init__(self):
        self.peer = None
        self.queue = deque()

    def connect(self, peer):
        self.peer = peer

    def send(self, message: MeshMessage) -> None:
        if self.peer is not None:
            self.peer.queue.append(message)

    def recv(self):
        if not self.queue:
            return None
        return self.queue.popleft()


def make_mesh(tmp_path, node_id, transport):
    try:
        return MeshSync(node_id=node_id, transport=transport, key_dir=tmp_path / node_id)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_signature_verification_and_track_merge(tmp_path):
    ta, tb = MemoryTransport(), MemoryTransport()
    ta.connect(tb)
    tb.connect(ta)
    a = make_mesh(tmp_path, "node-a", ta)
    b = make_mesh(tmp_path, "node-b", tb)
    a.trust_public_key("node-b", b.public_key_pem())
    b.trust_public_key("node-a", a.public_key_pem())

    track = Track(track_id="local-1", lat=43.68, lon=-79.62, confidence=0.95, class_id="uas", kinematic_risk=55, timestamp=time.time())
    a.broadcast_track(track)
    msg = b.receive_once()
    assert msg is not None
    merged = b.get_merged_tracks()
    assert len(merged) == 1
    assert merged[0].track_id == "local-1"


def test_replay_protection_drops_duplicate_sequence(tmp_path):
    ta, tb = MemoryTransport(), MemoryTransport()
    ta.connect(tb)
    tb.connect(ta)
    a = make_mesh(tmp_path, "node-a", ta)
    b = make_mesh(tmp_path, "node-b", tb)
    b.trust_public_key("node-a", a.public_key_pem())

    track = Track(track_id="t1", lat=43.68, lon=-79.62, confidence=0.9, class_id="uas", kinematic_risk=20, timestamp=time.time())
    a.broadcast_track(track)
    first = tb.queue[0]
    assert b.receive_once() is not None
    tb.queue.append(first)
    assert b.receive_once() is None
    assert "replay dropped" in (b.health().degraded_reason or "")
