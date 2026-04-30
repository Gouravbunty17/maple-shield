from __future__ import annotations

import abc
import base64
import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
except Exception:  # pragma: no cover
    Ed25519PrivateKey = None  # type: ignore
    Ed25519PublicKey = None  # type: ignore
    serialization = None  # type: ignore

from .models import HealthStatus, MeshMessage, Track

DEFAULT_KEY_DIR = Path("/etc/cairn/keys")


def _json_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _message_signing_payload(message: MeshMessage) -> bytes:
    return _json_bytes({"source_node": message.source_node, "sequence_number": message.sequence_number, "timestamp": message.timestamp, "payload": message.payload})


class Transport(abc.ABC):
    @abc.abstractmethod
    def send(self, message: MeshMessage) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def recv(self) -> Optional[MeshMessage]:
        raise NotImplementedError


class MulticastTransport(Transport):
    def __init__(self, group: str = "239.1.2.3", port: int = 5005, ttl: int = 2, bind_host: str = "0.0.0.0") -> None:
        self.group = group
        self.port = int(port)
        self.addr = (group, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(False)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        self.sock.bind((bind_host, self.port))
        mreq = struct.pack("4sl", socket.inet_aton(group), socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    def send(self, message: MeshMessage) -> None:
        self.sock.sendto(_json_bytes(message.as_payload()), self.addr)

    def recv(self) -> Optional[MeshMessage]:
        try:
            data, _ = self.sock.recvfrom(65535)
        except BlockingIOError:
            return None
        return MeshMessage(**json.loads(data.decode("utf-8")))


class UnicastGossipTransport(Transport):
    def __init__(self, peers: Iterable[Tuple[str, int]], bind_host: str = "0.0.0.0", bind_port: int = 5006, sync_interval_s: float = 2.0) -> None:
        self.peers = [(host, int(port)) for host, port in peers]
        self.sync_interval_s = float(sync_interval_s)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(False)
        self.sock.bind((bind_host, int(bind_port)))

    def send(self, message: MeshMessage) -> None:
        data = _json_bytes(message.as_payload())
        for peer in self.peers:
            self.sock.sendto(data, peer)

    def recv(self) -> Optional[MeshMessage]:
        try:
            data, _ = self.sock.recvfrom(65535)
        except BlockingIOError:
            return None
        return MeshMessage(**json.loads(data.decode("utf-8")))


class MeshSync:
    def __init__(self, node_id: str, transport: Transport, key_dir: str | Path = DEFAULT_KEY_DIR, heartbeat_interval_s: float = 5.0, peer_timeout_s: float = 15.0) -> None:
        if Ed25519PrivateKey is None or serialization is None:
            raise RuntimeError("cryptography package with Ed25519 support is required")
        self.node_id = node_id
        self.transport = transport
        self.key_dir = Path(key_dir)
        self.heartbeat_interval_s = heartbeat_interval_s
        self.peer_timeout_s = peer_timeout_s
        self._sequence = 0
        self._last_seen_sequence: Dict[str, int] = {}
        self._trusted_public_keys: Dict[str, Ed25519PublicKey] = {}
        self._tracks: Dict[str, Track] = {}
        self._peer_heartbeat: Dict[str, float] = {}
        self._last_heartbeat_sent = 0.0
        self._last_health = time.time()
        self._degraded_reason: Optional[str] = None
        self._private_key = self._load_or_create_private_key()
        self.trust_public_key(self.node_id, self.public_key_pem())

    def _load_or_create_private_key(self) -> Ed25519PrivateKey:
        self.key_dir.mkdir(parents=True, exist_ok=True)
        private_path = self.key_dir / f"{self.node_id}.ed25519.key"
        public_path = self.key_dir / f"{self.node_id}.ed25519.pub"
        if private_path.exists():
            return serialization.load_pem_private_key(private_path.read_bytes(), password=None)  # type: ignore[no-any-return]
        key = Ed25519PrivateKey.generate()
        private_path.write_bytes(key.private_bytes(encoding=serialization.Encoding.PEM, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption()))
        try:
            os.chmod(private_path, 0o600)
        except OSError:
            pass
        public_path.write_bytes(key.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo))
        return key

    def public_key_pem(self) -> str:
        return self._private_key.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo).decode("utf-8")

    def trust_public_key(self, node_id: str, public_key_pem: str) -> None:
        key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        self._trusted_public_keys[node_id] = key  # type: ignore[assignment]

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _sign(self, message: MeshMessage) -> MeshMessage:
        message.signature = base64.b64encode(self._private_key.sign(_message_signing_payload(message))).decode("ascii")
        return message

    def _verify(self, message: MeshMessage) -> bool:
        public_key = self._trusted_public_keys.get(message.source_node)
        if public_key is None:
            self._degraded_reason = f"untrusted source dropped: {message.source_node}"
            return False
        last_seen = self._last_seen_sequence.get(message.source_node, -1)
        if message.sequence_number <= last_seen:
            self._degraded_reason = f"replay dropped from {message.source_node}: seq={message.sequence_number} <= {last_seen}"
            return False
        try:
            public_key.verify(base64.b64decode(message.signature.encode("ascii")), _message_signing_payload(message))
        except Exception as exc:
            self._degraded_reason = f"signature verification failed: {exc}"
            return False
        self._last_seen_sequence[message.source_node] = message.sequence_number
        return True

    def broadcast_track(self, track: Track) -> MeshMessage:
        message = MeshMessage(type="TrackUpdate", source_node=self.node_id, sequence_number=self._next_sequence(), timestamp=time.time(), signature="", payload={"track": track.as_payload()})
        signed = self._sign(message)
        self.transport.send(signed)
        return signed

    def broadcast_heartbeat(self, force: bool = False) -> Optional[MeshMessage]:
        now = time.time()
        if not force and now - self._last_heartbeat_sent < self.heartbeat_interval_s:
            return None
        message = MeshMessage(type="Heartbeat", source_node=self.node_id, sequence_number=self._next_sequence(), timestamp=now, signature="", payload={"node_id": self.node_id, "status": "ok"})
        signed = self._sign(message)
        self.transport.send(signed)
        self._last_heartbeat_sent = now
        return signed

    def receive_once(self) -> Optional[MeshMessage]:
        message = self.transport.recv()
        if message is None:
            self._update_peer_loss_state()
            return None
        if message.source_node == self.node_id:
            return None
        if not self._verify(message):
            return None
        if message.type == "Heartbeat":
            self._peer_heartbeat[message.source_node] = message.timestamp
        elif message.type == "TrackUpdate":
            track_payload = message.payload.get("track")
            if isinstance(track_payload, dict):
                track = Track(**track_payload)
                current = self._tracks.get(track.track_id)
                if current is None or track.timestamp > current.timestamp:
                    self._tracks[track.track_id] = track
        self._last_health = time.time()
        self._update_peer_loss_state()
        return message

    def get_merged_tracks(self, max_age_s: float = 5.0) -> List[Track]:
        cutoff = time.time() - max_age_s
        return [track for track in self._tracks.values() if track.timestamp >= cutoff]

    def _update_peer_loss_state(self) -> None:
        now = time.time()
        lost = [peer for peer, ts in self._peer_heartbeat.items() if now - ts > self.peer_timeout_s]
        if lost:
            self._degraded_reason = "mesh peer lost: " + ", ".join(sorted(lost))

    def health(self) -> HealthStatus:
        self._update_peer_loss_state()
        return HealthStatus(module_name="mesh_sync", status="degraded" if self._degraded_reason else "ok", last_heartbeat=self._last_health, degraded_reason=self._degraded_reason)
