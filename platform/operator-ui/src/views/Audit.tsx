import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { AuditEntry } from "../types";

export default function Audit() {
  const [verified, setVerified] = useState<boolean | null>(null);
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [badSeq, setBadSeq] = useState<number | null>(null);

  useEffect(() => {
    api.audit().then((r) => {
      setVerified(r.verified);
      setEntries(r.entries);
      setBadSeq(r.first_bad_seq);
    }).catch(() => undefined);
  }, []);

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Audit log</h3>
        <span className={verified ? "audit-good" : "audit-bad"} style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }}>
          {verified === null ? "…" : verified ? "✓ chain verified" : `✗ chain broken at seq ${badSeq}`}
        </span>
      </div>
      <div className="audit-row header">
        <div>seq</div><div>ts</div><div>operator · action</div><div>target</div>
      </div>
      {entries.length === 0 ? <div className="empty-state">no entries</div> : null}
      {entries.map((e) => (
        <div key={e.seq} className="audit-row">
          <div>{e.seq}</div>
          <div>{e.ts}</div>
          <div>{e.operator_id} · {e.action}</div>
          <div>{e.target ?? "—"}</div>
        </div>
      ))}
    </div>
  );
}
