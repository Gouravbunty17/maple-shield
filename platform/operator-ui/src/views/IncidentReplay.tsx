import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Alert, Incident, IncidentStatus } from "../types";

const STATUSES: IncidentStatus[] = ["new", "acknowledged", "reviewed", "closed"];
const OPERATOR = "op-01"; // in production this comes from auth

export default function IncidentReplay() {
  const { incidentId } = useParams();
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [active, setActive] = useState<Incident | null>(null);
  const [frames, setFrames] = useState<{ ts: string; alert_id: string; severity: string }[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [note, setNote] = useState("");
  const [exportMd, setExportMd] = useState<string | null>(null);

  const refreshList = () => { api.listIncidents().then(setIncidents).catch(() => undefined); };
  useEffect(() => { refreshList(); }, []);

  useEffect(() => {
    if (!incidentId) { setActive(null); return; }
    api.getIncident(incidentId).then(setActive).catch(() => setActive(null));
    api.replay(incidentId).then((r) => setFrames(r.frames)).catch(() => setFrames([]));
    api.exportIncident(incidentId).then((r) => setAlerts(r.alerts)).catch(() => setAlerts([]));
  }, [incidentId]);

  const setStatus = async (s: IncidentStatus) => {
    if (!incidentId) return;
    await api.setIncidentStatus(incidentId, s, OPERATOR);
    api.getIncident(incidentId).then(setActive).catch(() => undefined);
    refreshList();
  };

  const addNote = async () => {
    if (!incidentId || !note.trim()) return;
    await api.addNote(incidentId, OPERATOR, note.trim());
    setNote("");
    api.getIncident(incidentId).then(setActive).catch(() => undefined);
  };

  const doExport = async () => {
    if (!incidentId) return;
    const r = await api.exportIncident(incidentId);
    setExportMd(r.summary_md);
  };

  return (
    <div className="cards">
      <section className="card">
        <h3>Incident replay</h3>
        {!active ? (
          <div className="empty-state">
            {incidents.length === 0
              ? "no incidents yet"
              : "select an incident:"}
            <ul style={{ marginTop: 8, paddingLeft: 0, listStyle: "none" }}>
              {incidents.map((i) => (
                <li key={i.incident_id} style={{ padding: "4px 0" }}>
                  <a href={`/replay/${i.incident_id}`}>{i.incident_id}</a>
                  <span style={{ marginLeft: 8, color: "#5f7a72" }}>· {i.status}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="detail">
            <div className="kvs">
              <b>incident</b><span style={{ fontFamily: "IBM Plex Mono, monospace" }}>{active.incident_id}</span>
              <b>status</b><span>{active.status}</span>
              <b>created</b><span>{active.created_ts}</span>
              <b>alerts</b><span>{active.alert_ids.length}</span>
            </div>
            <div className="btnrow">
              {STATUSES.map((s) => (
                <button key={s} className={`btn ${active.status === s ? "primary" : ""}`}
                        onClick={() => setStatus(s)}>{s}</button>
              ))}
              <button className="btn" onClick={doExport}>export summary</button>
            </div>
            <h3 style={{ marginTop: 8 }}>Notes</h3>
            <ul style={{ paddingLeft: 16 }}>
              {active.notes.map((n, i) => (
                <li key={i} style={{ fontSize: 13, marginBottom: 4 }}>
                  <span style={{ color: "#5f7a72" }}>{n.ts}</span> · <b>{n.operator_id}</b>: {n.text}
                </li>
              ))}
            </ul>
            <div style={{ display: "flex", gap: 8 }}>
              <input className="btn" style={{ flex: 1 }} value={note} onChange={(e) => setNote(e.target.value)} placeholder="add operator note" />
              <button className="btn primary" onClick={addNote}>add</button>
            </div>
            {exportMd ? (
              <pre style={{ background: "#06080a", padding: 12, borderRadius: 8, fontSize: 12, overflow: "auto", maxHeight: 200 }}>
                {exportMd}
              </pre>
            ) : null}
          </div>
        )}
      </section>

      <section className="card">
        <h3>Timeline</h3>
        <div className="replay-frames">
          {frames.length === 0 ? <div className="empty-state">no frames</div> : null}
          {frames.map((f, idx) => (
            <div key={idx} className="replay-frame">
              <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 11, color: "#5f7a72" }}>{f.ts}</div>
              <div style={{ marginTop: 4 }}>
                <span className={`sev ${f.severity}`}>{f.severity.toUpperCase()}</span>
              </div>
              <div style={{ marginTop: 8, fontSize: 12 }}>{f.alert_id}</div>
            </div>
          ))}
        </div>
        <h3 style={{ marginTop: 12 }}>Alerts in this incident</h3>
        <div className="alerts-list">
          {alerts.map((a) => (
            <div key={a.alert_id} className="alert-row">
              <span className={`sev ${a.severity}`}>{a.severity.toUpperCase()}</span>
              <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }}>
                {a.rule} · {a.track_id}
              </div>
              <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }}>{a.score.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
