import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { Alert, Incident, Severity } from "../types";

const SEVS: Severity[] = ["info", "low", "med", "high"];

export default function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [filter, setFilter] = useState<Severity | "">("");

  const refresh = () => {
    api.listAlerts(filter || undefined).then(setAlerts).catch(() => undefined);
    api.listIncidents().then(setIncidents).catch(() => undefined);
  };
  useEffect(refresh, [filter]);

  const incidentByAlertId = new Map<string, Incident>();
  incidents.forEach((inc) => inc.alert_ids.forEach((aid) => incidentByAlertId.set(aid, inc)));

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>Alerts</h3>
        <div className="btnrow">
          <button className={`btn ${filter === "" ? "primary" : ""}`} onClick={() => setFilter("")}>all</button>
          {SEVS.map((s) => (
            <button key={s} className={`btn ${filter === s ? "primary" : ""}`} onClick={() => setFilter(s)}>{s}</button>
          ))}
        </div>
      </div>
      <div className="alerts-list" style={{ marginTop: 12 }}>
        {alerts.length === 0 ? <div className="empty-state">no alerts</div> : null}
        {alerts.map((a) => {
          const inc = incidentByAlertId.get(a.alert_id);
          return (
            <div key={a.alert_id} className="alert-row">
              <span className={`sev ${a.severity}`}>{a.severity.toUpperCase()}</span>
              <div>
                <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12, color: "#a8b3b1" }}>
                  {a.rule} · track {a.track_id} · cam {a.camera_id}
                </div>
                <div style={{ fontSize: 12, color: "#5f7a72" }}>{a.ts}</div>
              </div>
              {inc ? (
                <Link className="btn" to={`/replay/${inc.incident_id}`}>review →</Link>
              ) : (
                <span className="empty-state" style={{ fontSize: 12 }}>no incident</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
