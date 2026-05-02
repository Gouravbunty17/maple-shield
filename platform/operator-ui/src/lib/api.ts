import type { Alert, AuditEntry, CairnHealth, Incident, IncidentStatus } from "../types";

const BASE = "/api";

async function j<T>(p: Promise<Response>): Promise<T> {
  const r = await p;
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const api = {
  health: () => j<{ status: string; lawful_use_ack: boolean }>(fetch(`${BASE}/healthz`)),
  cairnHealth: () => j<CairnHealth>(fetch(`${BASE}/cairn/health`)),
  listAlerts: (severity?: string) =>
    j<Alert[]>(fetch(`${BASE}/alerts${severity ? `?severity=${severity}` : ""}`)),
  listIncidents: () => j<Incident[]>(fetch(`${BASE}/incidents`)),
  getIncident: (id: string) => j<Incident>(fetch(`${BASE}/incidents/${id}`)),
  setIncidentStatus: (id: string, status: IncidentStatus, operator_id: string) =>
    j<Incident>(fetch(`${BASE}/incidents/${id}/status`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ status, operator_id }),
    })),
  addNote: (id: string, operator_id: string, text: string) =>
    j<Incident>(fetch(`${BASE}/incidents/${id}/notes`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ operator_id, text }),
    })),
  exportIncident: (id: string) =>
    j<{ incident: Incident; alerts: Alert[]; summary_md: string }>(
      fetch(`${BASE}/incidents/${id}/export`)
    ),
  replay: (id: string) =>
    j<{ incident: Incident; frames: { ts: string; alert_id: string; severity: string; snapshot_b64?: string | null }[] }>(
      fetch(`${BASE}/replay/${id}`)
    ),
  audit: () =>
    j<{ verified: boolean; first_bad_seq: number | null; entries: AuditEntry[] }>(
      fetch(`${BASE}/audit`)
    ),
};

export function openAlertSocket(onAlert: (a: Alert) => void): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/alerts`);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.kind === "alert" && msg.alert) onAlert(msg.alert as Alert);
    } catch {/* ignore malformed frames */}
  };
  return ws;
}
