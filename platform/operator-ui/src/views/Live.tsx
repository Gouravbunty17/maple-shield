import { useEffect, useMemo, useRef, useState } from "react";
import { api, openAlertSocket } from "../lib/api";
import type { Alert } from "../types";
import CairnHealth from "./CairnHealth";

interface BBoxOverlay {
  alertId: string;
  bbox: [number, number, number, number];
  imageSize: [number, number];
  severity: string;
  expires: number;
}

export default function Live() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [overlay, setOverlay] = useState<BBoxOverlay | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [lawfulAck, setLawfulAck] = useState<boolean>(false);

  // initial poll
  useEffect(() => {
    api.health()
      .then((h) => { setHealthy(true); setLawfulAck(!!h.lawful_use_ack); })
      .catch(() => setHealthy(false));
    api.listAlerts().then(setAlerts).catch(() => undefined);
  }, []);

  // websocket for live alerts
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      ws = openAlertSocket((a) => {
        setAlerts((prev) => [a, ...prev].slice(0, 200));
      });
    } catch { /* offline mode is fine */ }
    return () => ws?.close();
  }, []);

  const recent = alerts.slice(0, 50);

  return (
    <div className="cards">
      <section className="card">
        <h3>Live monitor — cam-01</h3>
        <CairnHealth />
        <div className="live-frame">
          <div className="empty">
            {healthy === false
              ? "command-api unreachable"
              : recent.length === 0
                ? "no detections in window"
                : "live frame source not connected (mock mode)"}
          </div>
          {overlay ? (
            <div
              className="bbox"
              style={{
                left: `${(overlay.bbox[0] / overlay.imageSize[0]) * 100}%`,
                top: `${(overlay.bbox[1] / overlay.imageSize[1]) * 100}%`,
                width: `${((overlay.bbox[2] - overlay.bbox[0]) / overlay.imageSize[0]) * 100}%`,
                height: `${((overlay.bbox[3] - overlay.bbox[1]) / overlay.imageSize[1]) * 100}%`,
              }}
            >
              <span className="bbox-label">drone · {overlay.severity}</span>
            </div>
          ) : null}
        </div>
        {!lawfulAck ? (
          <p className="empty-state">
            ⚠ MAPLE_SHIELD_LAWFUL_USE_ACK is not set. Confirm lawful deployment.
          </p>
        ) : null}
      </section>

      <section className="card">
        <h3>Recent alerts</h3>
        <div className="alerts-list">
          {recent.length === 0 ? <div className="empty-state">no alerts yet</div> : null}
          {recent.map((a) => (
            <div key={a.alert_id} className="alert-row">
              <span className={`sev ${a.severity}`}>{a.severity.toUpperCase()}</span>
              <div>
                <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12, color: "#a8b3b1" }}>
                  {a.rule} · {a.track_id}
                </div>
                <div style={{ fontSize: 12, color: "#5f7a72" }}>{a.ts}</div>
              </div>
              <div style={{ fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }}>
                {a.score.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
