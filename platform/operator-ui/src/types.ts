export type Severity = "info" | "low" | "med" | "high";
export type IncidentStatus = "new" | "acknowledged" | "reviewed" | "closed";

export interface Alert {
  alert_id: string;
  track_id: string;
  camera_id: string;
  severity: Severity;
  rule: string;
  score: number;
  ts: string;
  snapshot_b64?: string | null;
}

export interface IncidentNote {
  operator_id: string;
  ts: string;
  text: string;
}

export interface Incident {
  incident_id: string;
  status: IncidentStatus;
  alert_ids: string[];
  notes: IncidentNote[];
  created_ts: string;
  updated_ts: string;
}

export interface AuditEntry {
  seq: number;
  ts: string;
  operator_id: string;
  action: string;
  target?: string | null;
  payload: Record<string, unknown>;
  prev_hash: string;
  hash: string;
}

export interface CairnHealth {
  status: string;
  engine: string;
  engine_version: string;
  package_version: string;
  expected_adapter_version: string;
  compatible: boolean;
  frames_processed: number;
  runtime_s: number;
  started_ts: number;
  risk_config: Record<string, unknown>;
}
