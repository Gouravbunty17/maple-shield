import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CairnHealth as CairnHealthStatus } from "../types";

export default function CairnHealth() {
  const [health, setHealth] = useState<CairnHealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.cairnHealth()
      .then((h) => {
        if (!cancelled) {
          setHealth(h);
          setError(null);
        }
      })
      .catch(() => {
        if (!cancelled) setError("engine unreachable");
      });
    return () => { cancelled = true; };
  }, []);

  const statusText = error
    ? error
    : health
      ? `${health.engine} ${health.engine_version} - ${health.frames_processed} frames`
      : "checking engine";

  return (
    <button
      type="button"
      className={`cairn-health ${error ? "warn" : ""}`}
      onClick={() => setOpen((v) => !v)}
      role="status"
      aria-expanded={open}
    >
      <span className="cairn-dot" />
      <span>{statusText}</span>
      {open && health ? (
        <span className="cairn-details">
          runtime {health.runtime_s.toFixed(1)}s - adapter {health.compatible ? "compatible" : "version check"}
        </span>
      ) : null}
    </button>
  );
}
