import { NavLink, Route, Routes } from "react-router-dom";
import Live from "./views/Live";
import Alerts from "./views/Alerts";
import IncidentReplay from "./views/IncidentReplay";
import Audit from "./views/Audit";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Maple Shield <span className="ms-dot">●</span> Operator
        </div>
        <nav className="nav">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>Live</NavLink>
          <NavLink to="/alerts" className={({ isActive }) => (isActive ? "active" : "")}>Alerts</NavLink>
          <NavLink to="/replay" className={({ isActive }) => (isActive ? "active" : "")}>Replay</NavLink>
          <NavLink to="/audit" className={({ isActive }) => (isActive ? "active" : "")}>Audit</NavLink>
        </nav>
        <div className="boundary">passive · no jamming · no engagement</div>
      </header>
      <main className="layout">
        <Routes>
          <Route path="/" element={<Live />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/replay" element={<IncidentReplay />} />
          <Route path="/replay/:incidentId" element={<IncidentReplay />} />
          <Route path="/audit" element={<Audit />} />
        </Routes>
        <p className="compliance-note">
          <b>Decision support only.</b> Maple Shield observes airspace and surfaces
          alerts to trained operators. It does not intercept, jam, neutralize,
          target, or otherwise interfere with drones. All operator actions are
          recorded in a hash-chained audit log.
        </p>
      </main>
    </div>
  );
}
