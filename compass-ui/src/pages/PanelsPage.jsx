import React, { useState, useEffect, useCallback } from "react";
import {
  Check, ChevronDown, ChevronRight, Clock, Database, Download, ExternalLink,
  Eye, Folder, Layers, Loader2, Pencil, Plus, RefreshCw, Search, Trash2, X,
} from "lucide-react";
import { T, FONT, HEADING, MONO } from "../tokens";
import { useIsMobile } from "../hooks/useIsMobile";
import { Badge, Btn } from "../components/ui/index.jsx";
import { listPanels, createPanel, listJobs } from "../api";
import { MUTATIONS, ORGANISMS } from "../mockData";

const mutLabel = (m) => m.category === "gene_presence" ? m.gene : `${m.gene}_${m.ref}${m.pos}${m.alt}`;

const DEFAULT_PANELS = [
  {
    id: "mdr14",
    name: "MDR-TB 12-plex",
    organism: "mtb",
    description: "WHO-catalogued first- and second-line resistance panel. Covers 6 drug classes for comprehensive drug-susceptibility profiling.",
    mutations: MUTATIONS.map(mutLabel),
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "core5",
    name: "Core 5-plex (MTB)",
    organism: "mtb",
    description: "High-confidence tier-1 mutations only. Targets the most clinically actionable resistance determinants for rapid point-of-care screening.",
    mutations: ["rpoB_S531L", "katG_S315T", "fabG1_C-15T", "gyrA_D94G", "rrs_A1401G"],
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "rif",
    name: "Rifampicin Panel",
    organism: "mtb",
    description: "Focused panel for rifampicin mono-resistance detection. Covers the rpoB RRDR hotspot mutations conferring >95% of phenotypic RIF resistance.",
    mutations: ["rpoB_S531L", "rpoB_H526Y", "rpoB_D516V"],
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "ecoli_esbl",
    name: "E. coli ESBL/CRE Panel",
    organism: "ecoli",
    description: "ESBL + Enterobacterales carbapenemase detection. CTX-M, NDM, KPC, OXA-48, mcr-1, fluoroquinolone QRDR.",
    mutations: (ORGANISMS.find(o => o.id === "ecoli")?.mutations || []).map(mutLabel),
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "mrsa",
    name: "MRSA/VRSA Panel",
    organism: "saureus",
    description: "Methicillin and vancomycin resistance detection. mecA/C, vanA, fluoroquinolone, rifampicin, macrolide targets.",
    mutations: (ORGANISMS.find(o => o.id === "saureus")?.mutations || []).map(mutLabel),
    created_at: "2025-01-15T00:00:00Z",
  },
  {
    id: "ngono_amr",
    name: "N. gonorrhoeae AMR Panel",
    organism: "ngonorrhoeae",
    description: "ESC resistance (penA mosaic), azithromycin (23S rRNA + mtrR efflux), fluoroquinolone, tetracycline.",
    mutations: (ORGANISMS.find(o => o.id === "ngonorrhoeae")?.mutations || []).map(mutLabel),
    created_at: "2025-01-15T00:00:00Z",
  },
];

const PanelsPage = ({ connected }) => {
  const mobile = useIsMobile();
  const [panels, setPanels] = useState(DEFAULT_PANELS);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newMuts, setNewMuts] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editMuts, setEditMuts] = useState("");

  useEffect(() => {
    if (connected) {
      listPanels().then(({ data }) => { if (data) setPanels(data); });
    }
  }, [connected]);

  const handleCreate = async () => {
    const muts = newMuts.split(",").map((s) => s.trim()).filter(Boolean);
    if (connected) {
      const { data, error } = await createPanel(newName, newDesc, muts);
      if (data) setPanels((p) => [...p, data]);
    } else {
      setPanels((p) => [...p, { id: Date.now(), name: newName, description: newDesc, mutations: muts, created_at: new Date().toISOString() }]);
    }
    setShowNew(false);
    setNewName("");
    setNewDesc("");
    setNewMuts("");
  };

  const startEdit = (p) => {
    setEditingId(p.id || p.name);
    setEditName(p.name);
    setEditDesc(p.description || "");
    setEditMuts((p.mutations || []).join(", "));
  };

  const saveEdit = (panelId) => {
    const muts = editMuts.split(",").map((s) => s.trim()).filter(Boolean);
    setPanels((prev) => prev.map((p) => (p.id || p.name) === panelId ? { ...p, name: editName, description: editDesc, mutations: muts } : p));
    setEditingId(null);
  };

  const deletePanel = (panelId) => {
    setPanels((prev) => prev.filter((p) => (p.id || p.name) !== panelId));
  };

  return (
    <div style={{ padding: mobile ? "24px 16px" : "32px 40px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
        <div>
          <div style={{ fontSize: "11px", fontWeight: 600, color: T.primary, textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "8px" }}>Library</div>
          <h2 style={{ fontSize: "20px", fontWeight: 600, color: T.text, margin: 0, letterSpacing: "-0.02em", fontFamily: HEADING }}>Mutation Panels</h2>
        </div>
        <Btn icon={Plus} size="sm" onClick={() => setShowNew(!showNew)}>New Panel</Btn>
      </div>

      {showNew && (
        <div style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "24px", marginBottom: "24px" }}>
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Panel Name</label>
            <input value={newName} onChange={(e) => setNewName(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
          </div>
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Description</label>
            <input value={newDesc} onChange={(e) => setNewDesc(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
          </div>
          <div style={{ marginBottom: "16px" }}>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Mutations (comma-separated)</label>
            <textarea value={newMuts} onChange={(e) => setNewMuts(e.target.value)} rows={3} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: MONO, fontSize: "12px", outline: "none", resize: "vertical", boxSizing: "border-box" }} />
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            <Btn onClick={handleCreate} disabled={!newName.trim()} size="sm">Create</Btn>
            <Btn variant="ghost" size="sm" onClick={() => setShowNew(false)}>Cancel</Btn>
          </div>
        </div>
      )}

      {panels.length === 0 && !showNew && (
        <div style={{ textAlign: "center", padding: "64px 24px", color: T.textTer }}>
          <Layers size={40} strokeWidth={1} />
          <div style={{ fontSize: "14px", marginTop: "12px" }}>No panels yet. Create one to get started.</div>
        </div>
      )}

      <div style={{ display: "grid", gap: "12px" }}>
        {panels.map((p) => {
          const pid = p.id || p.name;
          const isEditing = editingId === pid;
          return (
            <div key={pid} style={{ background: T.bg, border: `1px solid ${T.border}`, borderRadius: "4px", padding: "20px" }}>
              {isEditing ? (
                <div>
                  <div style={{ marginBottom: "12px" }}>
                    <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Panel Name</label>
                    <input value={editName} onChange={(e) => setEditName(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ marginBottom: "12px" }}>
                    <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Description</label>
                    <input value={editDesc} onChange={(e) => setEditDesc(e.target.value)} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: FONT, fontSize: "13px", outline: "none", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ marginBottom: "16px" }}>
                    <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: T.textSec, marginBottom: "4px" }}>Mutations (comma-separated)</label>
                    <textarea value={editMuts} onChange={(e) => setEditMuts(e.target.value)} rows={3} style={{ width: "100%", padding: "8px 12px", borderRadius: "4px", border: `1px solid ${T.border}`, fontFamily: MONO, fontSize: "12px", outline: "none", resize: "vertical", boxSizing: "border-box" }} />
                  </div>
                  <div style={{ display: "flex", gap: "8px" }}>
                    <Btn onClick={() => saveEdit(pid)} disabled={!editName.trim()} size="sm">Save</Btn>
                    <Btn variant="ghost" size="sm" onClick={() => setEditingId(null)}>Cancel</Btn>
                  </div>
                </div>
              ) : (
                <>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <div style={{ fontSize: "14px", fontWeight: 600, color: T.text, fontFamily: HEADING }}>{p.name}</div>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <Badge variant="primary">{(p.mutations || []).length} mutations</Badge>
                      <button onClick={() => startEdit(p)} title="Edit panel" style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex", alignItems: "center", borderRadius: "4px" }}>
                        <Pencil size={14} color={T.textTer} />
                      </button>
                      <button onClick={() => deletePanel(pid)} title="Delete panel" style={{ background: "none", border: "none", cursor: "pointer", padding: "4px", display: "flex", alignItems: "center", borderRadius: "4px" }}>
                        <Trash2 size={14} color={T.textTer} />
                      </button>
                    </div>
                  </div>
                  {p.description && <div style={{ fontSize: "12px", color: T.textSec, marginBottom: "8px" }}>{p.description}</div>}
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                    {(p.mutations || []).slice(0, 10).map((m) => (
                      <span key={m} style={{ fontFamily: MONO, fontSize: "10px", padding: "3px 8px", borderRadius: "4px", background: T.bgSub, border: `1px solid ${T.borderLight}`, color: T.text }}>{m}</span>
                    ))}
                    {(p.mutations || []).length > 10 && <span style={{ fontSize: "10px", color: T.textTer, padding: "3px" }}>+{p.mutations.length - 10} more</span>}
                  </div>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};


export { DEFAULT_PANELS, PanelsPage };
