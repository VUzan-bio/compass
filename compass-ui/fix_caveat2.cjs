const fs = require('fs');
let code = fs.readFileSync('src/App.jsx', 'utf8');
const lines = code.split('\n');

// Find the InSilicoCaveat component boundaries
const startIdx = lines.findIndex(l => l.includes('const InSilicoCaveat = ()'));
if (startIdx < 0) { console.log('InSilicoCaveat not found'); process.exit(1); }

// Find the closing of the component (next const or line with just "};")
let endIdx = -1;
for (let i = startIdx + 1; i < lines.length; i++) {
  if (lines[i].trim() === '};' && lines[i-1].trim().includes('</div>')) {
    endIdx = i;
    break;
  }
}
if (endIdx < 0) { console.log('End of InSilicoCaveat not found'); process.exit(1); }

console.log(`InSilicoCaveat: lines ${startIdx+1} to ${endIdx+1}`);

// Extract the body content (paragraphs) - they're between the open div and closing div
const bodyStart = lines.findIndex((l, i) => i > startIdx && l.includes('<p style='));
const bodyEnd = lines.findIndex((l, i) => i > bodyStart + 2 && l.trim() === '</div>');

// Build the new component
const bodyLines = lines.slice(bodyStart, bodyEnd + 1);

const newComponent = `const InSilicoCaveat = () => {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginBottom: "20px" }}>
      <button onClick={() => setOpen(!open)} style={{ display: "flex", alignItems: "center", gap: "6px", background: "none", border: "none", cursor: "pointer", padding: "4px 0", marginBottom: open ? "2px" : 0, fontFamily: FONT, fontSize: "11px", fontWeight: 600, color: "#92400E", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        <ChevronDown size={12} style={{ transform: open ? "rotate(0deg)" : "rotate(-90deg)", transition: "transform 0.15s", color: "#D97706" }} />
        <AlertTriangle size={12} color="#D97706" strokeWidth={2} />
        In silico prediction: experimental validation required
      </button>
      {open && (
        <div style={{ background: "#FFFBEB", border: "1px solid #F59E0B33", borderRadius: "4px", padding: "12px 16px", fontSize: "11px", color: "#92400E", lineHeight: 1.7 }}>
${bodyLines.join('\n')}
      )}
    </div>
  );
};`;

// Replace the old component
const before = lines.slice(0, startIdx);
const after = lines.slice(endIdx + 1);
const result = [...before, newComponent, ...after].join('\n');

fs.writeFileSync('src/App.jsx', result);
console.log('InSilicoCaveat replaced successfully');
