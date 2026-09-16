import "./SectionPlaceholder.css";

export default function SectionPlaceholder({ title, note }: { title: string; note?: string }) {
  return (
    <div className="section-placeholder">
      <div className="section-placeholder-title">{title}</div>
      {note && <div className="section-placeholder-note">{note}</div>}
    </div>
  );
}
