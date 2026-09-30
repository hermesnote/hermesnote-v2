// 2026-09-30 研究進度簡報：比照 /hermes 的整合方式，以 iframe 嵌入定稿 HTML（public/meeting-2026-09-30-slides.html）。
// ?embedded=1 讓簡報隱藏自己的預覽導覽列，改用網站導覽列；簡報依 iframe 可用空間自動縮放，保留四頁版面與換頁操作。
export default function Meeting20260930Page() {
  return (
    <iframe
      src="/meeting-2026-09-30-slides.html?embedded=1"
      title="研究進度 2026.09.30"
      // 載入後把焦點交給簡報，方向鍵／PageUp／PageDown 換頁直接可用
      onLoad={(e) => e.currentTarget.contentWindow?.focus()}
      style={{ display: "block", width: "100%", height: "100%", border: 0, flex: "1 1 auto" }}
    />
  );
}
