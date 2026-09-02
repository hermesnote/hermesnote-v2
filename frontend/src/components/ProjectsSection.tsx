import { useEffect, useRef, useState } from "react";
import "./ProjectsSection.css";

const PROJECT_COUNT = 6;
const CARD_WIDTH_PCT = 46;
const GAP_PX = 24;
const LEFT_PEEK_PCT = 15;

// Real cards are 1..6. Clone the last at the front and the first at the end
// so the track can keep sliding in one direction forever; once a clone is
// reached we snap (no transition) back to the matching real slide.
const NUMBERS = [PROJECT_COUNT, ...Array.from({ length: PROJECT_COUNT }, (_, i) => i + 1), 1];

// 佔位文字，之後隨真正的專案內容替換
const CAPTIONS: Record<number, { title: string; desc: string }> = {
  1: { title: "專案標題先隨便放 01", desc: "這裡放這張圖的一句話說明，之後再換成真的內容。" },
  2: { title: "專案標題先隨便放 02", desc: "這裡放這張圖的一句話說明，之後再換成真的內容。" },
  3: { title: "專案標題先隨便放 03", desc: "這裡放這張圖的一句話說明，之後再換成真的內容。" },
  4: { title: "專案標題先隨便放 04", desc: "這裡放這張圖的一句話說明，之後再換成真的內容。" },
  5: { title: "專案標題先隨便放 05", desc: "這裡放這張圖的一句話說明，之後再換成真的內容。" },
  6: { title: "專案標題先隨便放 06", desc: "這裡放這張圖的一句話說明，之後再換成真的內容。" },
};

function ChevronLeft() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M15 18l-6-6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function ChevronRight() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 18l6-6-6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const LAST_INDEX = NUMBERS.length - 1;
// index must always stay inside [0, LAST_INDEX] — NUMBERS has no entries
// outside that range, so anything else would crash the render
const clampIndex = (i: number) => Math.max(0, Math.min(LAST_INDEX, i));

export default function ProjectsSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const el = sectionRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          io.disconnect();
        }
      },
      { threshold: 0.2 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // index 1 = real card "1" (index 0 is the cloned last card)
  const [index, setIndex] = useState(1);
  const [noTransition, setNoTransition] = useState(false);
  // buttons are disabled while true, so the browser itself refuses clicks
  // during a slide (or the boundary snap-back) — no manual click ever
  // races past the cloned array bounds
  const [isAnimating, setIsAnimating] = useState(false);

  const go = (dir: 1 | -1) => {
    if (isAnimating) return;
    setIsAnimating(true);
    setNoTransition(false);
    setIndex((i) => clampIndex(i + dir));
  };

  const handleTransitionEnd = (e: React.TransitionEvent) => {
    if (e.propertyName !== "transform") return;

    setIndex((current) => {
      if (current <= 0) {
        setNoTransition(true);
        return PROJECT_COUNT;
      }
      if (current >= LAST_INDEX) {
        setNoTransition(true);
        return 1;
      }
      setIsAnimating(false);
      return current;
    });
  };

  // After a no-transition snap, the browser can otherwise merge that jump
  // with the *next* animated step into one long slide (it looks like the
  // incoming card sits blank for a second while it slides in from far
  // away). Reading a layout property forces the browser to settle the
  // snap into its own frame before we unlock the buttons, so the next
  // click always animates a normal single-card distance.
  useEffect(() => {
    if (!noTransition) return;
    if (trackRef.current) {
      void trackRef.current.offsetWidth;
    }
    setIsAnimating(false);
  }, [noTransition]);

  const safeIndex = clampIndex(index);
  const trackTransform = `translateX(calc(${LEFT_PEEK_PCT}% - ${safeIndex} * (${CARD_WIDTH_PCT}% + ${GAP_PX}px)))`;
  const activeCaption = CAPTIONS[NUMBERS[safeIndex]] ?? CAPTIONS[1];

  return (
    <section
      ref={sectionRef}
      className={"projects" + (isVisible ? " is-visible" : "")}
    >
      <div className="projects-inner">
        <div className="projects-head">
          <span className="projects-eyebrow">Selected Work</span>
          <h2>過去專案</h2>
        </div>

        <div className="projects-track-wrap">
          <div
            ref={trackRef}
            className={"projects-track" + (noTransition ? " no-transition" : "")}
            style={{ transform: trackTransform }}
            onTransitionEnd={handleTransitionEnd}
          >
            {NUMBERS.map((n, i) => (
              <div className="project-card" key={i}>
                <div className="project-card-frame">
                  <span className="project-card-number">{n}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="project-caption">
          <span className="project-caption-title">{activeCaption.title}</span>
          <span className="project-caption-desc">{activeCaption.desc}</span>
        </div>

        <div className="projects-nav">
          <button
            className="projects-nav-btn"
            onClick={() => go(-1)}
            disabled={isAnimating}
            aria-label="上一個"
          >
            <ChevronLeft />
          </button>
          <button
            className="projects-nav-btn"
            onClick={() => go(1)}
            disabled={isAnimating}
            aria-label="下一個"
          >
            <ChevronRight />
          </button>
        </div>
      </div>
    </section>
  );
}
