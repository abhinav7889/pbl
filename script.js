// script.js
(() => {
  const track = document.getElementById("track");
  const slides = Array.from(document.querySelectorAll(".slide"));
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const dotsWrap = document.getElementById("dots");
  const countText = document.getElementById("countText");

  let index = 0;

  function clamp(n, min, max) {
    return Math.max(min, Math.min(max, n));
  }

  function buildDots() {
    dotsWrap.innerHTML = "";
    slides.forEach((_, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "dot";
      b.setAttribute("aria-label", `Go to slide ${i + 1}`);
      b.addEventListener("click", () => goTo(i));
      dotsWrap.appendChild(b);
    });
  }

  function setActiveSlide(i) {
    slides.forEach((s, idx) => s.classList.toggle("is-active", idx === i));
    const dots = Array.from(dotsWrap.querySelectorAll(".dot"));
    dots.forEach((d, idx) => d.classList.toggle("is-active", idx === i));
  }

  function updateUI() {
    track.style.transform = `translateX(${-index * 100}vw)`;
    setActiveSlide(index);
    countText.textContent = `${index + 1} / ${slides.length}`;

    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === slides.length - 1;

    prevBtn.style.opacity = prevBtn.disabled ? "0.45" : "1";
    nextBtn.style.opacity = nextBtn.disabled ? "0.45" : "1";
  }

  function goTo(i) {
    index = clamp(i, 0, slides.length - 1);
    updateUI();
  }

  function next() {
    goTo(index + 1);
  }
  function prev() {
    goTo(index - 1);
  }

  // Keyboard nav
  window.addEventListener(
    "keydown",
    (e) => {
      // Prevent browser scroll / back navigation via arrows in some contexts
      if (["ArrowLeft", "ArrowRight"].includes(e.key)) e.preventDefault();

      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") prev();
      if (e.key === "Home") goTo(0);
      if (e.key === "End") goTo(slides.length - 1);
    },
    { passive: false },
  );

  // Optional: swipe support (minimal)
  let touchStartX = null;
  window.addEventListener(
    "touchstart",
    (e) => {
      if (!e.touches || e.touches.length !== 1) return;
      touchStartX = e.touches[0].clientX;
    },
    { passive: true },
  );

  window.addEventListener(
    "touchend",
    (e) => {
      if (touchStartX === null) return;
      const endX =
        e.changedTouches && e.changedTouches[0]
          ? e.changedTouches[0].clientX
          : null;
      if (endX === null) return;

      const dx = endX - touchStartX;
      touchStartX = null;

      if (Math.abs(dx) < 40) return;
      if (dx < 0) next();
      else prev();
    },
    { passive: true },
  );

  // Buttons
  prevBtn.addEventListener("click", prev);
  nextBtn.addEventListener("click", next);

  // Initialize
  buildDots();
  updateUI();

  // Keep alignment consistent on resize (vw-based already, but this keeps active class state)
  window.addEventListener("resize", () => updateUI());
})();
