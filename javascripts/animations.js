/* M3 docs — page polish JS.

   Two responsibilities:

   1. Scroll-triggered fade-up:
      - Content is visible by default (CSS).
      - JS finds animation candidates, tags below-fold ones with
        `.anim-pending` (which sets opacity:0 + translate via CSS),
        and uses IntersectionObserver to flip them to `.is-visible`
        as they enter the viewport.
      - If anything fails (JS disabled, iframe with innerHeight=0,
        broken observer), no element ever gets `.anim-pending` →
        content remains fully visible. Fail-open by design.

   2. Sticky header shadow on scroll. */

(function () {
  // With reduced motion only the landing card runs, and it draws a still frame.
  const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const ANIM_SELECTOR =
    '.md-typeset .grid.cards > ul > li, ' +
    '.md-typeset .grid:not(.cards) > *, ' +
    '.anim-fade-up';

  const initFadeUp = () => {
    const targets = document.querySelectorAll(ANIM_SELECTOR);
    if (!targets.length) return;
    if (!('IntersectionObserver' in window)) return;

    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    // If viewport is degenerate (0 or unknown), bail — keep content visible.
    if (vh < 100) return;

    // Tag below-the-fold targets as pending (CSS hides them); keep above-fold
    // content fully visible to avoid any flash of empty space.
    targets.forEach(el => {
      const r = el.getBoundingClientRect();
      // r.top here is relative to viewport. If element's top is more than ~50px
      // below the visible area, animate. Otherwise leave visible.
      if (r.top > vh - 40) {
        el.classList.add('anim-pending');
      }
    });

    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    }, {
      rootMargin: '0px 0px 80px 0px',   // pre-reveal 80px before entering
      threshold: 0.01,
    });

    targets.forEach(el => {
      if (el.classList.contains('anim-pending')) io.observe(el);
    });
  };

  const stickyHeader = () => {
    const header = document.querySelector('.md-header');
    if (!header) return;
    const setShadow = () => header.setAttribute('data-md-state', window.scrollY > 8 ? 'shadow' : '');
    setShadow();                        // refresh state on every (re-)init
    if (window.__stickyWired) return;   // but attach the scroll listener only ONCE
    window.__stickyWired = true;        // (instant-nav re-runs init; don't stack handlers)
    let ticking = false;
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(() => { setShadow(); ticking = false; }); ticking = true; }
    }, { passive: true });
  };

  /* Scroll progress bar (Bioconductor reference) — slim violet bar at top */
  const scrollProgress = () => {
    if (document.querySelector('.scroll-progress')) return;
    const bar = document.createElement('div');
    bar.className = 'scroll-progress';
    document.body.appendChild(bar);

    let ticking = false;
    const update = () => {
      const h = document.documentElement;
      const total = h.scrollHeight - h.clientHeight;
      const pct = total > 0 ? (window.scrollY / total) * 100 : 0;
      bar.style.width = pct + '%';
      bar.classList.toggle('is-scrolling', window.scrollY > 80);
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  };

  /* Back-to-top button — fastai-style, appears after 600px scroll */
  const backToTop = () => {
    if (document.querySelector('.back-to-top')) return;
    const btn = document.createElement('button');
    btn.className = 'back-to-top';
    btn.setAttribute('aria-label', 'Back to top');
    btn.innerHTML = '↑';
    btn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(btn);

    let ticking = false;
    const update = () => {
      btn.classList.toggle('is-visible', window.scrollY > 600);
      ticking = false;
    };
    window.addEventListener('scroll', () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  };

  const enableSectionNumbersOnHome = () => {
    const path = (window.location.pathname || '').replace(/\/+$/, '/');
    // A landing page is any page carrying the `.lp` wrapper (portal + every tool
    // home) — detect that so the hero styling is identical across all of them,
    // not just at the hardcoded `/m3/` path or the site root.
    const isHome = !!document.querySelector('.lp') || /\/m3\/$/.test(path) || path === '/' || /index\.html?$/i.test(path);
    if (isHome) document.body.classList.add('has-section-numbers');
    // Tutorial pages (notebooks) get step-numbering
    if (/\/notebooks\//.test(path)) {
      document.body.classList.add('has-tutorial-numbers');
    }
  };

  /* Map each H2 to an editorial eyebrow label for the section divider.
     The CSS ::before uses attr(data-eyebrow); fall-back is "Section". */
  const tagSectionEyebrows = () => {
    if (!document.body.classList.contains('has-section-numbers')) return;
    const map = {
      'how-it-works': 'How it works',
      'start-here':   'Start here',
      'why-m3':       'Why M3',
      'cite-m3':      'Citation',
    };
    document.querySelectorAll('.md-typeset h2[id]').forEach((h) => {
      const slug = h.id;
      if (map[slug]) h.setAttribute('data-eyebrow', map[slug]);
    });
  };

  /* Hero reveal sequence — Linear-style restrained orchestration.
     One reveal per browser session via sessionStorage gate.
     Sequence (total ~1200ms):
       t=0     hero eyebrow fades in
       t=100   wordmark word-by-word stagger
       t=600   tagline slides up
       t=750   CTA pair scales + fades in
       t=900   right column (illustration) fades in */
  const heroReveal = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!document.body.classList.contains('has-section-numbers')) return; // home only
    if (sessionStorage.getItem('m3-hero-seen') === '1') return;
    return;  // fade-free hero: render the title instantly, skip the opacity-0 letter-stagger

    const eyebrow = document.querySelector('.hero-eyebrow');
    const h1      = document.querySelector('.md-typeset > h1:first-of-type');
    const tagline = document.querySelector('.hero-tagline');
    const cta     = document.querySelector('.hero-cta');
    const badges  = document.querySelector('.hero-badges');
    const illus   = document.querySelector('.hero-illustration');

    const stage = (el, css) => {
      if (!el) return;
      Object.assign(el.style, css);
    };

    // Stage: initial state (hidden offset)
    stage(eyebrow, { opacity: '0', transform: 'translateY(6px)', transition: 'opacity 400ms ease, transform 400ms ease' });
    stage(tagline, { opacity: '0', transform: 'translateY(8px)', transition: 'opacity 400ms ease, transform 400ms ease' });
    stage(cta,     { opacity: '0', transform: 'scale(0.97)',       transition: 'opacity 400ms ease, transform 400ms ease' });
    stage(badges,  { opacity: '0',                                  transition: 'opacity 400ms ease' });
    stage(illus,   { opacity: '0', transform: 'translateY(8px)', transition: 'opacity 500ms ease, transform 500ms ease' });

    // Wordmark: split into letters for stagger
    let wordmarkSpans = [];
    if (h1) {
      const text = h1.textContent.trim();
      h1.textContent = '';
      [...text].forEach((ch, i) => {
        const sp = document.createElement('span');
        sp.textContent = ch === ' ' ? ' ' : ch;
        sp.style.opacity = '0';
        sp.style.display = 'inline-block';
        sp.style.transform = 'translateY(12px)';
        sp.style.transition = 'opacity 380ms cubic-bezier(0.22,1,0.36,1), transform 380ms cubic-bezier(0.22,1,0.36,1)';
        sp.style.transitionDelay = (100 + i * 35) + 'ms';
        h1.appendChild(sp);
        wordmarkSpans.push(sp);
      });
    }

    // Reveal sequence
    requestAnimationFrame(() => {
      setTimeout(() => stage(eyebrow, { opacity: '1', transform: 'translateY(0)' }), 0);
      wordmarkSpans.forEach(sp => {
        sp.style.opacity = '1';
        sp.style.transform = 'translateY(0)';
      });
      setTimeout(() => stage(tagline, { opacity: '1', transform: 'translateY(0)' }), 600);
      setTimeout(() => stage(cta,     { opacity: '1', transform: 'scale(1)' }),    750);
      setTimeout(() => stage(badges,  { opacity: '1' }),                          800);
      setTimeout(() => stage(illus,   { opacity: '1', transform: 'translateY(0)' }), 900);
    });

    sessionStorage.setItem('m3-hero-seen', '1');
  };

  /* === SWISS-WHITE v0.5 ADDITIONS === */

  /* Magnetic buttons — opt-in via .magnetic class only.
     (Get Started / .lp-cta intentionally excluded — plain button, no pull.) */
  const magneticButtons = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (window.matchMedia('(pointer: coarse)').matches) return; // touch devices
    const targets = document.querySelectorAll('.magnetic');
    targets.forEach((el) => {
      let raf = null;
      const STRENGTH = 0.18;   // 0..1, higher = more pull
      const RANGE   = 90;      // px of activation radius beyond button bounds
      el.addEventListener('mousemove', (e) => {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const r = el.getBoundingClientRect();
          const cx = r.left + r.width / 2;
          const cy = r.top + r.height / 2;
          const dx = e.clientX - cx;
          const dy = e.clientY - cy;
          const dist = Math.hypot(dx, dy);
          const max = Math.max(r.width, r.height) / 2 + RANGE;
          if (dist < max) {
            el.style.transform = `translate3d(${dx * STRENGTH}px, ${dy * STRENGTH}px, 0)`;
          }
        });
      });
      el.addEventListener('mouseleave', () => {
        if (raf) cancelAnimationFrame(raf);
        el.style.transform = '';
      });
    });
  };

  /* Section scroll-spy rail — left-fixed §1 §2 §3 navigator */
  const sectionRail = () => {
    if (!document.body.classList.contains('has-section-numbers')) return;
    if (window.innerWidth < 1280) return;
    const sections = document.querySelectorAll('.lp-hero, .lp-section');
    if (sections.length === 0) return;

    /* Build the rail */
    const rail = document.createElement('aside');
    rail.className = 'section-rail';
    rail.setAttribute('aria-label', 'Section navigation');
    sections.forEach((s, i) => {
      const item = document.createElement('a');
      item.className = 'section-rail__item';
      item.setAttribute('data-rail-idx', String(i));
      item.textContent = '§' + (i + 1);
      item.addEventListener('click', (e) => {
        e.preventDefault();
        s.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      rail.appendChild(item);
    });
    document.body.appendChild(rail);

    /* Activate on scroll via IntersectionObserver */
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const idx = Array.prototype.indexOf.call(sections, entry.target);
          rail.querySelectorAll('.section-rail__item').forEach((it, i) => {
            it.classList.toggle('is-active', i === idx);
          });
        }
      });
    }, { threshold: 0.3, rootMargin: '-30% 0px -30% 0px' });
    sections.forEach((s) => io.observe(s));
  };

  /* Number counter — count up to target on viewport entry */
  const numberCounters = () => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const els = document.querySelectorAll('.count-up');
    if (!els.length || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        const target = parseFloat(el.getAttribute('data-target') || el.textContent || '0');
        const duration = 450;
        const start = performance.now();
        const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);
        const step = (now) => {
          const t = Math.min(1, (now - start) / duration);
          const value = Math.round(target * easeOutQuart(t));
          el.textContent = value.toLocaleString();
          if (t < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
        io.unobserve(el);
      });
    }, { threshold: 0.6 });
    els.forEach((el) => io.observe(el));
  };

  /* GSAP ScrollTrigger — tasteful landing choreography (iter 49, "A 轻度增强").
     Loads only if GSAP global present + motion allowed + on a landing page.
     Fail-open: if GSAP missing, elements render at natural CSS state (visible). */
  const gsapLanding = () => {
    const gsap = window.gsap;
    const ScrollTrigger = window.ScrollTrigger;
    if (!gsap || !ScrollTrigger) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (!document.querySelector('.lp')) return;

    gsap.registerPlugin(ScrollTrigger);
    /* instant-nav safety: clear triggers from a previous page */
    ScrollTrigger.getAll().forEach((t) => t.kill());

    const OUT = 'power2.out';

    /* (a) Hero text — stagger children in on load */
    const heroBits = document.querySelectorAll(
      '.lp-hero-text .lp-eyebrow, .lp-hero-text > h1, .lp-hero-text .lp-sub,' +
      '.lp-hero-text .highlight, .lp-hero-text .lp-cta'
    );
    const introTweens = [];
    if (heroBits.length) {
      introTweens.push(gsap.from(heroBits, {
        y: 0, duration: 0.5, ease: OUT,
        stagger: 0.09, delay: 0.05, clearProps: 'all',
      }));
    }

    /* (b) Hero diagram card — fade up on load */
    const card = document.querySelector('.lp-hero-card');
    if (card) {
      introTweens.push(gsap.from(card, {
        y: 0, duration: 0.6, ease: OUT,
        delay: 0.25, clearProps: 'all',
      }));
    }

    /* Safety net (truly fail-open): an on-load from() holds the above-the-fold hero
       at opacity:0 until the rAF ticker advances it — but rAF is PAUSED in a
       backgrounded tab, so the hero could stay blank for a never-focused tab, a
       crawler, or a prerender/screenshot. setTimeout still fires in the background,
       so force any unfinished intro tween to its visible end-state. No-op once the
       (sub-second) entrance has already completed in a normal foreground load. */
    setTimeout(() => {
      introTweens.forEach((tw) => { if (tw && tw.progress() < 1) tw.progress(1); });
      const introEls = [...heroBits, card].filter(Boolean);
      if (introEls.length) gsap.set(introEls, { clearProps: 'all' });
    }, 1600);

    /* Only animate blocks that start BELOW the fold; anything already in the
       initial viewport (e.g. the short ecosystem landing) stays visible, so a
       flaky/late ScrollTrigger or a re-init can never strand it at opacity:0. */
    const belowFold = (el) => el.getBoundingClientRect().top > (window.innerHeight || 800) * 0.9;

    /* (c) Section eyebrows (FEATURES / TUTORIALS) — slide in from left on scroll */
    gsap.utils.toArray('.lp-section .lp-eyebrow').forEach((eb) => {
      if (!belowFold(eb)) return;
      gsap.from(eb, {
        scrollTrigger: { trigger: eb, start: 'top 88%' },
        x: -16, duration: 0.45, ease: OUT, clearProps: 'all',
      });
    });

    /* (d) Cards — staggered fade-up per grid on scroll */
    gsap.utils.toArray('.lp-grid-4').forEach((grid) => {
      const cards = grid.querySelectorAll('.lp-card');
      if (!cards.length) return;
      if (!belowFold(grid)) return;   // in view at setup → keep visible, don't hide
      gsap.from(cards, {
        scrollTrigger: { trigger: grid, start: 'top 85%' },
        y: 26, duration: 0.5, ease: OUT,
        stagger: 0.08, clearProps: 'all',
      });
    });

    /* (e) Feature columns — staggered fade-up on scroll */
    const featRow = document.querySelector('.lp-feature-row');
    if (featRow && belowFold(featRow)) {
      const feats = featRow.querySelectorAll('.lp-feat');
      if (feats.length) {
        gsap.from(feats, {
          scrollTrigger: { trigger: featRow, start: 'top 88%' },
          y: 22, duration: 0.5, ease: OUT,
          stagger: 0.08, clearProps: 'all',
        });
      }
    }

    /* (f) Footer bar — fade up on scroll */
    const footer = document.querySelector('.lp-footer-bar');
    if (footer && belowFold(footer)) {
      gsap.from(footer, {
        scrollTrigger: { trigger: footer, start: 'top 92%' },
        y: 16, duration: 0.5, ease: OUT, clearProps: 'all',
      });
    }

    ScrollTrigger.refresh();

    /* Fail-open safety net for the scroll-revealed blocks (cards / features /
       footer / eyebrows). gsap.from(...{scrollTrigger}) parks them at opacity:0
       until the trigger fires. On a SHORT landing they already sit in the initial
       viewport, so if ScrollTrigger doesn't fire on load (backgrounded tab,
       prerender, headless screenshot, or odd scroll metrics) they'd stay blank.
       After a beat, force any still-hidden, in-view block to its visible state.
       Below-the-fold blocks on a long page are untouched — they reveal on scroll
       as designed (a working foreground load makes this a no-op). */
    setTimeout(() => {
      const vh = window.innerHeight || 800;
      document.querySelectorAll(
        '.lp-card, .lp-feat, .lp-footer-bar, .lp-section .lp-eyebrow'
      ).forEach((el) => {
        const r = el.getBoundingClientRect();
        const inView = r.top < vh && r.bottom > 0;
        if (inView && parseFloat(getComputedStyle(el).opacity) < 0.5) {
          gsap.set(el, { clearProps: 'opacity,transform' });
          el.style.opacity = '';
          el.style.transform = '';
        }
      });
    }, 1700);
  };

  /* Disentanglement animation (iter 50) — the hero centrepiece.
     A cloud of mixed-colour dots (entangled latent z) sorts itself into
     three clean factor clusters (Cell type / Condition / Batch effect),
     then re-mixes. Loops. Represents M3's factorised embeddings. */
  let disentTl = null;
  const disentanglement = () => {
    const svg = document.querySelector('.lp-disent svg');
    if (!svg) return;
    const dotsG = svg.querySelector('.disent-dots');
    if (!dotsG) return;

    /* instant-nav safety: clear prior dots + timeline */
    if (disentTl) { disentTl.kill(); disentTl = null; }
    dotsG.innerHTML = '';

    const NS = 'http://www.w3.org/2000/svg';
    const lanes = [
      { x: 68,  color: '#7c3aed' },  // Cell type   — violet  (biological)
      { x: 178, color: '#e11d48' },  // Condition 1 — rose
      { x: 288, color: '#fb7185' },  // Condition 2 — salmon (condition family)
      { x: 398, color: '#64748b' },  // Batch effect — slate  (removed in correction)
    ];
    const PER = 5;                   // fewer dots
    const R = 6;                     // bigger dots
    const blobCx = 233, blobCy = 116, blobR = 52;
    const laneTop = 74, laneBot = 206;
    const rand = (a, b) => a + Math.random() * (b - a);

    const dots = [];
    lanes.forEach((lane, li) => {
      for (let i = 0; i < PER; i++) {
        const ang = Math.random() * Math.PI * 2;
        const rr = Math.sqrt(Math.random()) * blobR;
        const ex = blobCx + Math.cos(ang) * rr;
        const ey = blobCy + Math.sin(ang) * rr * 0.82;
        const dx = lane.x + rand(-14, 14);
        const dy = rand(laneTop, laneBot);
        const c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', ex.toFixed(1));
        c.setAttribute('cy', ey.toFixed(1));
        c.setAttribute('r', String(R));
        c.setAttribute('fill', lane.color);
        c.__ex = ex; c.__ey = ey; c.__dx = dx; c.__dy = dy; c.__lane = li;
        dotsG.appendChild(c);
        dots.push(c);
      }
    });
    const batchDots = dots.filter((d) => d.__lane === 3);

    const laneLabels = svg.querySelectorAll('.disent-label');
    const axes = svg.querySelectorAll('.disent-axis line');
    const zLabel = svg.querySelector('.disent-z');
    const batchNote = svg.querySelector('.disent-batch-note');
    const batchLabel = laneLabels[3];
    const batchAxis = axes[3];
    const gsap = window.gsap;

    /* Fallback: no GSAP or reduced motion → rest in disentangled state. */
    if (!gsap || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      dots.forEach((c) => { c.setAttribute('cx', c.__dx.toFixed(1)); c.setAttribute('cy', c.__dy.toFixed(1)); });
      laneLabels.forEach((l) => (l.style.opacity = 1));
      axes.forEach((l) => (l.style.opacity = 1));
      if (zLabel) zLabel.style.opacity = 0;
      return;
    }

    const E = 'power2.inOut';
    gsap.set(laneLabels, { opacity: 0 });
    gsap.set(axes, { opacity: 0, transformOrigin: 'center', scaleX: 0.4 });
    gsap.set(zLabel, { opacity: 1 });
    gsap.set(batchNote, { opacity: 0 });

    disentTl = gsap.timeline({ repeat: -1 });

    /* ── beat 1: hold entangled ── */
    disentTl.to({}, { duration: 0.9 });

    /* ── beat 2: DISENTANGLE into 4 factor clusters ── */
    disentTl.to(zLabel, { opacity: 0, duration: 0.3 }, 'sep');
    disentTl.to(dots, {
      duration: 1.5, ease: E, stagger: 0.015,
      attr: { cx: (i, el) => el.__dx, cy: (i, el) => el.__dy },
    }, 'sep');
    disentTl.to(axes, { opacity: 1, scaleX: 1, duration: 0.5, ease: E }, 'sep+=0.8');
    disentTl.to(laneLabels, { opacity: 1, duration: 0.4, stagger: 0.05 }, 'sep+=0.9');

    /* ── beat 3: hold disentangled ── */
    disentTl.to({}, { duration: 0.7 });

    /* ── beat 4: BATCH CORRECTION — batch cluster fades + drifts away ── */
    disentTl.to(batchDots, { opacity: 0.16, y: 18, duration: 0.6, ease: E }, 'rem');
    disentTl.to([batchLabel, batchAxis], { opacity: 0.28, duration: 0.4 }, 'rem');
    disentTl.to(batchNote, { opacity: 1, duration: 0.4 }, 'rem+=0.25');

    /* ── beat 5: hold corrected ── */
    disentTl.to({}, { duration: 1.0 });

    /* ── beat 6: RE-ENTANGLE (restore batch, merge all) ── */
    disentTl.to(batchNote, { opacity: 0, duration: 0.3 }, 'mer');
    disentTl.to(batchDots, { opacity: 1, y: 0, duration: 0.4, ease: E }, 'mer');
    disentTl.to([batchLabel, batchAxis], { opacity: 0, duration: 0.3 }, 'mer');
    disentTl.to(laneLabels, { opacity: 0, duration: 0.3 }, 'mer');
    disentTl.to(axes, { opacity: 0, scaleX: 0.4, duration: 0.3 }, 'mer');
    disentTl.to(dots, {
      duration: 1.4, ease: E, stagger: 0.015,
      attr: { cx: (i, el) => el.__ex, cy: (i, el) => el.__ey },
    }, 'mer');
    disentTl.to(zLabel, { opacity: 1, duration: 0.4 }, 'mer+=0.7');
  };

  /* ── shared SVG helpers for the task-flow ── */
  const TF_NS = 'http://www.w3.org/2000/svg';
  const tfMk = (t, a) => { const e = document.createElementNS(TF_NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const tfScene = () => { const e = tfMk('g', { class: 'tf-scene', opacity: 0 }); return e; };
  const tfLabel = (x, y, str, anchor, size, fill, weight) => {
    const t = tfMk('text', { x, y, 'text-anchor': anchor || 'start', 'font-size': size || 8.5, fill: fill || 'currentColor' });
    if (weight) t.setAttribute('font-weight', weight);
    t.textContent = str; return t;
  };
  /* organic blob path (smooth closed curve around a centre) */
  const tfBlob = (cx, cy, rx, ry) => {
    const n = 12, pts = [];
    for (let i = 0; i < n; i++) { const a = (i / n) * Math.PI * 2; const k = 0.93 + Math.random() * 0.12; pts.push([cx + Math.cos(a) * rx * k, cy + Math.sin(a) * ry * k]); }
    let d = `M ${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)} `;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
      const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
      const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += `C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2[0].toFixed(1)} ${p2[1].toFixed(1)} `;
    }
    return d + 'Z';
  };

  /* right-side legend; appended to scene and registered as a fade-in morph */
  const tfLegend = (morphs, sceneG, x, yTop, rows, immediate) => {
    const g = tfMk('g', {});
    let y = yTop;
    rows.forEach((r) => {
      if (r.title) { g.appendChild(tfLabel(x, y, r.title, 'start', 9, 'currentColor', '700')); y += r.gap || 17; return; }
      if (r.swatch === 'tri') g.appendChild(tfMk('polygon', { points: `${x + 5},${y - 8} ${x + 10},${y} ${x},${y}`, fill: r.color || '#64748b' }));
      else if (r.swatch === 'ring') { const rc = tfMk('circle', { cx: x + 5, cy: y - 3, r: 4.5, fill: 'none', stroke: r.color || 'currentColor', 'stroke-width': 1.4 }); if (r.dashed) rc.setAttribute('stroke-dasharray', '2.5 1.8'); g.appendChild(rc); }
      else if (r.swatch === 'striped') { g.appendChild(tfMk('circle', { cx: x + 5, cy: y - 3, r: 5, fill: r.color })); g.appendChild(tfMk('line', { x1: x + 1.5, y1: y, x2: x + 8.5, y2: y - 7, stroke: '#fff', 'stroke-width': 1.1 })); }
      else if (r.swatch === 'sq') g.appendChild(tfMk('rect', { x, y: y - 8, width: 10, height: 10, rx: 1, fill: r.color }));
      else if (r.swatch === 'dot') g.appendChild(tfMk('circle', { cx: x + 5, cy: y - 3, r: 5, fill: r.color }));
      g.appendChild(tfLabel(x + 18, y, r.text, 'start', 8.5));
      y += r.gap || 15;
    });
    sceneG.appendChild(g);
    // immediate → legend rides the scene fade-in (visible from the start), no delayed reveal
    if (!immediate) morphs.push({ el: g, from: { opacity: 0 }, to: { opacity: 1 }, dur: 0.4, delay: 0.5 });
    return g;
  };

  /* Still frame without GSAP: give every registered element of a scene its
     final 'to' state (opacity, SVG attributes, x/y as a translate). Morphs are
     applied in the order they were added, so the last value of a property wins. */
  const tfStill = (scene) => {
    const offsets = new Map();
    scene.morphs.forEach(({ el, to }) => {
      Object.keys(to).forEach((k) => {
        const v = to[k];
        if (k === 'attr') Object.keys(v).forEach((a) => el.setAttribute(a, v[a]));
        else if (k === 'opacity') el.setAttribute('opacity', v);
        else if (k === 'x' || k === 'y') {
          const o = offsets.get(el) || { x: 0, y: 0 };
          o[k] = v;
          offsets.set(el, o);
        }
      });
    });
    offsets.forEach((o, el) => el.setAttribute('transform', `translate(${o.x} ${o.y})`));
    scene.g.setAttribute('opacity', 1);
  };

  /* Task-flow showcase — the landing card's run → evaluate → plot scenes,
     smooth in-place morphs + per-scene legends. GSAP master timeline. */
  const taskFlow = () => {
    const wrap = document.querySelector('.taskflow');
    // instant-nav: a previous page's hero card may have left its perpetual
    // repeat:-1 timeline running on now-detached nodes (burning rAF forever).
    // Kill it when the current page has no task-flow card.
    if (!wrap) {
      if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
      if (window.__tfIO) { window.__tfIO.disconnect(); window.__tfIO = null; }
      return;
    }
    const scenesG = wrap.querySelector('.tf-scenes');
    const headNum = wrap.querySelector('.tf-num');
    const headTitle = wrap.querySelector('.tf-title');
    const dots = wrap.querySelectorAll('.tf-dots span');
    if (!scenesG) return;
    if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
    if (window.__tfIO) { window.__tfIO.disconnect(); window.__tfIO = null; }
    scenesG.innerHTML = '';

    const mk = tfMk, label = tfLabel;
    const CT = ['#3b82f6', '#10b981', '#ef4444'];

    const reg = (arr, el, from, to, dur, delay) => arr.push({ el, from, to, dur: dur || 1.3, delay: delay || 0 });

    /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       Scene sets. Each builder returns a fresh `scenes[]` array; the active set
       is chosen by `.taskflow[data-taskflow]`.
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
    const BUILDERS = {};

    /* ══════════════════════════ scMultiBench ══════════════════════════
       run → evaluate → plot, the scIB-style benchmark pipeline.
       Choreography (polish pass): no static arrows — a selection ring picks the
       method and a flowing dashed wire carries the run "signal" (m3 wire idiom);
       integration is SHOWN (grey scatter drifts into coloured clusters), every
       metric bar reports its value, and the bubble table ends on its signature
       move: whole rows glide into overall-rank order. */
    BUILDERS.multibench = () => {
    const scenes = [];
    const VIO = '#7c3aed', VIO_DEEP = '#6d28d9', VIO_SOFT = '#a78bfa', COLB = '#0ea5e9';
    const CHIP_BG = '#eef2f7', INK_MUT = '#475569', GREY_CELL = '#cbd5e1';

    /* ① Run — the registry list cascades in, a selection ring drops onto SCALEX
       (the docs' worked example), and the signal flowing down the wire turns a
       grey unintegrated scatter into three coloured clusters. */
    (() => {
      const g = tfScene(); const m = [];
      g.appendChild(label(34, 40, '36 methods', 'start', 9, 'currentColor', '700'));
      // real registry methods only (no tools the benchmark does not cover)
      const names = ['totalVI', 'Multigrate', 'SCALEX', 'Seurat WNN', 'MOFA2', '+ 31 more'];
      const SEL = 2;                                  // SCALEX — the tutorials' worked example
      const rw = 108, rh = 19, rgap = 5, rx = 34, ry0 = 50;
      const rowY = (i) => ry0 + i * (rh + rgap);
      const selCY = rowY(SEL) + rh / 2;
      names.forEach((nm, i) => {
        const y = rowY(i), sel = (i === SEL);
        const chip = mk('rect', { x: rx, y, width: rw, height: rh, rx: 4, fill: CHIP_BG, opacity: 0 });
        g.appendChild(chip);
        const txt = label(rx + 10, y + 13, nm, 'start', 9, INK_MUT, '500');
        txt.setAttribute('opacity', 0);
        g.appendChild(txt);
        // beat 1 — the list cascades in, every chip still neutral
        reg(m, chip, { opacity: 0, x: -8 }, { opacity: 1, x: 0 }, 0.45, 0.05 + i * 0.07);
        reg(m, txt, { opacity: 0, x: -8 }, { opacity: 1, x: 0 }, 0.4, 0.12 + i * 0.07);
        // beat 2 — the pick lands: SCALEX fills violet, the rest recede
        if (sel) {
          reg(m, chip, {}, { attr: { fill: VIO } }, 0.35, 1.4);
          reg(m, txt, {}, { attr: { fill: '#fff', 'font-weight': 700 } }, 0.35, 1.4);
        } else {
          reg(m, chip, {}, { opacity: 0.45 }, 0.35, 1.4);
          reg(m, txt, {}, { opacity: 0.55 }, 0.35, 1.4);
        }
      });
      // the selection ring drops down the list and stops on the picked row
      const ring = mk('rect', { x: rx - 3, y: rowY(0) - 3, width: rw + 6, height: rh + 6, rx: 6, fill: 'none', stroke: VIO, 'stroke-width': 1.8, opacity: 0 });
      g.appendChild(ring);
      reg(m, ring, { opacity: 0 }, { opacity: 1 }, 0.25, 0.6);
      reg(m, ring, {}, { y: rowY(SEL) - rowY(0) }, 0.55, 0.85);
      reg(m, ring, {}, { opacity: 0 }, 0.35, 1.75);     // hand over to the filled chip
      // flowing wire — the run signal leaves the picked chip …
      const wire = mk('path', { d: `M ${rx + rw + 4} ${selCY} C 200 ${selCY}, 214 105, 246 105`, fill: 'none', stroke: VIO, 'stroke-width': 1.8, 'stroke-dasharray': '5 5', opacity: 0 });
      g.appendChild(wire);
      reg(m, wire, { opacity: 0, attr: { 'stroke-dashoffset': 60 } }, { opacity: 0.85, attr: { 'stroke-dashoffset': 0 } }, 0.8, 1.5);
      reg(m, wire, {}, { attr: { 'stroke-dashoffset': -60 } }, 0.9, 2.3);   // keep the current flowing
      // … and three signal dots ride it into the embedding
      for (let k = 0; k < 3; k++) {
        const dot = mk('circle', { cx: rx + rw + 6, cy: selCY, r: 2.6, fill: VIO, opacity: 0 });
        g.appendChild(dot);
        reg(m, dot, { opacity: 0 }, { opacity: 1 }, 0.15, 1.6 + k * 0.22);
        reg(m, dot, {}, { x: 96, y: 105 - selCY, opacity: 0 }, 0.7, 1.75 + k * 0.22);
      }
      // the payoff: an unintegrated grey scatter DRIFTS into three clusters and
      // takes its cell-type colour — integration made visible, not implied
      const clusters = [
        { col: CT[0], cx: 296, cy: 70 },
        { col: CT[1], cx: 394, cy: 84 },
        { col: CT[2], cx: 336, cy: 148 },
      ];
      clusters.forEach((cl) => {
        const blob = mk('path', { d: tfBlob(cl.cx, cl.cy, 34, 27), fill: cl.col, opacity: 0 });
        g.appendChild(blob);
        reg(m, blob, { opacity: 0 }, { opacity: 0.13 }, 0.5, 3.1);
        for (let i = 0; i < 6; i++) {
          const ang = (i / 6) * Math.PI * 2 + Math.random() * 0.5;
          const rr2 = 5 + Math.random() * 14;
          const tx = cl.cx + Math.cos(ang) * rr2 * 1.25, ty = cl.cy + Math.sin(ang) * rr2 * 0.9;
          const sx = 262 + Math.random() * 170, sy = 42 + Math.random() * 138;
          const d = mk('circle', { cx: sx, cy: sy, r: 5, fill: GREY_CELL, opacity: 0 });
          g.appendChild(d);
          reg(m, d, { opacity: 0 }, { opacity: 0.9 }, 0.4, 0.5 + Math.random() * 0.4);
          reg(m, d, {}, { x: tx - sx, y: ty - sy, attr: { fill: cl.col }, opacity: 1 }, 1.0, 2.45 + Math.random() * 0.25);
        }
      });
      const cap = label(340, 196, 'Integrated embedding', 'middle', 9, 'currentColor', '600');
      cap.setAttribute('opacity', 0);
      g.appendChild(cap);
      reg(m, cap, { opacity: 0 }, { opacity: 1 }, 0.5, 3.2);
      scenes.push({ num: 1, title: 'Run a method', g, morphs: m });
    })();

    /* ② Evaluate — the scene-① embedding feeds the scIB panel; each metric bar
       grows and reports its value as it lands. */
    (() => {
      const g = tfScene(); const m = [];
      // input glyph: the integrated embedding, miniaturised (continuity with ①)
      const mini = [
        { col: CT[0], cx: 46, cy: 84 },
        { col: CT[1], cx: 78, cy: 92 },
        { col: CT[2], cx: 58, cy: 118 },
      ];
      mini.forEach((cl, ci) => {
        for (let i = 0; i < 5; i++) {
          const ang = (i / 5) * Math.PI * 2;
          const cx = cl.cx + Math.cos(ang) * 9, cy = cl.cy + Math.sin(ang) * 7.5;
          const d = mk('circle', { cx, cy, r: 3.4, fill: cl.col, opacity: 0 });
          g.appendChild(d);
          reg(m, d, { opacity: 0, scale: 0, svgOrigin: cx + ' ' + cy }, { opacity: 1, scale: 1, svgOrigin: cx + ' ' + cy }, 0.4, 0.1 + ci * 0.08 + i * 0.03);
        }
      });
      const glab = label(62, 148, 'Embedding', 'middle', 8.5, 'currentColor', '600');
      glab.setAttribute('opacity', 0); g.appendChild(glab);
      reg(m, glab, { opacity: 0 }, { opacity: 1 }, 0.4, 0.4);
      // wire into the metric panel
      const wire = mk('path', { d: 'M 96 100 C 116 100, 122 100, 140 100', fill: 'none', stroke: '#94a3b8', 'stroke-width': 1.6, 'stroke-dasharray': '4 4', opacity: 0 });
      g.appendChild(wire);
      reg(m, wire, { opacity: 0, attr: { 'stroke-dashoffset': 32 } }, { opacity: 0.9, attr: { 'stroke-dashoffset': 0 } }, 0.6, 0.5);
      const metrics = [
        { name: 'iLISI', v: 0.78, kind: 'batch' },
        { name: 'kBET',  v: 0.61, kind: 'batch' },
        { name: 'ASW',   v: 0.85, kind: 'bio' },
        { name: 'ARI',   v: 0.72, kind: 'bio' },
        { name: 'NMI',   v: 0.89, kind: 'bio' },
      ];
      const x0 = 186, y0 = 44, bh = 15, gap = 11, maxW = 156;
      metrics.forEach((mt, i) => {
        const y = y0 + i * (bh + gap), col = mt.kind === 'batch' ? COLB : VIO;
        g.appendChild(label(x0 - 8, y + bh - 4, mt.name, 'end', 9, 'currentColor', '600'));
        g.appendChild(mk('rect', { x: x0, y, width: maxW, height: bh, rx: 3, fill: CHIP_BG }));
        const bar = mk('rect', { x: x0, y, width: 0, height: bh, rx: 3, fill: col });
        g.appendChild(bar);
        const del = 0.7 + i * 0.14;
        reg(m, bar, { attr: { width: 0 } }, { attr: { width: Math.round(maxW * mt.v) } }, 0.8, del);
        // the value lands with its bar
        const val = label(x0 + Math.round(maxW * mt.v) + 6, y + bh - 4, mt.v.toFixed(2), 'start', 8, col, '700');
        val.setAttribute('opacity', 0);
        g.appendChild(val);
        reg(m, val, { opacity: 0, x: -6 }, { opacity: 1, x: 0 }, 0.35, del + 0.8);
      });
      // a faded extra row + ellipsis — only a few of the metrics scMultiBench computes
      const eY = y0 + metrics.length * (bh + gap);
      g.appendChild(mk('rect', { x: x0, y: eY, width: Math.round(maxW * 0.45), height: bh, rx: 3, fill: CHIP_BG, opacity: 0.6 }));
      g.appendChild(label(x0 - 8, eY + bh - 4, '…', 'end', 12, 'currentColor', '700'));
      tfLegend(m, g, 356, 72, [
        { title: 'scIB metrics' },
        { swatch: 'sq', color: COLB, text: 'Batch removal' },
        { swatch: 'sq', color: VIO, text: 'Bio conservation' },
      ]);
      scenes.push({ num: 2, title: 'Evaluate (scIB)', g, morphs: m });
    })();

    /* ③ Plot — the bubble table lands row by row, then makes its signature
       move: whole rows glide into overall-rank order; the Overall column and
       the winner's band arrive last. */
    (() => {
      const g = tfScene(); const m = [];
      // generic method labels — scMultiBench is a neutral benchmark
      const methods = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'];
      const rank = [2, 3, 0, 1, 4];             // rank[i] = final slot; Method 3 wins
      const BEST = 2;
      const metrics = ['ARI', 'NMI', 'ASW', 'iLISI', 'kBET', 'cLISI'];
      const nMet = metrics.length, x0 = 150, y0 = 62, cw = 33, ch = 25, ovX = 118;
      // winner band sits BEHIND the rows (appended first), lands after the sort
      const band = mk('rect', { x: 56, y: y0 - 12, width: 304, height: 24, rx: 6, fill: VIO, opacity: 0 });
      g.appendChild(band);
      reg(m, band, { opacity: 0 }, { opacity: 0.07 }, 0.45, 2.2);
      g.appendChild(label(x0 + (nMet - 1) * cw / 2, 30, 'Metrics', 'middle', 9, 'currentColor', '700'));
      metrics.forEach((mt, j) => g.appendChild(label(x0 + j * cw, y0 - 16, mt, 'middle', 7.5, 'currentColor', '600')));
      const ovHead = label(ovX, y0 - 16, 'Overall', 'middle', 7.5, VIO_DEEP, '700');
      ovHead.setAttribute('opacity', 0); g.appendChild(ovHead);
      reg(m, ovHead, { opacity: 0 }, { opacity: 1 }, 0.4, 2.25);
      methods.forEach((nm, i) => {
        const row = mk('g', {});
        row.appendChild(label(94, y0 + i * ch + 3, nm, 'end', 8, 'currentColor', i === BEST ? '700' : '500'));
        for (let j = 0; j < nMet; j++) {
          const cx = x0 + j * cw, cy = y0 + i * ch;
          const score = 0.35 + ((i * 7 + j * 3) % 10) / 15;
          const rr = 3.5 + score * 6.5;
          const c = mk('circle', { cx, cy, r: rr, fill: i === BEST ? VIO_DEEP : VIO_SOFT, opacity: i === BEST ? 1 : 0.8 });
          row.appendChild(c);
          reg(m, c, { opacity: 0, scale: 0, svgOrigin: cx + ' ' + cy }, { opacity: i === BEST ? 1 : 0.8, scale: 1, svgOrigin: cx + ' ' + cy }, 0.45, 0.15 + (i * nMet + j) * 0.02);
        }
        // Overall bubble appears AFTER the sort, sized by final rank
        const ocy = y0 + i * ch, orr = 9 - rank[i] * 1.4;
        const oc = mk('circle', { cx: ovX, cy: ocy, r: orr, fill: i === BEST ? VIO_DEEP : VIO_SOFT, opacity: 0 });
        row.appendChild(oc);
        reg(m, oc, { opacity: 0, scale: 0, svgOrigin: ovX + ' ' + ocy }, { opacity: i === BEST ? 1 : 0.85, scale: 1, svgOrigin: ovX + ' ' + ocy }, 0.4, 2.3 + rank[i] * 0.07);
        g.appendChild(row);
        // the signature move — the whole row glides to its overall-rank slot
        if (rank[i] !== i) reg(m, row, {}, { y: (rank[i] - i) * ch }, 0.85, 1.35);
      });
      // ellipsis column — many more scIB metrics than the six shown
      const eX = x0 + nMet * cw;
      g.appendChild(label(eX, y0 - 16, '…', 'middle', 12, 'currentColor', '700'));
      for (let i = 0; i < methods.length; i++) g.appendChild(mk('circle', { cx: eX, cy: y0 + i * ch, r: 4, fill: VIO_SOFT, opacity: 0.3 }));
      tfLegend(m, g, 368, 64, [
        { title: 'Bubble = score' },
        { swatch: 'dot', color: VIO_DEEP, text: 'Top method' },
        { swatch: 'dot', color: VIO_SOFT, text: 'Others' },
      ]);
      scenes.push({ num: 3, title: 'Rank & plot results', g, morphs: m });
    })();

    return scenes;
    };  /* end BUILDERS.multibench */

    /* ── Render the active scene-set; dots are rebuilt to match its length ── */
    const render = (key) => {
      if (!BUILDERS[key]) key = 'multibench';
      if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
      scenesG.innerHTML = '';
      const scenes = BUILDERS[key]();
      scenes.forEach((s) => scenesG.appendChild(s.g));

      const dotsWrap = wrap.querySelector('.tf-dots');
      if (dotsWrap) {
        dotsWrap.innerHTML = '';
        scenes.forEach((_, i) => {
          const sp = document.createElement('span');
          if (i === 0) sp.className = 'is-active';
          dotsWrap.appendChild(sp);
        });
      }
      const dots = wrap.querySelectorAll('.tf-dots span');

    const setHead = (i) => {
      headNum.textContent = scenes[i].num;
      headTitle.textContent = scenes[i].title;
      dots.forEach((d, j) => d.classList.toggle('is-active', j === i));
    };

    const gsap = window.gsap;
    setHead(0);
    // No GSAP (blocked CDN) or reduced motion: show the first scene as a still frame.
    if (!gsap || REDUCED_MOTION) { tfStill(scenes[0]); return; }

    const master = gsap.timeline({ repeat: -1 });
    window.__tfTl = master;
    const startTimes = [];                   // where each scene begins on the master
    const builtAt = [];                      // where each scene's build-up has finished
    scenes.forEach((s, idx) => {
      startTimes[idx] = master.duration();   // current end = this scene's start time
      const st = gsap.timeline();
      st.call(() => setHead(idx), null, 0);
      st.set(s.g, { opacity: 0 }, 0);
      s.morphs.forEach((mo) => st.set(mo.el, mo.from, 0));
      st.to(s.g, { opacity: 1, duration: 0.4, ease: 'power2.out' }, 0);
      let maxEnd = 0.45;
      s.morphs.forEach((mo) => {
        st.to(mo.el, Object.assign({ duration: mo.dur, ease: 'power2.inOut', delay: mo.delay }, mo.to), 0.45);
        maxEnd = Math.max(maxEnd, 0.45 + mo.delay + mo.dur);
      });
      st.to(s.g, { opacity: 0, duration: 0.45, ease: 'power2.in' }, maxEnd + 3.3); /* +2s hold */
      builtAt[idx] = startTimes[idx] + maxEnd;
      master.add(st);
    });

    /* Open the loop on the first scene's finished frame. seek() renders at
       once, without waiting for an animation frame, so the card has content
       in a headless render, a background tab or a paused timeline. The next
       loops play the first scene's build-up from the start. */
    master.seek(builtAt[0]);

    /* Clickable dots: jump to any scene and keep auto-playing from there.
       The master is one declarative timeline, so seeking to a scene's start
       renders that scene's exact state; play() resumes the loop. Dots are
       rebuilt on every render(), so handlers are wired fresh each switch
       (old dot elements — and their listeners — are discarded). */
    dots.forEach((dot, i) => {
      dot.addEventListener('click', () => { setHead(i); master.seek(startTimes[i]).play(); });
    });
    };  /* end render() */

    render((wrap.dataset.taskflow) || 'multibench');

    /* Perf: pause the perpetual loop when the card is scrolled off-screen, so it
       isn't burning CPU/rAF the whole time you're reading further down the page. */
    if ('IntersectionObserver' in window) {
      /* The observer must die with the timeline it controls: a stale observer
         from a previous instant-nav visit can deliver a late "not intersecting"
         record for its now-detached card and pause the NEW page's timeline
         through the shared window.__tfTl - the "animation stuck after
         navigating back" bug. Hence the global handle + isConnected guard. */
      window.__tfIO = new IntersectionObserver((ents) => {
        ents.forEach((e) => {
          if (!e.target.isConnected) return;
          const tl = window.__tfTl;
          if (tl) { e.isIntersecting ? tl.play() : tl.pause(); }
        });
      }, { threshold: 0.01 });
      window.__tfIO.observe(wrap);
    }
  };

  /* Sliding TOC indicator — one vertical bar that animates to the active heading,
     instead of a border jumping between items.
     Tracks Material's `.md-nav__link--active`, which the scrollspy toggles. */
  const tocSlider = () => {
    const getCtx = () => {
      const list = document.querySelector('.md-sidebar--secondary .md-nav--secondary > .md-nav__list');
      if (!list) return null;
      let bar = list.querySelector(':scope > .toc-indicator');
      if (!bar) { bar = document.createElement('div'); bar.className = 'toc-indicator'; list.appendChild(bar); }
      return { list, bar };
    };
    let raf = null;
    const move = () => {
      raf = null;
      const ctx = getCtx();
      if (!ctx) return;
      const active = document.querySelector('.md-sidebar--secondary .md-nav__link--active');
      // Keep the bar where it is during the brief moments the scrollspy has no
      // active link (it toggles classes in two steps) — avoids flicker.
      if (!active) return;
      const a = active.getBoundingClientRect(), l = ctx.list.getBoundingClientRect();
      ctx.bar.style.height = Math.round(a.height) + 'px';
      ctx.bar.style.transform = 'translateY(' + Math.round(a.top - l.top) + 'px)';
      ctx.bar.style.opacity = '1';
    };
    const schedule = () => { if (raf == null) raf = requestAnimationFrame(move); };
    if (!window.__tocSliderWired) {
      window.__tocSliderWired = true;
      window.addEventListener('scroll', schedule, { passive: true });
      window.addEventListener('resize', schedule, { passive: true });
      // NOTE: a document.body-wide class MutationObserver used to re-run move()
      // (a getBoundingClientRect reflow) on EVERY class change Material makes —
      // scrollspy, header autohide, nav state — thrashing layout on every nav
      // click and every scroll frame. Removed; scroll/resize + the timed move()
      // calls below and the per-navigation re-init keep the indicator in sync.
    }
    move();                      // create + position the bar immediately
    setTimeout(move, 400);       // re-position after fonts/layout settle
    setTimeout(move, 1200);      // and once more after instant-nav / late render
  };

  const init = () => {
    if (REDUCED_MOTION) { taskFlow(); return; }
    enableSectionNumbersOnHome();
    tagSectionEyebrows();
    heroReveal();
    initFadeUp();
    stickyHeader();
    scrollProgress();
    backToTop();
    magneticButtons();
    numberCounters();
    gsapLanding();
    taskFlow();
    tocSlider();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  if (typeof document$ !== 'undefined' && document$.subscribe) {
    document$.subscribe(init);
  }
})();
