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

  /* ── Task-flow card: the landing hero's run → evaluate → plot animation ──

     One stage, one GSAP timeline. The same SVG elements live through all
     three steps, so nothing is swapped behind a cross-fade:
       Run       the code line types mtb.run(...) with Matilda on D11
                 (CITE-seq, one batch); the grey cells of the unintegrated
                 scatter regroup by cell type, taking the cell-type colour.
       Evaluate  the embedding steps aside; one bar per metric fills to its
                 score, the value counting with it.
       Plot      each bar closes into the bubble of its metric in the first
                 row; the other rows join, the Overall column fills, and the
                 rows glide into rank order.
     The loop has no blank frame: it opens on the grey scatter, and after
     the table clears the scatter comes back, so the last frame is the first.
     Motion grammar: power3.out for anything that enters, power3.inOut for
     anything that travels, power2.out for exits, no overshoot. Colours come from the site's CSS
     tokens (light and dark), read when the card is drawn. The scores are an
     illustration, not benchmark results. */
  const TF_NS = 'http://www.w3.org/2000/svg';
  const tfMk = (t, a, parent) => {
    const e = document.createElementNS(TF_NS, t);
    for (const k in a) e.setAttribute(k, a[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const tfText = (parent, x, y, str, o) => {
    o = o || {};
    const t = tfMk('text', { x, y, 'text-anchor': o.anchor || 'start', 'font-size': o.size || 9.5, fill: o.fill || 'currentColor' }, parent);
    if (o.weight) t.setAttribute('font-weight', o.weight);
    if (o.cls) t.setAttribute('class', o.cls);
    t.textContent = str;
    return t;
  };
  /* seeded PRNG (mulberry32): every visit draws the same cells */
  const tfRng = (seed) => () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  /* about N(0, 1) (Irwin-Hall, four uniforms): clouds get a dense core */
  const tfGauss = (rnd) => (rnd() + rnd() + rnd() + rnd() - 2) * 1.732;

  const taskFlow = () => {
    const wrap = document.querySelector('.taskflow');
    // instant-nav: a previous page's card may have left its endless timeline
    // running on detached nodes. Kill it, with its observers.
    if (window.__tfTl) { window.__tfTl.kill(); window.__tfTl = null; }
    if (window.__tfIO) { window.__tfIO.disconnect(); window.__tfIO = null; }
    if (window.__tfMO) { window.__tfMO.disconnect(); window.__tfMO = null; }
    if (!wrap) return;
    const svg = wrap.querySelector('.tf-stage svg');
    const scenesG = wrap.querySelector('.tf-scenes');
    const headNum = wrap.querySelector('.tf-num');
    const headTitle = wrap.querySelector('.tf-title');
    const dotsWrap = wrap.querySelector('.tf-dots');
    const gsap = window.gsap;
    // Without GSAP the static fallback drawn in index.md stays as it is.
    if (!svg || !scenesG || !gsap) return;

    const W = 480, H = 222;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    wrap.classList.add('is-live');

    const TITLES = ['Run a method', 'Evaluate', 'Plot'];
    let dots = [];
    if (dotsWrap) {
      dotsWrap.innerHTML = '';
      dots = TITLES.map((t) => {
        const d = document.createElement('span');
        d.setAttribute('role', 'button');
        d.setAttribute('aria-label', t);
        dotsWrap.appendChild(d);
        return d;
      });
    }

    const tok = () => {
      const cs = getComputedStyle(wrap);
      const v = (n, fb) => (cs.getPropertyValue(n) || '').trim() || fb;
      return {
        ink: v('--text', '#0f172a'), faint: v('--text-faint', '#64748b'),
        border: v('--border', '#e2e8f0'), subtle: v('--bg-subtle', '#f1f5f9'),
        accent: v('--tf-accent', '#0046ad'), bubble: v('--tf-bubble', '#8fb8e4'),
        cellGrey: v('--tf-cell', '#94a3b8'),
        ct: [v('--tf-c1', '#2563eb'), v('--tf-c2', '#0d9488'), v('--tf-c3', '#d97706')],
      };
    };

    /* ━━━━━━━━━━━━━━━━━━ geometry and content ━━━━━━━━━━━━━━━━━━ */
    const CODE = [
      'mtb.run("D11", "vertical", "out/", methods=["Matilda"])',
      'mtb.evaluate(emb, labels="celltype")',
      'mtb.plot.bubble(mtb.load_results("vertical", dataset="D11"))',
    ];
    // the metrics the stored results of D11 hold (one batch: no batch metrics)
    const METRICS = ['ARI', 'NMI', 'ASW', 'cLISI', 'iASW', 'iF1'];
    const ROWS = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'];
    const SCORES = [
      [0.78, 0.61, 0.85, 0.72, 0.89, 0.80],
      [0.66, 0.52, 0.74, 0.60, 0.70, 0.72],
      [0.90, 0.82, 0.92, 0.86, 0.93, 0.88],
      [0.84, 0.70, 0.88, 0.80, 0.86, 0.84],
      [0.45, 0.40, 0.58, 0.42, 0.55, 0.50],
    ];
    const OVERALL = SCORES.map((r) => r.reduce((a, b) => a + b, 0) / r.length);
    const ORDER = OVERALL.map((_, i) => i).sort((a, b) => OVERALL[b] - OVERALL[a]);
    const RANK = OVERALL.map((_, i) => ORDER.indexOf(i));
    const BEST = ORDER[0];
    const bubR = (s) => 2.8 + (s - 0.35) * 12.5;

    // scatter: one batch, three cell types
    const TYPE = [{ x: 172, y: 86 }, { x: 322, y: 86 }, { x: 247, y: 176 }];
    const PER = 26;                                               // cells per type
    // evaluate: the scatter steps aside to the left
    const AWAY = { s: 0.55, x: -37, y: 56 };
    // metric bars
    const BAR = { x: 292, w: 138, h: 11, y0: 56, dy: 24 };
    // bubble table
    const TB = { nameX: 96, ovX: 130, x0: 180, cw: 44, y0: 80, ch: 30, headY: 57 };
    const colX = (j) => TB.x0 + j * TB.cw;
    const rowY = (i) => TB.y0 + i * TB.ch;

    /* ━━━━━━━━━━━━━━━━━━ timeline marks (seconds) ━━━━━━━━━━━━━━━━━━ */
    const T = { type1: 0.25, run: 1.45, move: 1.6, s2: 5.0, s3: 9.2, out: 14.8, back: 15.1, end: 15.8 };
    const SCENE_START = [0, T.s2, T.s3];
    const SCENE_BUILT = [3.3, T.s2 + 2.3, T.s3 + 3.6];

    const build = () => {
      const C = tok();
      scenesG.innerHTML = '';
      const root = tfMk('g', {}, scenesG);
      const tl = gsap.timeline({ repeat: -1, paused: true });
      const set = (el, vars) => tl.set(el, vars, 0);
      const to = (el, vars, at) => tl.to(el, vars, at);
      const IN = 'power3.out', MOVE = 'power3.inOut', OUT = 'power2.out';

      /* ── code line ── */
      const bar = tfMk('g', {}, root);
      tfMk('rect', { x: 0.5, y: 0.5, width: W - 1, height: 30, rx: 8, fill: C.subtle, stroke: C.border }, bar);
      tfText(bar, 13, 19.5, '>>>', { fill: C.faint, cls: 'tf-code' });
      const code = tfText(bar, 37, 19.5, '', { fill: C.ink, cls: 'tf-code' });
      const caret = tfMk('rect', { x: 38, y: 10, width: 1.4, height: 12, fill: C.accent }, bar);
      const CH = 5.72;                                 // advance of one 9.5px mono glyph
      const show = (str, n) => {
        const k = Math.round(n);
        code.textContent = str.slice(0, k);
        caret.setAttribute('x', 37 + k * CH + 1);
      };
      const typeLine = (str, at, dur) => {
        const p = { n: 0 };
        tl.fromTo(p, { n: 0 }, { n: str.length, duration: dur, ease: 'none', immediateRender: false, onUpdate: () => show(str, p.n) }, at);
      };
      const eraseLine = (str, at, dur) => {
        const p = { n: str.length };
        tl.fromTo(p, { n: str.length }, { n: 0, duration: dur, ease: 'power1.in', immediateRender: false, onUpdate: () => show(str, p.n) }, at);
      };
      tl.call(() => show('', 0), null, 0);
      set(caret, { opacity: 1 });

      /* ── ① Run ── */
      const scat = tfMk('g', {}, root);
      set(scat, { opacity: 1, x: 0, y: 0, scale: 1, svgOrigin: '0 0' });
      const rnd = tfRng(7);
      const cells = [];
      for (let t = 0; t < TYPE.length; t++) {
        for (let k = 0; k < PER; k++) {
          // before: one broad grey cloud with no cell-type structure
          const bx = 247 + tfGauss(rnd) * 62, by = 130 + tfGauss(rnd) * 30;
          const ax = TYPE[t].x + tfGauss(rnd) * 17, ay = TYPE[t].y + tfGauss(rnd) * 12;
          const el = tfMk('circle', { cx: bx, cy: by, r: 3.8, fill: C.cellGrey }, scat);
          cells.push({ el, t, bx, by, ax, ay });
        }
      }
      cells.forEach((c) => {
        const paint = 'fill';
        set(c.el, { opacity: 1, attr: { cx: c.bx, cy: c.by, [paint]: C.cellGrey } });
        const u = Math.min(1, Math.max(0, (c.bx - 90) / 320));    // 0 at the left edge, 1 at the right
        // loop: once the table has cleared, the grey scatter of time 0 comes back
        tl.set(c.el, { opacity: 0, attr: { cx: c.bx, cy: c.by, [paint]: C.cellGrey } }, T.back);
        to(c.el, { opacity: 1, duration: 0.5, ease: IN }, T.back + 0.05 + u * 0.3);
        // the regrouping sweeps left to right, so the eye can follow it
        const d = u * 0.38;
        to(c.el, { attr: { cx: c.ax, cy: c.ay }, duration: 1.15, ease: MOVE }, T.move + d);
        to(c.el, { attr: { [paint]: C.ct[c.t] }, duration: 0.7, ease: 'power2.out' }, T.move + d + 0.3);
      });

      typeLine(CODE[0], T.type1, 1.05);
      to(caret, { opacity: 0, duration: 0.15 }, T.run);

      /* ── ② Evaluate ── */
      to(caret, { opacity: 1, duration: 0.1 }, T.s2);
      eraseLine(CODE[0], T.s2, 0.35);
      typeLine(CODE[1], T.s2 + 0.45, 0.75);
      to(caret, { opacity: 0, duration: 0.15 }, T.s2 + 1.4);

      to(scat, { x: AWAY.x, y: AWAY.y, scale: AWAY.s, duration: 0.95, ease: MOVE }, T.s2 + 0.1);
      const embLab = tfText(root, 99, 186, 'Embedding', { fill: C.faint, anchor: 'middle', weight: 600 });
      set(embLab, { opacity: 0, y: 4 });
      to(embLab, { opacity: 1, y: 0, duration: 0.4, ease: IN }, T.s2 + 0.85);

      const evalG = tfMk('g', {}, root);
      set(evalG, { opacity: 1 });
      const bars = [], vals = [];
      METRICS.forEach((name, j) => {
        const y = BAR.y0 + j * BAR.dy, v = SCORES[0][j];
        const lab = tfText(evalG, BAR.x - 10, y + 9, name, { fill: C.ink, anchor: 'end', weight: 600 });
        const tr = tfMk('rect', { x: BAR.x, y, width: BAR.w, height: BAR.h, rx: BAR.h / 2, fill: C.subtle }, evalG);
        const br = tfMk('rect', { x: BAR.x, y, width: 0, height: BAR.h, rx: BAR.h / 2, fill: C.accent }, root);
        const val = tfText(root, BAR.x + BAR.w + 34, y + 9, v.toFixed(2), { fill: C.ink, anchor: 'end', weight: 600, cls: 'tf-tnum' });
        bars.push(br); vals.push(val);
        const at = T.s2 + 0.8 + j * 0.05;
        set([lab, tr, val], { opacity: 0, x: -6 });
        to([lab, tr], { opacity: 1, x: 0, duration: 0.45, ease: IN }, at);
        // the bar fills to its score from the left (width, not scaleX, so the
        // round ends keep their shape); the value counts with it
        set(br, { opacity: 0, y: 0, attr: { x: BAR.x, y, width: 0, height: BAR.h, rx: BAR.h / 2, fill: C.accent } });
        const fillAt = T.s2 + 1.1 + j * 0.08;
        to(br, { opacity: 1, duration: 0.01 }, fillAt);
        to(br, { attr: { width: BAR.w * v }, duration: 0.9, ease: IN }, fillAt);
        to(val, { opacity: 1, x: 0, duration: 0.35, ease: IN }, fillAt);
        const cnt = { v: 0 };
        tl.fromTo(cnt, { v: 0 }, { v, duration: 0.9, ease: IN, immediateRender: false,
          onUpdate: () => { val.textContent = cnt.v.toFixed(2); } }, fillAt);
      });

      /* ── ③ Plot ── */
      to(caret, { opacity: 1, duration: 0.1 }, T.s3);
      eraseLine(CODE[1], T.s3, 0.3);
      typeLine(CODE[2], T.s3 + 0.4, 1.0);
      to(caret, { opacity: 0, duration: 0.15 }, T.s3 + 1.55);

      to([scat, embLab], { opacity: 0, duration: 0.35, ease: OUT }, T.s3);
      to(evalG, { opacity: 0, duration: 0.3, ease: OUT }, T.s3);
      to(vals, { opacity: 0, duration: 0.2, ease: OUT }, T.s3);

      const table = tfMk('g', {}, root);
      set(table, { opacity: 1 });
      // the top row's band sits behind the rows (appended first)
      const band = tfMk('rect', { x: 22, y: rowY(0) - 14, width: colX(5) + 22 - 22, height: 28, rx: 7, fill: C.accent }, table);
      set(band, { opacity: 0, y: 0 });
      const heads = METRICS.map((name, j) => tfText(table, colX(j), TB.headY, name, { size: 9, fill: C.faint, anchor: 'middle', weight: 600 }));
      set(heads, { opacity: 0, y: 3 });
      to(heads, { opacity: 1, y: 0, duration: 0.4, ease: IN, stagger: 0.03 }, T.s3 + 0.5);
      const ovHead = tfText(table, TB.ovX, TB.headY, 'Overall', { size: 9, fill: C.ink, anchor: 'middle', weight: 700 });
      const ruleX = (TB.ovX + TB.x0) / 2;
      const rule = tfMk('line', { x1: ruleX, y1: TB.headY + 9, x2: ruleX, y2: rowY(4) + 14, stroke: C.border, 'stroke-width': 1 }, table);
      set([ovHead, rule], { opacity: 0 });

      const rowEls = ROWS.map(() => []);
      const ovBubs = [], bestEls = [];
      ROWS.forEach((name, i) => {
        const y = rowY(i), first = i === 0;
        const nm = tfText(table, TB.nameX, y + 3.5, name, { size: 9, fill: C.ink, anchor: 'end', weight: 500 });
        rowEls[i].push(nm);
        if (i === BEST) bestEls.push(nm);
        set(nm, { opacity: 0, x: -5, y: 0 });
        to(nm, { opacity: 1, x: 0, duration: 0.4, ease: IN }, first ? T.s3 + 0.75 : T.s3 + 1.3 + i * 0.09);
        if (!first) {
          SCORES[i].forEach((s, j) => {
            const cx = colX(j), b = tfMk('circle', { cx, cy: y, r: bubR(s), fill: C.bubble }, table);
            rowEls[i].push(b);
            if (i === BEST) bestEls.push(b);
            set(b, { opacity: 0, scale: 0.4, y: 0, attr: { fill: C.bubble }, svgOrigin: `${cx} ${y}` });
            to(b, { opacity: 1, scale: 1, duration: 0.45, ease: IN }, T.s3 + 1.3 + i * 0.09 + j * 0.025);
          });
        }
        const ob = tfMk('circle', { cx: TB.ovX, cy: y, r: bubR(OVERALL[i]), fill: C.bubble }, table);
        rowEls[i].push(ob); ovBubs.push(ob);
        if (i === BEST) bestEls.push(ob);
        set(ob, { opacity: 0, scale: 0.4, y: 0, attr: { fill: C.bubble }, svgOrigin: `${TB.ovX} ${y}` });
      });
      // the bars of ② close into the bubbles of the first row
      bars.forEach((br, j) => {
        const r = bubR(SCORES[0][j]), cx = colX(j), cy = rowY(0), by = BAR.y0 + j * BAR.dy;
        rowEls[0].push(br);
        to(br, { attr: { x: BAR.x + BAR.w * SCORES[0][j] / 2 - r, y: by + BAR.h / 2 - r, width: 2 * r, height: 2 * r, rx: r, fill: C.bubble }, duration: 0.4, ease: MOVE }, T.s3 + 0.1 + j * 0.03);
        to(br, { attr: { x: cx - r, y: cy - r }, duration: 0.75, ease: MOVE }, T.s3 + 0.45 + j * 0.04);
      });
      // Overall: the mean of the row, filled in rank order
      const T_OV = T.s3 + 2.1;
      to([ovHead, rule], { opacity: 1, duration: 0.4, ease: IN }, T_OV);
      ovBubs.forEach((ob, i) => to(ob, { opacity: 1, scale: 1, duration: 0.45, ease: IN }, T_OV + 0.15 + RANK[i] * 0.07));
      // rows glide into rank order (transform y only)
      const T_SORT = T_OV + 0.85;
      ROWS.forEach((_, i) => {
        const dy = (RANK[i] - i) * TB.ch;
        if (dy) to(rowEls[i], { y: dy, duration: 0.85, ease: MOVE }, T_SORT + i * 0.03);
      });
      // the top row, now first, takes the brand colour
      const T_TOP = T_SORT + 0.95;
      to(band, { opacity: 0.08, duration: 0.5, ease: IN }, T_TOP);
      bestEls.forEach((el) => to(el, el.tagName === 'text'
        ? { attr: { 'font-weight': 700, fill: C.accent }, duration: 0.3, ease: IN }
        : { attr: { fill: C.accent }, duration: 0.4, ease: IN }, T_TOP));
      bestEls.filter((el) => el.tagName === 'text').forEach((el) => set(el, { attr: { 'font-weight': 500, fill: C.ink } }));

      /* ── loop: the finished table rests, then the stage clears ── */
      to(caret, { opacity: 1, duration: 0.1 }, T.out);
      eraseLine(CODE[2], T.out, 0.4);
      to([table, ...bars], { opacity: 0, duration: 0.45, ease: OUT }, T.out);
      tl.set(scat, { opacity: 1, x: 0, y: 0, scale: 1 }, T.back);
      tl.set({}, {}, T.end);                          // fixes the loop length

      return tl;
    };

    /* ━━━━━━━━━━━━━━━━━━ playback ━━━━━━━━━━━━━━━━━━ */
    const sceneAt = (t) => (t >= T.back ? 0 : t >= T.s3 ? 2 : t >= T.s2 ? 1 : 0);
    let lastScene = -1;
    const paintHead = (t) => {
      const s = sceneAt(t);
      if (s === lastScene) return;
      lastScene = s;
      if (headNum) headNum.textContent = s + 1;
      if (headTitle) headTitle.textContent = TITLES[s];
      dots.forEach((d, i) => d.classList.toggle('is-active', i === s));
    };

    const start = (at) => {
      if (window.__tfTl) window.__tfTl.kill();
      const tl = build();
      window.__tfTl = tl;
      tl.eventCallback('onUpdate', () => paintHead(tl.time()));
      tl.seek(at, false);
      paintHead(at);
      if (!REDUCED_MOTION) tl.play();
      return tl;
    };
    /* Open on the grey scatter and play the run at once. With reduced motion
       the card shows the first step's finished frame instead. */
    start(REDUCED_MOTION ? SCENE_BUILT[0] : 0);

    // Dots jump to a step and keep playing from there. With reduced motion
    // each dot shows its step's finished frame and nothing moves.
    dots.forEach((d, i) => d.addEventListener('click', () => {
      const tl = window.__tfTl;
      if (!tl) return;
      if (REDUCED_MOTION) { tl.seek(SCENE_BUILT[i], false); paintHead(SCENE_BUILT[i]); }
      else tl.seek(SCENE_START[i] + 0.001, false).play();
    }));

    // Colours are read from the theme when drawn: redraw on a theme switch,
    // at the same moment of the loop.
    window.__tfMO = new MutationObserver(() => {
      const tl = window.__tfTl;
      if (!tl || !wrap.isConnected) return;
      const t = tl.time(), playing = !tl.paused();
      const nt = start(t);
      if (!playing) nt.pause();
    });
    window.__tfMO.observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });

    // Pause the loop while the card is off screen.
    if ('IntersectionObserver' in window && !REDUCED_MOTION) {
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
