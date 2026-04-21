(function () {
  const body = document.querySelector('.post-body');
  const tocContainer = document.querySelector('.post-toc');
  const nav = document.querySelector('.post-toc-nav');
  const article = document.querySelector('.post-layout');

  function slugify(text) {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-');
  }

  function buildToc() {
    if (!body || !tocContainer || !nav) return null;

    const headings = Array.from(body.querySelectorAll('h2, h3'));
    if (headings.length < 2) {
      tocContainer.classList.add('is-empty');
      return null;
    }

    const ol = document.createElement('ol');
    const items = [];
    headings.forEach(function (h) {
      if (!h.id) h.id = slugify(h.textContent);
      const li = document.createElement('li');
      li.className = h.tagName === 'H3' ? 'is-h3' : 'is-h2';
      const a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      ol.appendChild(li);
      items.push({ id: h.id, el: h, link: a });
    });
    nav.appendChild(ol);
    return items;
  }

  function setupScrollSpy(items) {
    if (!items || items.length === 0) return;

    let activeIndex = -1;
    const setActive = function (index) {
      if (index === activeIndex) return;
      activeIndex = index;
      items.forEach(function (item, i) {
        item.link.classList.toggle('is-active', i === index);
      });
    };

    const update = function () {
      const viewportTop = window.scrollY + 120;
      let newIndex = 0;
      for (let i = 0; i < items.length; i++) {
        const rect = items[i].el.getBoundingClientRect();
        const absTop = rect.top + window.scrollY;
        if (absTop <= viewportTop) {
          newIndex = i;
        } else {
          break;
        }
      }
      setActive(newIndex);
    };

    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  function setupProgressBar() {
    if (!article) return;

    const bar = document.createElement('div');
    bar.className = 'reading-progress';
    document.body.appendChild(bar);

    const update = function () {
      const rect = article.getBoundingClientRect();
      const total = article.offsetHeight - window.innerHeight;
      if (total <= 0) {
        bar.style.width = '100%';
        return;
      }
      const scrolled = -rect.top;
      const pct = Math.max(0, Math.min(1, scrolled / total));
      bar.style.width = (pct * 100).toFixed(2) + '%';
    };

    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    update();
  }

  const items = buildToc();
  setupScrollSpy(items);
  setupProgressBar();
})();
