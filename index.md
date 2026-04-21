---
layout: default
title: Home
---

<section class="panel home-panel">
  <div class="section-heading section-heading-home">
    <h2>Current focus</h2>
    <p>Using probes and evals to study deceptive behavior in language models.</p>
  </div>
  <div class="hero-actions">
    <a class="button-link button-link-primary" href="{{ '/writing/' | relative_url }}">Read writing</a>
    <a class="button-link" href="https://github.com/neveratdennys">View GitHub</a>
  </div>

  <div class="focus-list" aria-label="Current focus items">
    <article class="focus-item">
      <h3>Detecting deception in language models</h3>
      <p>Probe-based analysis of deceptive behavior and internal model signals.</p>
    </article>
  </div>
</section>

<section class="panel">
  <div class="section-heading">
    <h2>Recent writing</h2>
  </div>
  {% include post_feed.html posts=site.posts limit=3 %}
</section>
