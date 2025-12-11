---
layout: default
title: Home
nav_order: 1
---

# Quantitative Research


## Technical Indicators

<ul>
  {% assign indicators = site.pages
       | where_exp: "p", "p.path contains 'docs/technical_indicators/'"
       | sort: "nav_order" %}
  {% for p in indicators %}
    {% if p.title and p.nav_exclude != true %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## Strategies & Backtests

<ul>
  {% assign strategies = site.pages
       | where_exp: "p", "p.path contains 'docs/strategies/'"
       | sort: "nav_order" %}
  {% for p in strategies %}
    {% if p.title and p.nav_exclude != true %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>