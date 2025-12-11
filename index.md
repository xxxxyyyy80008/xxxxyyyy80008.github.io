---
layout: default
title: Home
nav_order: 1
---

# Quantitative Research

.

## Contents

<ul>
  {% assign docs_pages = site.pages
       | where_exp: "p", "p.path contains 'docs/'"
       | sort: "nav_order" %}
  {% for p in docs_pages %}
    {% if p.title and p.nav_exclude != true %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>



