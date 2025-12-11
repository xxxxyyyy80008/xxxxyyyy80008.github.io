---
layout: default
title: Quantitative Research
nav_order: 1
---

# Quantitative Research

Welcome to the site. Below is an overview of the main sections.

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