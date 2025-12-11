---
layout: default
title: Quantitative Research
nav_order: 1
---

# Quantitative Research

Notes and projects in quantitative finance, statistics, algo trading, and machine learning.

## Time Series

<ul>
  {% assign time_series = site.pages
       | where_exp: "p", "p.path contains 'docs/time_series/'"
       | sort: "nav_order" %}
  {% for p in time_series %}
    {% if p.title and p.nav_exclude != true %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

## Portfolio Management

<ul>
  {% assign portfolio = site.pages
       | where_exp: "p", "p.path contains 'docs/portfolio_management/'"
       | sort: "nav_order" %}
  {% for p in portfolio %}
    {% if p.title and p.nav_exclude != true %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>


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