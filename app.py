"""
Goodreads Recommender - Streamlit app (OPAN6604, Project 2).

Collaborative Filtering on Goodreads ratings with an optional Gemini re-ranking
layer. Pick a user and model settings; the app shows that user's top-N unseen
books ranked by predicted rating, then can personalize that candidate list to a
stated preference.
"""
import json
import os
import base64
import re
from html import escape

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False

try:
    from surprise import BaselineOnly, Dataset, KNNBasic, Reader, SVD
    SURPRISE_AVAILABLE = True
except ModuleNotFoundError:
    SURPRISE_AVAILABLE = False


# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Goodreads Recommender",
    page_icon="Books",
    layout="wide",
)


# ------------------------------------------------------------- visual theme
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.22), transparent 34%),
            radial-gradient(circle at top right, rgba(96, 165, 250, 0.20), transparent 32%),
            linear-gradient(135deg, #0e2730 0%, #143342 48%, #1d3554 100%);
        color: #e5eef7;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2029 0%, #14283a 100%);
        border-right: 2px solid rgba(250, 204, 21, 0.72);
        box-shadow: 0 0 26px rgba(250, 204, 21, 0.14), inset -1px 0 0 rgba(250, 204, 21, 0.36);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 26px;
        border: 1px solid rgba(250, 204, 21, 0.32);
        border-radius: 10px;
        margin: 12px 10px;
        padding-left: 14px;
        padding-right: 14px;
        background:
            radial-gradient(circle at top, rgba(250, 204, 21, 0.08), transparent 34%),
            rgba(13, 32, 41, 0.58);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc;
        text-shadow: 0 0 12px rgba(45, 212, 191, 0.34);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: #c9e7f2 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(18, 39, 56, 0.94) !important;
        border: 1px solid rgba(250, 204, 21, 0.55) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        box-shadow: 0 0 14px rgba(250, 204, 21, 0.12);
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #fde68a !important;
        box-shadow: 0 0 0 1px #fde68a, 0 0 18px rgba(250, 204, 21, 0.28) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: 6px 0 14px 0;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background: #fde68a !important;
        border: 2px solid #fff7cc !important;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.55);
    }
    .control-panel {
        padding: 14px 14px 16px 14px;
        margin: 0 0 18px 0;
        background:
            radial-gradient(circle at top left, rgba(250, 204, 21, 0.14), transparent 42%),
            rgba(18, 39, 56, 0.72);
        border: 1px solid rgba(250, 204, 21, 0.55);
        border-radius: 8px;
        box-shadow: 0 0 22px rgba(250, 204, 21, 0.13);
    }
    .control-title {
        color: #fde68a;
        font-size: 20px;
        font-weight: 900;
        margin-bottom: 4px;
        text-shadow: 0 0 12px rgba(250, 204, 21, 0.34);
    }
    .control-subtitle {
        color: #8fb7c8;
        font-size: 12px;
        line-height: 1.35;
    }
    .demo-pill {
        display: inline-block;
        margin-top: 10px;
        padding: 7px 9px;
        color: #fde68a;
        background: rgba(250, 204, 21, 0.10);
        border: 1px solid rgba(250, 204, 21, 0.30);
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.35;
    }
    .prompt-panel {
        margin-top: 16px;
        padding: 14px 16px;
        border-radius: 8px;
        border: 1px solid rgba(125, 211, 252, 0.26);
        background: rgba(18, 39, 56, 0.78);
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.10);
    }
    .prompt-label {
        color: #bff7ea;
        font-size: 18px;
        font-weight: 900;
        text-shadow: 0 0 10px rgba(45, 212, 191, 0.28);
        margin-bottom: 4px;
    }
    .prompt-help {
        color: #a8c7d8;
        font-size: 13px;
        margin-bottom: 10px;
    }
    .main-title {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 0;
        color: #f8fafc;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.40);
        margin-bottom: 4px;
    }
    .subtitle {
        color: #a8c7d8;
        font-size: 16px;
        margin-bottom: 22px;
    }
    .hero-panel {
        padding: 20px 22px;
        border: 1px solid rgba(94, 234, 212, 0.22);
        background: rgba(13, 31, 43, 0.82);
        box-shadow: 0 0 24px rgba(14, 165, 233, 0.16);
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .hero-content {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .logo-badge {
        background: #f4f1ea;
        border: 1px solid rgba(250, 204, 21, 0.55);
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 0 18px rgba(250, 204, 21, 0.12);
        flex: 0 0 auto;
    }
    .goodreads-logo {
        display: block;
        width: 220px;
        max-width: 28vw;
        height: auto;
    }
    .hero-copy {
        min-width: 0;
    }
    .section-title {
        display: inline-block;
        font-size: 23px;
        font-weight: 850;
        color: #bff7ea;
        border-bottom: 2px solid #2dd4bf;
        padding-bottom: 4px;
        margin: 10px 0 16px 0;
        text-shadow: 0 0 10px rgba(45, 212, 191, 0.30);
    }
    .stat-box {
        padding: 16px;
        background: radial-gradient(circle at top, #1d3b52 0%, #102536 66%);
        border: 1px solid rgba(125, 211, 252, 0.26);
        border-radius: 8px;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.14);
        min-height: 96px;
    }
    .stat-value {
        font-size: 31px;
        font-weight: 900;
        color: #7dd3fc;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 13px;
        color: #aab8c5;
        margin-top: 6px;
    }
    .stat-box.user-stat {
        background:
            radial-gradient(circle at top left, rgba(250, 204, 21, 0.20), transparent 40%),
            radial-gradient(circle at top right, rgba(45, 212, 191, 0.18), transparent 42%),
            #102f37;
        border: 1px solid rgba(250, 204, 21, 0.55);
        box-shadow: 0 0 22px rgba(250, 204, 21, 0.15), 0 0 18px rgba(45, 212, 191, 0.10);
    }
    .stat-box.user-stat .stat-value {
        color: #fde68a;
        text-shadow: 0 0 14px rgba(250, 204, 21, 0.28);
    }
    .stat-box.user-stat .stat-label {
        color: #d8f3ec;
    }
    .stat-chip {
        display: inline-block;
        margin-bottom: 8px;
        padding: 4px 7px;
        border-radius: 999px;
        border: 1px solid rgba(250, 204, 21, 0.45);
        color: #fde68a;
        background: rgba(250, 204, 21, 0.10);
        font-size: 11px;
        font-weight: 850;
    }
    .book-card {
        padding: 14px 16px;
        background: rgba(18, 39, 56, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-left: 3px solid #2dd4bf;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .book-rank {
        color: #7dd3fc;
        font-weight: 900;
        font-size: 13px;
    }
    .book-title {
        color: #f8fafc;
        font-weight: 800;
        font-size: 16px;
    }
    .book-meta {
        color: #a8c7d8;
        font-size: 13px;
        margin-top: 4px;
    }
    .model-note {
        background: rgba(250, 204, 21, 0.12);
        border: 1px solid rgba(250, 204, 21, 0.34);
        color: #fde68a;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 14px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        overflow: hidden;
    }
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(18, 39, 56, 0.78);
        font-size: 13px;
    }
    .dark-table th {
        background: rgba(11, 30, 43, 0.96);
        color: #7dd3fc;
        text-align: left;
        padding: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        font-weight: 850;
    }
    .dark-table td {
        color: #dceaf7;
        padding: 9px 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        vertical-align: top;
    }
    .dark-table tr:nth-child(even) td {
        background: rgba(45, 75, 98, 0.34);
    }
    .dark-table tr:hover td {
        background: rgba(45, 212, 191, 0.09);
    }
    .rank-cell {
        color: #7dd3fc !important;
        font-weight: 850;
        text-align: center;
        width: 54px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 8px 14px;
        border-radius: 8px;
        border: 1px solid rgba(125, 211, 252, 0.18);
        background: rgba(18, 39, 56, 0.70);
        color: #a8c7d8;
        font-weight: 800;
        letter-spacing: 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(45, 212, 191, 0.10);
        border-color: rgba(45, 212, 191, 0.35);
        color: #e5f8ff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(45, 212, 191, 0.24), rgba(56, 189, 248, 0.16)) !important;
        border-color: rgba(45, 212, 191, 0.72) !important;
        color: #f8fafc !important;
        box-shadow: 0 0 18px rgba(45, 212, 191, 0.18);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #2dd4bf;
        height: 3px;
        border-radius: 999px;
    }
    div[data-testid="stTextArea"] label {
        color: #bff7ea !important;
        font-size: 15px !important;
        font-weight: 850 !important;
    }

    /* Lighter page background with the original dark dashboard surface style */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(20, 184, 166, 0.18), transparent 34%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.15), transparent 32%),
            linear-gradient(135deg, #071014 0%, #0b1622 48%, #111827 100%);
        color: #e5eef7;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d2029 0%, #14283a 100%);
        border-right: 2px solid rgba(250, 204, 21, 0.72);
        box-shadow: 0 0 26px rgba(250, 204, 21, 0.14), inset -1px 0 0 rgba(250, 204, 21, 0.36);
    }
    [data-testid="stSidebar"] > div:first-child {
        background:
            radial-gradient(circle at top, rgba(250, 204, 21, 0.08), transparent 34%),
            rgba(13, 32, 41, 0.58);
        border-color: rgba(250, 204, 21, 0.32);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: #f4f1ea !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(18, 39, 56, 0.94) !important;
        border: 1px solid rgba(250, 204, 21, 0.55) !important;
        color: #f8fafc !important;
        box-shadow: 0 0 14px rgba(250, 204, 21, 0.12);
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #fde68a !important;
        box-shadow: 0 0 0 1px #fde68a, 0 0 18px rgba(250, 204, 21, 0.28) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background: #fde68a !important;
        border: 2px solid #fff7cc !important;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.55);
    }
    .control-panel {
        background:
            radial-gradient(circle at top left, rgba(250, 204, 21, 0.14), transparent 42%),
            rgba(18, 39, 56, 0.72);
        border-color: rgba(250, 204, 21, 0.55);
        box-shadow: 0 0 22px rgba(250, 204, 21, 0.13);
    }
    .control-title {
        color: #fde68a;
        text-shadow: 0 0 12px rgba(250, 204, 21, 0.34);
    }
    .control-subtitle {
        color: #8fb7c8;
    }
    .demo-pill,
    .model-note {
        color: #fde68a;
        background: rgba(250, 204, 21, 0.10);
        border-color: rgba(250, 204, 21, 0.30);
    }
    .hero-panel,
    .stat-box,
    .book-card,
    .prompt-panel {
        background: rgba(18, 39, 56, 0.78);
        border-color: rgba(125, 211, 252, 0.26);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.14);
    }
    .main-title,
    .book-title {
        color: #f8fafc;
        text-shadow: 0 0 18px rgba(56, 189, 248, 0.40);
    }
    .subtitle,
    .book-meta,
    .prompt-help,
    .stat-label {
        color: #a8c7d8;
    }
    .section-title,
    .prompt-label {
        color: #bff7ea;
        border-bottom-color: #2dd4bf;
        text-shadow: 0 0 10px rgba(45, 212, 191, 0.30);
    }
    .stat-value,
    .book-rank,
    .rank-cell,
    .dark-table th {
        color: #7dd3fc !important;
    }
    .stat-box.user-stat {
        background:
            radial-gradient(circle at top left, rgba(250, 204, 21, 0.20), transparent 40%),
            radial-gradient(circle at top right, rgba(45, 212, 191, 0.18), transparent 42%),
            #102f37;
        border-color: rgba(250, 204, 21, 0.55);
        box-shadow: 0 0 22px rgba(250, 204, 21, 0.15), 0 0 18px rgba(45, 212, 191, 0.10);
    }
    .stat-box.user-stat .stat-value {
        color: #fde68a;
        text-shadow: 0 0 14px rgba(250, 204, 21, 0.28);
    }
    .stat-chip {
        color: #fde68a;
        background: rgba(250, 204, 21, 0.10);
        border-color: rgba(250, 204, 21, 0.45);
    }
    .dark-table {
        background: rgba(18, 39, 56, 0.78);
        border-color: rgba(148, 163, 184, 0.22);
    }
    .dark-table th {
        background: rgba(11, 30, 43, 0.96);
        border-bottom-color: rgba(148, 163, 184, 0.18);
    }
    .dark-table td {
        color: #dceaf7;
        border-bottom-color: rgba(148, 163, 184, 0.12);
    }
    .dark-table tr:nth-child(even) td {
        background: rgba(45, 75, 98, 0.34);
    }
    .dark-table tr:hover td {
        background: rgba(45, 212, 191, 0.09);
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(18, 39, 56, 0.70);
        color: #a8c7d8;
        border-color: rgba(125, 211, 252, 0.18);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(45, 212, 191, 0.10);
        border-color: rgba(45, 212, 191, 0.35);
        color: #e5f8ff;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(45, 212, 191, 0.24), rgba(56, 189, 248, 0.16)) !important;
        border-color: rgba(45, 212, 191, 0.72) !important;
        color: #f8fafc !important;
        box-shadow: 0 0 18px rgba(45, 212, 191, 0.18);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #2dd4bf;
    }
    div[data-testid="stTextArea"] textarea {
        background: #f8fafc !important;
        color: #111827 !important;
        border: 1px solid rgba(125, 211, 252, 0.26) !important;
    }
    .model-explainer {
        margin-top: 10px;
        padding: 10px 11px;
        border-radius: 8px;
        border: 1px solid rgba(125, 211, 252, 0.28);
        background: rgba(18, 39, 56, 0.76);
        color: #dceaf7;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.10);
        font-size: 13px;
        line-height: 1.38;
    }
    .model-explainer b {
        color: #bff7ea;
    }
    .model-guide {
        margin-top: 14px;
        padding: 12px 12px;
        border-radius: 8px;
        border: 1px solid rgba(250, 204, 21, 0.36);
        background: rgba(18, 39, 56, 0.54);
        color: #dceaf7;
        font-size: 12px;
        line-height: 1.42;
    }
    .guide-title {
        color: #fde68a;
        font-size: 15px;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .guide-block {
        padding: 9px 9px;
        border-left: 3px solid rgba(125, 211, 252, 0.30);
        margin-bottom: 10px;
        background: rgba(8, 16, 24, 0.22);
        border-radius: 6px;
    }
    .guide-block.active-guide {
        border-left-color: #fde68a;
        background: rgba(250, 204, 21, 0.10);
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.10);
    }
    .guide-heading {
        color: #bff7ea;
        font-weight: 900;
        margin-bottom: 3px;
    }
    .guide-example {
        color: #a8c7d8;
        margin-top: 5px;
    }

    /* Goodreads-inspired final palette */
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(64, 157, 105, 0.10), transparent 32%),
            linear-gradient(135deg, #f4f1ea 0%, #eee8dc 54%, #e7dccb 100%);
        color: #382110;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #382110 0%, #4d3624 100%);
        border-right: 2px solid #8d6748;
        box-shadow: 0 0 22px rgba(56, 33, 16, 0.22), inset -1px 0 0 rgba(244, 241, 234, 0.20);
    }
    [data-testid="stSidebar"] > div:first-child {
        background:
            radial-gradient(circle at top, rgba(244, 241, 234, 0.08), transparent 36%),
            rgba(56, 33, 16, 0.34);
        border-color: rgba(196, 173, 139, 0.52);
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: #f4f1ea !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #f4f1ea !important;
        border: 1px solid #c4ad8b !important;
        color: #382110 !important;
        box-shadow: 0 0 10px rgba(196, 173, 139, 0.24);
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #00635d !important;
        box-shadow: 0 0 0 1px #00635d, 0 0 14px rgba(0, 99, 93, 0.22) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background: #00635d !important;
        border: 2px solid #f4f1ea !important;
        box-shadow: 0 0 10px rgba(0, 99, 93, 0.38);
    }
    .control-panel,
    .model-guide,
    .model-explainer {
        background: rgba(244, 241, 234, 0.10);
        border-color: rgba(196, 173, 139, 0.72);
        box-shadow: none;
    }
    .control-title,
    .guide-title {
        color: #f4f1ea;
        text-shadow: none;
    }
    .control-subtitle,
    .guide-example {
        color: #d8cabb;
    }
    .demo-pill,
    .model-note {
        color: #382110;
        background: #f4f1ea;
        border-color: #c4ad8b;
    }
    .hero-panel,
    .stat-box,
    .book-card,
    .prompt-panel {
        background: rgba(255, 252, 246, 0.90);
        border-color: #d6c8b6;
        box-shadow: 0 8px 24px rgba(56, 33, 16, 0.08);
    }
    .main-title,
    .book-title {
        color: #382110;
        text-shadow: none;
    }
    .subtitle,
    .book-meta,
    .prompt-help,
    .stat-label {
        color: #5f4b37;
    }
    .section-title,
    .prompt-label,
    .guide-heading,
    .model-explainer b {
        color: #00635d;
        text-shadow: none;
    }
    .section-title {
        border-bottom-color: #00635d;
    }
    .stat-value,
    .book-rank,
    .rank-cell,
    .dark-table th {
        color: #00635d !important;
    }
    .stat-box.user-stat {
        background: #f6eedd;
        border-color: #8d6748;
        box-shadow: 0 8px 24px rgba(141, 103, 72, 0.13);
    }
    .stat-box.user-stat .stat-value {
        color: #382110;
        text-shadow: none;
    }
    .stat-box.user-stat .stat-label {
        color: #5f4b37;
    }
    .stat-chip {
        color: #382110;
        background: rgba(196, 173, 139, 0.34);
        border-color: #8d6748;
    }
    .dark-table {
        background: rgba(255, 252, 246, 0.92);
        border-color: #d6c8b6;
    }
    .dark-table th {
        background: #eee8dc;
        border-bottom-color: #d6c8b6;
    }
    .dark-table td {
        color: #382110;
        border-bottom-color: #e4d8c7;
    }
    .dark-table tr:nth-child(even) td {
        background: rgba(244, 241, 234, 0.78);
    }
    .dark-table tr:hover td,
    .guide-block.active-guide {
        background: rgba(64, 157, 105, 0.12);
    }
    .book-card {
        border-left-color: #00635d;
    }
    .guide-block {
        background: rgba(244, 241, 234, 0.10);
        border-left-color: #c4ad8b;
    }
    .guide-block.active-guide {
        border-left-color: #00635d;
        box-shadow: none;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 252, 246, 0.78);
        color: #5f4b37;
        border-color: #d6c8b6;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(64, 157, 105, 0.12);
        border-color: #00635d;
        color: #382110;
    }
    .stTabs [aria-selected="true"] {
        background: #00635d !important;
        border-color: #00635d !important;
        color: #fffaf2 !important;
        box-shadow: 0 4px 14px rgba(0, 99, 93, 0.18);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #8d6748;
    }
    div[data-testid="stTextArea"] textarea {
        background: #fffaf2 !important;
        color: #382110 !important;
        border: 1px solid #c4ad8b !important;
    }

    /* Sidebar readability refinement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #294a3f 0%, #355c4f 100%);
        border-right: none;
        box-shadow: none;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
        border: none;
        box-shadow: none;
    }
    .control-panel,
    .model-guide {
        background: rgba(244, 241, 234, 0.14);
        border-color: rgba(244, 241, 234, 0.42);
    }
    .model-guide {
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .guide-block {
        background: rgba(255, 252, 246, 0.16);
        border-left-color: #c4ad8b;
    }
    .guide-block.active-guide {
        background: rgba(255, 252, 246, 0.22);
        border-left-color: #f4f1ea;
    }
    .guide-heading {
        color: #fffaf2;
    }
    .guide-example,
    .control-subtitle {
        color: #e7dccb;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 13px;
        color: #382110;
    }
    .stTabs [aria-selected="true"] {
        font-weight: 900;
    }
    div[role="radiogroup"] {
        gap: 8px;
        border-bottom: 1px solid #d6c8b6;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    div[role="radiogroup"] > label {
        background: rgba(255, 252, 246, 0.78) !important;
        border: 1px solid #d6c8b6 !important;
        border-radius: 8px !important;
        padding: 8px 13px !important;
        color: #5f4b37 !important;
        font-weight: 750 !important;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        background: #00635d !important;
        border-color: #00635d !important;
        color: #fffaf2 !important;
        box-shadow: 0 4px 14px rgba(0, 99, 93, 0.18);
    }
    div[role="radiogroup"] > label:has(input:checked) * {
        color: #fffaf2 !important;
    }
    div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p {
        color: #fffaf2 !important;
        font-weight: 850 !important;
    }

    .model-guide {
        background: rgba(255, 252, 246, 0.94) !important;
        border: 1px solid #d6c8b6 !important;
        box-shadow: 0 8px 24px rgba(56, 33, 16, 0.08) !important;
    }
    .model-guide .guide-title {
        color: #382110 !important;
    }
    .model-guide .guide-heading {
        color: #00635d !important;
    }
    .model-guide .guide-block,
    .model-guide .guide-block.active-guide {
        background: transparent !important;
        border-left-color: #00635d !important;
        box-shadow: none !important;
    }
    .model-guide,
    .model-guide .guide-example {
        color: #5f4b37 !important;
    }
    .prompt-panel {
        background: rgba(255, 252, 246, 0.94) !important;
        border: 1px solid #d6c8b6 !important;
        box-shadow: 0 8px 24px rgba(56, 33, 16, 0.08) !important;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .prompt-label {
        color: #00635d !important;
        font-size: 20px !important;
        font-weight: 900 !important;
    }
    .prompt-help {
        color: #5f4b37 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        font-size: 17px !important;
        line-height: 1.45 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        font-size: 18px !important;
        min-height: 46px !important;
    }
    [data-testid="stSidebar"] .model-guide {
        padding: 18px 20px !important;
    }
    [data-testid="stSidebar"] .model-guide .guide-title {
        font-size: 19px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stSidebar"] .model-guide .guide-heading {
        font-size: 16px !important;
        line-height: 1.35 !important;
    }
    [data-testid="stSidebar"] .model-guide .guide-block,
    [data-testid="stSidebar"] .model-guide .guide-example {
        font-size: 15px !important;
        line-height: 1.45 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: 15px !important;
        line-height: 1.45 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p {
        font-size: 15px !important;
        line-height: 1.45 !important;
    }
    .app-footer {
        margin-top: 48px;
        padding: 22px 16px 28px;
        border-top: 1px solid #d6c8b6;
        color: #00635d;
        font-size: 18px;
        font-weight: 900;
        text-align: center;
    }
    .field-label {
        color: #00635d !important;
        font-size: 18px !important;
        font-weight: 900 !important;
        margin: 18px 0 8px;
    }
    [data-testid="stTextArea"] label,
    [data-testid="stTextArea"] label p {
        color: #00635d !important;
        font-size: 18px !important;
        font-weight: 900 !important;
    }
    .book-card-with-cover {
        display: grid;
        grid-template-columns: 86px minmax(0, 1fr);
        gap: 13px;
        align-items: start;
        padding: 12px !important;
    }
    .book-cover-wrap {
        width: 86px;
        aspect-ratio: 2 / 3;
        border-radius: 6px;
        overflow: hidden;
        background: #eee8dc;
        border: 1px solid #d6c8b6;
        box-shadow: 0 8px 18px rgba(56, 33, 16, 0.14);
    }
    .book-cover {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }
    .book-cover-placeholder {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 8px;
        color: #8a6f4d;
        font-size: 12px;
        font-weight: 800;
        line-height: 1.15;
        text-align: center;
    }
    .book-card-body {
        min-width: 0;
    }
    .book-description {
        margin-top: 7px;
        color: #5f4b37;
        font-size: 12.5px;
        line-height: 1.35;
    }
    .book-score-row {
        margin-top: 8px;
        color: #5f4b37;
        font-size: 12.5px;
        line-height: 1.35;
    }
    .book-score-row b {
        color: #00635d;
    }
    @media (max-width: 700px) {
        .book-card-with-cover {
            grid-template-columns: 70px minmax(0, 1fr);
        }
        .book-cover-wrap {
            width: 70px;
        }
    }

    /* Rotten Tomatoes-inspired final theme override */
    .stApp {
        background: #ffffff !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] {
        background: #f3f4f6 !important;
        border-right: 4px solid #fa320a !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #111111 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 2px solid #fa320a !important;
        color: #111111 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background: #fa320a !important;
        border: 2px solid #ffffff !important;
        box-shadow: none !important;
    }
    .hero-panel {
        background: #f3f4f6 !important;
        border: none !important;
        border-radius: 0 !important;
        border-top: 8px solid #fa320a !important;
        box-shadow: none !important;
    }
    .logo-badge {
        background: #ffffff !important;
        border: 2px solid #fa320a !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    .main-title {
        color: #111111 !important;
        text-shadow: none !important;
    }
    .subtitle {
        color: #fa320a !important;
    }
    .section-title,
    .prompt-label,
    .field-label,
    .guide-heading,
    .model-explainer b {
        color: #111111 !important;
        border-bottom-color: #fa320a !important;
        text-shadow: none !important;
    }
    .section-title::before {
        content: "";
        display: inline-block;
        width: 7px;
        height: 22px;
        background: #fa320a;
        margin-right: 10px;
        vertical-align: -4px;
    }
    .book-card,
    .stat-box,
    .prompt-panel,
    .model-guide,
    .model-explainer {
        background: #ffffff !important;
        border: 1px solid #d9d9d9 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    .book-card {
        border-left: 5px solid #fa320a !important;
    }
    .book-title,
    .stat-value {
        color: #111111 !important;
        text-shadow: none !important;
    }
    .book-rank,
    .rank-cell,
    .dark-table th,
    .book-score-row b {
        color: #fa320a !important;
    }
    .book-meta,
    .book-description,
    .book-score-row,
    .prompt-help,
    .stat-label,
    .guide-example,
    .control-subtitle {
        color: #4b5563 !important;
    }
    .book-cover-wrap {
        background: #f3f4f6 !important;
        border: 1px solid #d9d9d9 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    .dark-table {
        background: #ffffff !important;
        border: 1px solid #d9d9d9 !important;
        border-radius: 4px !important;
    }
    .dark-table th {
        background: #111111 !important;
        color: #ffffff !important;
        border-bottom: 3px solid #fa320a !important;
    }
    .dark-table td {
        color: #111111 !important;
        border-bottom-color: #eeeeee !important;
    }
    .dark-table tr:nth-child(even) td {
        background: #f7f7f7 !important;
    }
    .dark-table tr:hover td {
        background: #fff4cc !important;
    }
    .stTabs [data-baseweb="tab"],
    div[role="radiogroup"] > label {
        background: #ffffff !important;
        border: 1px solid #d9d9d9 !important;
        border-radius: 4px !important;
        color: #111111 !important;
        box-shadow: none !important;
    }
    .stTabs [aria-selected="true"],
    div[role="radiogroup"] > label:has(input:checked) {
        background: #fa320a !important;
        border-color: #fa320a !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] *,
    div[role="radiogroup"] > label:has(input:checked) * {
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #f8c51c !important;
    }
    .model-note,
    .demo-pill,
    .stat-box.user-stat,
    .stat-chip {
        background: #fff4cc !important;
        border: 1px solid #f8c51c !important;
        color: #111111 !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #111111 !important;
        border: 2px solid #d9d9d9 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #fa320a !important;
        box-shadow: 0 0 0 1px #fa320a !important;
    }
    .stButton button {
        background: #fa320a !important;
        color: #ffffff !important;
        border: 1px solid #fa320a !important;
        border-radius: 4px !important;
        font-weight: 900 !important;
    }
    .stButton button:hover {
        background: #d92905 !important;
        border-color: #d92905 !important;
        color: #ffffff !important;
    }
    .app-footer {
        background: #111111 !important;
        border-top: 5px solid #fa320a !important;
        color: #ffffff !important;
    }
    [data-testid="stMarkdownContainer"] code {
        display: none !important;
    }

    /* Letterboxd-inspired book discovery theme */
    .stApp {
        background:
            radial-gradient(circle at top, rgba(69, 85, 103, 0.20), transparent 28%),
            linear-gradient(180deg, #1f2a35 0%, #14191f 28%, #0f1419 100%) !important;
        color: #d8e0e8 !important;
        font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif !important;
    }
    .letterboxd-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        margin: -22px -12px 26px;
        padding: 18px 28px;
        background: rgba(13, 18, 23, 0.94);
        border-bottom: 1px solid rgba(123, 150, 174, 0.18);
    }
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
    }
    .brand-dots {
        display: flex;
        gap: 0;
    }
    .brand-dot {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-block;
        margin-right: -4px;
    }
    .brand-dot.orange { background: #ff8000; }
    .brand-dot.green { background: #00e054; }
    .brand-dot.blue { background: #40bcf4; }
    .brand-name {
        color: #ffffff;
        font-size: 33px;
        font-weight: 900;
        letter-spacing: -1px;
        line-height: 1;
    }
    .letterboxd-links {
        display: flex;
        gap: 22px;
        align-items: center;
        flex-wrap: wrap;
        justify-content: flex-end;
        color: #c8d7e6;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .letterboxd-links span {
        white-space: nowrap;
    }
    .letterboxd-search {
        width: 148px;
        height: 34px;
        border-radius: 999px;
        background: rgba(120, 135, 150, 0.34);
        border: 1px solid rgba(172, 190, 207, 0.16);
    }
    .hero-panel {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 20px 0 26px !important;
        margin-bottom: 12px !important;
    }
    .hero-content {
        display: block !important;
        text-align: center;
    }
    .logo-badge {
        display: none !important;
    }
    .main-title {
        color: #ffffff !important;
        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 44px !important;
        font-weight: 900 !important;
        line-height: 1.22 !important;
        letter-spacing: -0.5px !important;
        max-width: 980px;
        margin: 0 auto 18px !important;
    }
    .subtitle {
        color: #9ab7d2 !important;
        font-size: 17px !important;
        letter-spacing: 0.3px !important;
    }
    .letterboxd-feature-title,
    .letterboxd-section-kicker {
        color: #9ab7d2;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 18px 0 12px;
    }
    .letterboxd-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 0 0 28px;
    }
    .letterboxd-feature-card {
        min-height: 98px;
        padding: 18px 20px;
        background: #44576a;
        border: 1px solid #53687c;
        border-radius: 4px;
        color: #eef5fb;
        font-size: 18px;
        line-height: 1.35;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .letterboxd-feature-icon {
        color: #9fb4c8;
        font-size: 34px;
        font-weight: 900;
        min-width: 46px;
        text-align: center;
    }
    .section-title,
    .prompt-label,
    .field-label {
        color: #9ab7d2 !important;
        border-bottom: 1px solid #435364 !important;
        font-weight: 500 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    .section-title::before {
        display: none !important;
    }
    .book-card,
    .stat-box,
    .prompt-panel,
    .model-guide,
    .model-explainer,
    .dark-table {
        background: #1f2933 !important;
        border: 1px solid #394858 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        color: #d8e0e8 !important;
    }
    .book-card {
        border-left: 4px solid #00e054 !important;
    }
    .book-title {
        color: #ffffff !important;
        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 18px !important;
        text-shadow: none !important;
    }
    .book-rank,
    .rank-cell,
    .book-score-row b,
    .stat-value {
        color: #00e054 !important;
    }
    .book-meta,
    .book-description,
    .book-score-row,
    .prompt-help,
    .stat-label,
    .guide-example,
    .control-subtitle,
    p {
        color: #9ab7d2 !important;
    }
    .book-cover-wrap {
        background: #2c3440 !important;
        border: 1px solid #53687c !important;
        border-radius: 4px !important;
        position: relative;
        transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }
    .book-cover-wrap:hover {
        border-color: #00e054 !important;
        box-shadow: 0 0 0 3px #00e054, 0 10px 22px rgba(0, 0, 0, 0.36) !important;
        transform: translateY(-1px);
    }
    .book-cover-wrap::before {
        content: attr(data-title);
        position: absolute;
        left: 50%;
        top: -44px;
        transform: translateX(-50%);
        max-width: 240px;
        padding: 8px 11px;
        background: #44576a;
        color: #d7e6f5;
        border-radius: 4px;
        font-size: 13px;
        font-weight: 800;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        opacity: 0;
        pointer-events: none;
        z-index: 20;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.26);
        transition: opacity 120ms ease, top 120ms ease;
    }
    .book-cover-wrap::after {
        content: "";
        position: absolute;
        left: 50%;
        top: -12px;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 8px solid #44576a;
        opacity: 0;
        pointer-events: none;
        z-index: 20;
        transition: opacity 120ms ease, top 120ms ease;
    }
    .book-cover-wrap:hover::before {
        top: -48px;
        opacity: 1;
    }
    .book-cover-wrap:hover::after {
        top: -15px;
        opacity: 1;
    }
    [data-testid="stSidebar"] {
        background: #151c23 !important;
        border-right: 1px solid #2d3945 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #c8d7e6 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        background: #2c3440 !important;
        border: 1px solid #44576a !important;
        color: #ffffff !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] .model-guide,
    [data-testid="stSidebar"] .model-guide .guide-block,
    [data-testid="stSidebar"] .model-guide .guide-block.active-guide {
        background: #e8f0f7 !important;
        border-color: #9ab7d2 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .model-guide {
        color: #17212b !important;
    }
    [data-testid="stSidebar"] .model-guide .guide-title {
        color: #17212b !important;
    }
    [data-testid="stSidebar"] .model-guide .guide-heading,
    [data-testid="stSidebar"] .model-guide b {
        color: #007c3d !important;
    }
    [data-testid="stSidebar"] .model-guide .guide-example,
    [data-testid="stSidebar"] .model-guide .guide-block,
    [data-testid="stSidebar"] .model-guide p {
        color: #34495d !important;
    }
    [data-testid="stSidebar"] .demo-pill {
        background: #fff4b8 !important;
        border-color: #ffcc33 !important;
        color: #17212b !important;
    }
    [data-testid="stSidebar"] .stCaptionContainer,
    [data-testid="stSidebar"] .stCaptionContainer p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #c8d7e6 !important;
    }
    .stButton button,
    .stTabs [aria-selected="true"],
    div[role="radiogroup"] > label:has(input:checked) {
        background: #ff8000 !important;
        border-color: #ff8000 !important;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab"],
    div[role="radiogroup"] > label {
        background: #1f2933 !important;
        border: 1px solid #394858 !important;
        color: #c8d7e6 !important;
        border-radius: 4px !important;
    }
    .dark-table th {
        background: #151c23 !important;
        color: #9ab7d2 !important;
        border-bottom: 1px solid #435364 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .dark-table td {
        background: #1f2933 !important;
        color: #d8e0e8 !important;
        border-bottom: 1px solid #394858 !important;
    }
    .dark-table tr:nth-child(even) td,
    .dark-table tr:hover td {
        background: #25313d !important;
    }
    .catalog-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 260px));
        gap: 24px;
        align-items: start;
        justify-content: start;
        margin-top: 16px;
    }
    .catalog-card {
        min-width: 0;
        position: relative;
        width: 100%;
        max-width: 260px;
    }
    .catalog-cover {
        width: 100%;
        height: 100%;
        max-width: 260px;
        aspect-ratio: 2 / 3;
        object-fit: cover;
        display: block;
        border-radius: 4px;
        border: 1px solid #53687c;
        background: #2c3440;
        box-shadow: 0 10px 18px rgba(0, 0, 0, 0.28);
        transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }
    .catalog-cover-frame {
        width: 100%;
        max-width: 260px;
        aspect-ratio: 2 / 3;
        overflow: hidden;
        border-radius: 4px;
        border: 1px solid #53687c;
        background: #2c3440;
        box-shadow: 0 10px 18px rgba(0, 0, 0, 0.28);
    }
    .catalog-cover-frame .catalog-cover {
        width: 100%;
        height: 100%;
        max-width: none;
        border: 0;
        box-shadow: none;
        border-radius: 0;
        object-fit: cover;
    }
    .generated-cover {
        width: 100%;
        height: 100%;
        padding: 18px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background:
            radial-gradient(circle at top left, rgba(64, 188, 244, 0.24), transparent 38%),
            linear-gradient(145deg, #24313c 0%, #141b22 62%, #0f1419 100%);
        color: #ffffff;
    }
    .generated-cover-mark {
        align-self: flex-start;
        color: #00e054;
        border: 2px solid #00e054;
        border-radius: 999px;
        padding: 8px 10px;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 1px;
    }
    .generated-cover-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 25px;
        font-weight: 900;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }
    .generated-cover-author {
        color: #9ab7d2;
        font-size: 14px;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .catalog-card:hover .catalog-cover {
        border-color: #00e054;
        box-shadow: 0 0 0 3px #00e054, 0 12px 24px rgba(0, 0, 0, 0.42);
        transform: translateY(-1px);
    }
    .catalog-card:hover .catalog-cover-frame {
        border-color: #00e054;
        box-shadow: 0 0 0 3px #00e054, 0 12px 24px rgba(0, 0, 0, 0.42);
        transform: translateY(-1px);
    }
    .catalog-card::before {
        content: attr(data-title);
        position: absolute;
        left: 50%;
        top: -44px;
        transform: translateX(-50%);
        max-width: min(260px, 95%);
        padding: 8px 11px;
        background: #44576a;
        color: #d7e6f5;
        border-radius: 4px;
        font-size: 14px;
        font-weight: 800;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        opacity: 0;
        pointer-events: none;
        z-index: 20;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.26);
        transition: opacity 120ms ease, top 120ms ease;
    }
    .catalog-card::after {
        content: "";
        position: absolute;
        left: 50%;
        top: -12px;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 8px solid #44576a;
        opacity: 0;
        pointer-events: none;
        z-index: 20;
        transition: opacity 120ms ease, top 120ms ease;
    }
    .catalog-card:hover::before {
        top: -48px;
        opacity: 1;
    }
    .catalog-card:hover::after {
        top: -15px;
        opacity: 1;
    }
    .catalog-title {
        color: #ffffff;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 24px;
        font-weight: 900;
        line-height: 1.18;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }
    .catalog-meta {
        color: #9ab7d2;
        font-size: 16px;
        line-height: 1.35;
        margin-top: 6px;
        overflow-wrap: anywhere;
    }
    .catalog-stats {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        color: #9ab7d2;
        font-size: 15px;
        margin-top: 8px;
    }
    .catalog-stats b {
        color: #00e054;
    }
    .reviewer-panel {
        padding: 0 0 10px;
        min-width: 280px;
    }
    .reviewer-heading {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #9ab7d2;
        border-bottom: 1px solid #435364;
        font-size: 18px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    .reviewer-more {
        font-size: 13px;
        color: #9ab7d2;
    }
    button[kind="tertiary"] {
        background: transparent !important;
        border: 0 !important;
        color: #9ab7d2 !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: 0 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }
    button[kind="tertiary"]:hover {
        color: #ffffff !important;
        background: transparent !important;
        border: 0 !important;
    }
    .reviewer-row {
        display: grid;
        grid-template-columns: 58px minmax(0, 1fr);
        gap: 13px;
        align-items: center;
        padding: 13px 0;
        border-bottom: 1px solid #2d3945;
    }
    .reviewer-avatar {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        object-fit: cover;
        border: 1px solid #53687c;
        background: #2c3440;
    }
    .reviewer-name {
        color: #ffffff;
        font-size: 21px;
        font-weight: 900;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }
    .reviewer-stats {
        color: #7f99b3;
        font-size: 15px;
        margin-top: 4px;
    }
    .reviewer-id {
        color: #6f8498;
        font-size: 12px;
        margin-top: 3px;
        letter-spacing: 0.4px;
    }
    .reader-profile-card {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 18px 20px;
        margin: 8px 0 18px;
        background: #1f2933;
        border: 1px solid #394858;
        border-radius: 4px;
    }
    .reader-profile-avatar {
        width: 92px;
        height: 92px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #53687c;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.26);
        flex: 0 0 auto;
    }
    .reader-profile-name {
        color: #ffffff;
        font-size: 30px;
        font-weight: 900;
        line-height: 1.1;
    }
    .reader-profile-meta {
        color: #9ab7d2;
        font-size: 16px;
        margin-top: 6px;
    }
    .pagination-note {
        color: #9ab7d2;
        font-size: 14px;
        margin: 22px 0 8px;
    }
    .pagination-ellipsis {
        color: #ffcc33;
        font-size: 24px;
        font-weight: 900;
        text-align: center;
        padding-top: 8px;
    }
    div[data-testid="stHorizontalBlock"]:has(.pagination-note) .stButton button {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 900 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
        min-height: 42px !important;
        border-radius: 4px !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.pagination-note) .stButton button:disabled {
        background: #44576a !important;
        border-color: #53687c !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }
    .cant-find-panel {
        margin-top: 30px;
    }
    .cant-find-title {
        color: #9ab7d2;
        border-bottom: 1px solid #435364;
        font-size: 18px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }
    .cant-find-copy {
        color: #9ab7d2;
        font-size: 17px;
        line-height: 1.45;
    }
    .cant-find-copy b {
        color: #ffffff;
    }
    .app-footer {
        background: transparent !important;
        border-top: 1px solid #394858 !important;
        color: #9ab7d2 !important;
    }
    @media (max-width: 850px) {
        .letterboxd-nav {
            align-items: flex-start;
            flex-direction: column;
        }
        .letterboxd-feature-grid {
            grid-template-columns: 1fr;
        }
        .main-title {
            font-size: 34px !important;
            text-align: left;
        }
        .hero-content {
            text-align: left;
        }
    }

    /* Letterboxd browse-bar style for app sections */
    div[role="radiogroup"] {
        display: inline-flex !important;
        align-items: stretch !important;
        gap: 0 !important;
        padding: 0 !important;
        margin: 14px 0 24px !important;
        border: 1px solid #2c3a46 !important;
        border-radius: 4px !important;
        overflow: hidden !important;
        background: #141b22 !important;
        box-shadow: none !important;
    }
    div[role="radiogroup"] > label {
        min-height: 44px !important;
        padding: 0 22px !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        border-width: 0 1px 0 0 !important;
        border-color: #2c3a46 !important;
        border-style: solid !important;
        border-radius: 0 !important;
        background: #141b22 !important;
        color: #9ab7d2 !important;
        box-shadow: none !important;
    }
    div[role="radiogroup"] > label:last-child {
        border-right: 0 !important;
    }
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {
        color: #9ab7d2 !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        line-height: 1 !important;
        margin: 0 !important;
        text-transform: uppercase !important;
        white-space: nowrap !important;
    }
    div[role="radiogroup"] > label [data-testid="stMarkdownContainer"] p::after {
        content: "";
        display: inline-block;
        width: 7px;
        height: 7px;
        margin-left: 9px;
        border-right: 2px solid #71879a;
        border-bottom: 2px solid #71879a;
        transform: rotate(45deg) translateY(-3px);
    }
    div[role="radiogroup"] > label:hover {
        background: #1b252e !important;
    }
    div[role="radiogroup"] > label:hover [data-testid="stMarkdownContainer"] p {
        color: #d7e6f5 !important;
    }
    div[role="radiogroup"] > label:hover [data-testid="stMarkdownContainer"] p::after {
        border-color: #9ab7d2;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        background: #24313c !important;
        border-color: #2c3a46 !important;
        box-shadow: inset 0 -3px 0 #00e054 !important;
    }
    div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    div[role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p::after {
        border-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ data and model
# Caching strategy:
#   @st.cache_data     for DataFrames and lookup dictionaries.
#   @st.cache_resource for fitted Surprise models, because fitting is slower.

@st.cache_data
def load_data():
    """Load Goodreads CSVs and build a book_id -> metadata lookup."""
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")

    # Some book metadata has UTF-8 text that was previously mis-decoded
    # (example: GrandPrÃ© instead of GrandPré). Repair those strings for display.
    def fix_text(value):
        if not isinstance(value, str):
            return value
        if not any(marker in value for marker in ["Ã", "Â", "â"]):
            return value
        try:
            return value.encode("latin1").decode("utf-8")
        except UnicodeError:
            return value

    text_cols = books.select_dtypes(include="object").columns
    for col in text_cols:
        books[col] = books[col].map(fix_text)

    # Keep IDs consistent for Surprise and dictionary lookups.
    books["book_id"] = books["book_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)

    book_lookup = books.set_index("book_id").to_dict(orient="index")
    return books, ratings, book_lookup


@st.cache_resource
def fit_cf_model(model_choice):
    """Fit the selected collaborative-filtering model on the full dataset."""
    if not SURPRISE_AVAILABLE:
        return {"model_type": "fallback", "model_choice": model_choice}

    _, ratings, _ = load_data()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()

    if model_choice == "Baseline":
        model = BaselineOnly(verbose=False)
    elif model_choice == "User-Based CF":
        model = KNNBasic(
            k=50,
            sim_options={"name": "pearson", "user_based": True},
            verbose=False,
        )
    elif model_choice == "Item-Based CF":
        model = KNNBasic(
            k=50,
            sim_options={"name": "pearson", "user_based": False},
            verbose=False,
        )
    else:
        model = SVD(
            n_factors=50,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.05,
            random_state=6604,
        )

    model.fit(trainset)
    return {"model_type": "surprise", "model": model}


@st.cache_data
def build_fallback_tables():
    """Build simple tables for the backup recommender if Surprise is unavailable."""
    books, ratings, _ = load_data()
    counts = ratings["book_id"].value_counts()
    avg_rating = ratings.groupby("book_id")["rating"].mean()
    global_mean = float(ratings["rating"].mean())
    user_mean = ratings.groupby("user_id")["rating"].mean()

    book_stats = (
        pd.DataFrame({
            "book_id": avg_rating.index,
            "avg_rating_project": avg_rating.values,
            "n_ratings": counts.reindex(avg_rating.index).values,
        })
        .merge(books[["book_id", "authors"]], on="book_id", how="left")
    )
    return book_stats, global_mean, user_mean


def fallback_predict(
    meta,
    avg_rating_project,
    n_ratings,
    global_mean,
    user_avg,
    liked_author_counts,
    liked_title_token_counts,
    model_choice,
):
    """Approximate a score when scikit-surprise is not installed.

    This is only for getting the Streamlit demo running on Python versions where
    Surprise cannot install. The formal modeling notebook still uses Surprise,
    as required by the assignment.
    """
    title = str(meta.get("title", ""))
    candidate_authors = set(a.strip() for a in str(meta.get("authors", "")).split(",") if a.strip())
    candidate_title_tokens = {
        token.strip("()[],:;#").lower()
        for token in title.replace("-", " ").split()
        if len(token.strip("()[],:;#")) >= 4
    }
    author_affinity = sum(liked_author_counts.get(author, 0) for author in candidate_authors)
    title_affinity = sum(liked_title_token_counts.get(token, 0) for token in candidate_title_tokens)

    # Convert raw overlap counts into bounded boosts so demo mode visibly changes
    # across model choices without letting one repeated author completely dominate.
    author_boost = min(0.85, 0.18 * author_affinity)
    title_boost = min(0.85, 0.12 * title_affinity)

    # Shrink book averages toward the global mean so tiny samples do not dominate.
    shrink_weight = n_ratings / (n_ratings + 25)
    book_component = (shrink_weight * avg_rating_project) + ((1 - shrink_weight) * global_mean)

    if model_choice == "Baseline":
        # Baseline is intentionally non-personalized: mostly book average with a
        # small support bonus for books with more ratings in the project sample.
        support_bonus = min(0.35, n_ratings / 300)
        score = book_component + support_bonus
    elif model_choice == "Item-Based CF":
        # Item-style demo logic: books get a boost when their titles/series words
        # overlap with books this reader rated highly.
        score = (0.70 * book_component) + (0.10 * user_avg) + title_boost
    elif model_choice == "SVD":
        # SVD-style demo logic: blend user tendency and item quality, with small
        # author/title affinities standing in for latent preference factors.
        score = (0.58 * book_component) + (0.22 * user_avg) + (0.10 * author_boost) + (0.10 * title_boost)
    else:
        # User-style demo logic: books get a boost when the reader previously
        # liked the same authors.
        score = (0.68 * book_component) + (0.16 * user_avg) + author_boost

    return min(5.0, max(1.0, score))


def recommend(model_bundle, user_id, min_ratings, top_n=10):
    """Return top-N candidate books for a user, excluding already-rated books."""
    books, ratings, book_lookup = load_data()

    counts = ratings["book_id"].value_counts()
    avg_rating = ratings.groupby("book_id")["rating"].mean()
    popular = set(counts[counts >= min_ratings].index)
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])

    if model_bundle["model_type"] == "fallback":
        _, global_mean, user_mean = build_fallback_tables()
        user_avg = float(user_mean.get(user_id, global_mean))
        liked_book_ids = set(ratings.loc[(ratings["user_id"] == user_id) & (ratings["rating"] >= 4), "book_id"])
        liked_author_counts = {}
        liked_title_token_counts = {}
        for liked_id in liked_book_ids:
            liked_meta = book_lookup.get(liked_id, {})
            authors = str(liked_meta.get("authors", ""))
            for author in [a.strip() for a in authors.split(",") if a.strip()]:
                liked_author_counts[author] = liked_author_counts.get(author, 0) + 1
            title = str(liked_meta.get("title", ""))
            for token in title.replace("-", " ").split():
                clean_token = token.strip("()[],:;#").lower()
                if len(clean_token) >= 4:
                    liked_title_token_counts[clean_token] = liked_title_token_counts.get(clean_token, 0) + 1
    else:
        global_mean = None
        user_avg = None
        liked_author_counts = {}
        liked_title_token_counts = {}

    scored = []
    for book_id in books["book_id"]:
        if book_id in seen or book_id not in popular:
            continue

        meta = book_lookup.get(book_id, {})
        if model_bundle["model_type"] == "surprise":
            predicted = model_bundle["model"].predict(user_id, book_id).est
        else:
            predicted = fallback_predict(
                meta,
                float(avg_rating.get(book_id, 0.0)),
                int(counts.get(book_id, 0)),
                global_mean,
                user_avg,
                liked_author_counts,
                liked_title_token_counts,
                model_bundle["model_choice"],
            )

        scored.append({
            "book_id": int(book_id),
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "year": meta.get("original_publication_year", ""),
            "language": meta.get("language_code", ""),
            "predicted": predicted,
            "n_ratings": int(counts.get(book_id, 0)),
            "avg_rating_project": float(avg_rating.get(book_id, 0.0)),
            "avg_rating_goodreads": meta.get("average_rating", ""),
            "image_url": meta.get("image_url", ""),
            "small_image_url": meta.get("small_image_url", ""),
        })

    scored.sort(key=lambda row: row["predicted"], reverse=True)
    return scored[:top_n]


def get_gemini_key():
    """Read the Gemini key without ever hard-coding it into the app."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def rerank_with_gemini(candidates, user_preference):
    """Ask Gemini to re-rank only the CF candidates and explain each pick."""
    from google import genai
    from pydantic import BaseModel, Field

    client = genai.Client(api_key=get_gemini_key())
    model = "gemini-2.5-flash-lite"

    class GeminiBookPick(BaseModel):
        book_id: int = Field(description="Exact book_id from the candidate list.")
        title: str = Field(description="Exact book title from the candidate list.")
        reason: str = Field(description="One short sentence explaining why this book fits the reader's request.")

    system_instruction = (
        "You are a Goodreads book concierge. Re-rank only the provided candidate books "
        "for the reader's request. Do not invent books. Use exact book_id values from "
        "the list. Return concise, friendly reasons that a non-technical reader can understand."
    )

    catalog = "\n".join(
        (
            f"- book_id={row['book_id']} | title={row['title']} | authors={row['authors']} | "
            f"year={format_book_year(row.get('year', '')) or 'N/A'} | "
            f"predicted_rating={round(row['predicted'], 3)} | "
            f"goodreads_average={row.get('avg_rating_goodreads', 'N/A')}"
        )
        for row in candidates
    )

    prompt = (
        f"Reader request: {user_preference}\n\n"
        f"Candidate books from the recommender model:\n{catalog}\n\n"
        "Return the candidates in best-to-worst order for the reader's request."
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": list[GeminiBookPick],
            "temperature": 0.2,
            "seed": 6604,
            "max_output_tokens": 1400,
        },
    )

    parsed = response.parsed
    by_id = {row["book_id"]: row for row in candidates}

    reranked = []
    for item in parsed or []:
        book_id = int(item["book_id"] if isinstance(item, dict) else item.book_id)
        if book_id in by_id:
            row = dict(by_id[book_id])
            row["llm_reason"] = item["reason"] if isinstance(item, dict) else item.reason
            reranked.append(row)
    return reranked


@st.cache_data
def dataset_overview():
    """Return dataset summary values for the rubric-oriented overview tab."""
    books, ratings, _ = load_data()
    ratings_per_user = ratings.groupby("user_id")["rating"].count()
    ratings_per_book = ratings.groupby("book_id")["rating"].count()
    return {
        "Books in catalog": f"{len(books):,}",
        "Ratings analyzed": f"{len(ratings):,}",
        "Users": f"{ratings['user_id'].nunique():,}",
        "Books with ratings": f"{ratings['book_id'].nunique():,}",
        "Avg ratings per user": f"{ratings_per_user.mean():.1f}",
        "Avg ratings per book": f"{ratings_per_book.mean():.1f}",
        "Rating scale": "1-5 stars",
    }


@st.cache_data
def model_comparison_table():
    """Return the final notebook metrics used to justify the app model."""
    return pd.DataFrame([
        {
            "Model": "UBCF Pearson k=50",
            "RMSE": 1.0268,
            "Precision@10": 0.6598,
            "Recall@10": 0.7939,
            "Status": "Selected for app - best Top-10 quality",
        },
        {
            "Model": "UBCF Pearson k=20",
            "RMSE": 1.0290,
            "Precision@10": 0.6594,
            "Recall@10": 0.7938,
            "Status": "Very close user-based alternative",
        },
        {
            "Model": "Popularity Baseline",
            "RMSE": 0.8423,
            "Precision@10": 0.6568,
            "Recall@10": 0.7912,
            "Status": "Best RMSE benchmark",
        },
        {
            "Model": "SVD 50 factors, reg=0.05",
            "RMSE": 0.8428,
            "Precision@10": 0.6565,
            "Recall@10": 0.7910,
            "Status": "Strong matrix-factorization benchmark",
        },
        {
            "Model": "IBCF Pearson k=50",
            "RMSE": 0.8642,
            "Precision@10": 0.6490,
            "Recall@10": 0.7805,
            "Status": "Best item-based CF model",
        },
    ])


def preference_tokens(text):
    stop_words = {
        "want", "book", "books", "read", "reading", "with", "that", "this",
        "from", "have", "something", "about", "strong", "story", "novel",
        "like", "recommend", "recommendation", "and", "the", "for",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if token not in stop_words
    }


def search_library_catalog(query, limit=5):
    """Find direct catalog matches for the user's search words."""
    tokens = preference_tokens(query)
    if not tokens:
        return []

    matches = []
    for _, row in books.iterrows():
        title = str(row.get("title", ""))
        authors = str(row.get("authors", ""))
        haystack = f"{title} {authors}".lower()
        token_hits = sum(1 for token in tokens if token in haystack)
        if token_hits == 0:
            continue

        ratings_count = pd.to_numeric(row.get("ratings_count", 0), errors="coerce")
        ratings_count = 0 if pd.isna(ratings_count) else int(ratings_count)
        matches.append({
            "Book": title,
            "Author(s)": authors,
            "Avg rating": row.get("average_rating", ""),
            "# ratings": ratings_count,
            "match_score": token_hits,
        })

    matches.sort(key=lambda item: (item["match_score"], item["# ratings"]), reverse=True)
    return matches[:limit]


@st.cache_data
def popular_reviewers(limit=5):
    """Create a Letterboxd-style reviewer list from Goodreads user activity."""
    user_stats = (
        ratings.groupby("user_id")["book_id"]
        .agg(books="nunique", reviews="count")
        .sort_values(["reviews", "books"], ascending=False)
        .head(limit)
        .reset_index()
    )
    avatar_urls = [
        "https://i.pravatar.cc/96?img=12",
        "https://i.pravatar.cc/96?img=32",
        "https://i.pravatar.cc/96?img=56",
        "https://i.pravatar.cc/96?img=47",
        "https://i.pravatar.cc/96?img=68",
        "https://i.pravatar.cc/96?img=5",
        "https://i.pravatar.cc/96?img=23",
        "https://i.pravatar.cc/96?img=41",
        "https://i.pravatar.cc/96?img=15",
        "https://i.pravatar.cc/96?img=60",
        "https://i.pravatar.cc/96?img=29",
        "https://i.pravatar.cc/96?img=50",
        "https://i.pravatar.cc/96?img=8",
        "https://i.pravatar.cc/96?img=36",
        "https://i.pravatar.cc/96?img=64",
    ]
    display_names = [
        "Maya Chen",
        "Jordan Brooks",
        "Sofia Alvarez",
        "Ethan Walker",
        "Nina Patel",
        "Lucas Bennett",
        "Avery Morgan",
        "Grace Kim",
        "Marcus Rivera",
        "Lena Thompson",
        "Noah Williams",
        "Priya Shah",
        "Isabella Carter",
        "Daniel Park",
        "Amelia Foster",
    ]
    rows = []
    for index, row in user_stats.iterrows():
        user_id_value = int(row["user_id"])
        name = display_names[index % len(display_names)]
        rows.append({
            "name": name,
            "user_id": user_id_value,
            "books": int(row["books"]),
            "reviews": int(row["reviews"]),
            "avatar": avatar_urls[index % len(avatar_urls)],
        })
    return rows


def local_preference_rank(candidates, user_preference):
    """Rank model candidates against the user's words when Gemini is unavailable."""
    tokens = preference_tokens(user_preference)
    preference_aliases = {
        "fantasy": {"magic", "wizard", "dragon", "king", "queen", "chronicle", "narnia", "hobbit", "rings"},
        "thriller": {"mystery", "murder", "crime", "detective", "secret", "spy", "suspense"},
        "romance": {"love", "heart", "wedding", "wife", "husband", "bride", "romance"},
        "classic": {"classic", "austen", "tolstoy", "dickens", "orwell", "bronte", "shakespeare"},
        "funny": {"funny", "humor", "comic", "laugh", "witty", "diary"},
    }
    expanded_tokens = set(tokens)
    for token in list(tokens):
        expanded_tokens.update(preference_aliases.get(token, set()))

    ranked = []
    for row in candidates:
        haystack = f"{row['title']} {row['authors']} {row.get('year', '')}".lower()
        token_hits = sum(1 for token in expanded_tokens if token in haystack)
        score = (0.65 * row["predicted"]) + (0.20 * row["avg_rating_project"]) + (0.55 * token_hits)
        ranked.append({**row, "preference_score": score, "token_hits": token_hits})
    ranked.sort(key=lambda item: item["preference_score"], reverse=True)
    return ranked


# ------------------------------------------------------------------- UI ----
@st.cache_data
def load_logo_data_url():
    logo_path = os.path.join(os.path.dirname(__file__), "GoodReads_logo.png")
    with open(logo_path, "rb") as logo_file:
        encoded = base64.b64encode(logo_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def stat_box(value, label, user_specific=False):
    extra_class = " user-stat" if user_specific else ""
    chip = "<div class='stat-chip'>User-specific</div>" if user_specific else ""
    html = (
        f"<div class='stat-box{extra_class}'>"
        f"{chip}"
        f"<div class='stat-value'>{value}</div>"
        f"<div class='stat-label'>{label}</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def format_book_year(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def format_card_rating(value):
    if pd.isna(value) or str(value).strip() == "":
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def clean_display_text(value, fallback="Unknown"):
    text = str(value or "").strip()
    bad_markers = ["Ã", "Â", "Ø", "Ù", "�", "\x9c", "\x9d"]
    if not text or any(marker in text for marker in bad_markers):
        return fallback
    return text


def is_missing_cover_url(value):
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    return not text or "nophoto" in text or "no_photo" in text


def generated_book_cover(title, authors):
    title_text = escape(clean_display_text(title, "Untitled book"))
    author_text = escape(clean_display_text(authors, "Unknown author"))
    initials = "".join(word[0] for word in re.findall(r"[A-Za-z]+", title_text)[:3]).upper() or "B"
    return f"""
        <div class="generated-cover">
            <div class="generated-cover-mark">{initials}</div>
            <div class="generated-cover-title">{title_text}</div>
            <div class="generated-cover-author">{author_text}</div>
        </div>
    """


def make_book_description(row):
    year = format_book_year(row.get("year", ""))
    author_text = str(row.get("authors", "") or "Unknown author")
    goodreads_avg = format_card_rating(row.get("avg_rating_goodreads", ""))
    year_text = f"Published in {year}. " if year else ""
    return (
        f"{year_text}A Goodreads title by {author_text} with an average rating of "
        f"{goodreads_avg}. Recommended because this reader's rating pattern suggests a strong fit."
    )


def render_book_card(row, rank, reason=None):
    title = escape(str(row.get("title", "Untitled")))
    authors = escape(str(row.get("authors", "Unknown author")))
    image_url = row.get("image_url", "") or row.get("small_image_url", "")
    description = escape(make_book_description(row))
    reason_html = (
        f"<div class='book-description'><b>Why it fits:</b> {escape(str(reason))}</div>"
        if reason else ""
    )
    if not is_missing_cover_url(image_url):
        cover_html = (
            f"<img class='book-cover' src='{escape(str(image_url))}' "
            f"alt='Cover of {title}'>"
        )
    else:
        cover_html = generated_book_cover(title, authors)

    st.markdown(
        f"""
        <div class="book-card book-card-with-cover">
            <div class="book-cover-wrap" data-title="{title}">{cover_html}</div>
            <div class="book-card-body">
                <div class="book-rank">Rank {rank}</div>
                <div class="book-title">{title}</div>
                <div class="book-meta">{authors}</div>
                <div class="book-description">{description}</div>
                <div class="book-score-row">
                    Predicted rating: <b>{float(row.get("predicted", 0.0)):.2f}</b> |
                    Project avg: <b>{float(row.get("avg_rating_project", 0.0)):.2f}</b> |
                    Ratings in sample: <b>{int(row.get("n_ratings", 0))}</b>
                </div>
                {reason_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_catalog_grid(catalog_rows, columns_per_row=5):
    rows = list(catalog_rows.iterrows())
    for start in range(0, len(rows), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, (_, row) in zip(cols, rows[start:start + columns_per_row]):
            with col:
                render_catalog_card(row)


def render_catalog_card(row):
    title = escape(clean_display_text(row.get("title", ""), "Untitled book"))
    authors = escape(clean_display_text(row.get("authors", ""), "Unknown author"))
    year = format_book_year(row.get("original_publication_year", ""))
    image_url = row.get("image_url", "")
    if pd.isna(image_url) or str(image_url).strip() == "":
        image_url = row.get("small_image_url", "")
    if pd.isna(image_url):
        image_url = ""
    avg_rating = format_card_rating(row.get("average_rating", ""))
    ratings_count = pd.to_numeric(row.get("ratings_count", 0), errors="coerce")
    ratings_count = 0 if pd.isna(ratings_count) else int(ratings_count)
    cover_html = (
        f"<img class='catalog-cover' src='{escape(str(image_url))}' alt='Cover of {title}'>"
        if not is_missing_cover_url(image_url) else
        generated_book_cover(title, authors)
    )
    card_html = f"""
        <div class="catalog-card" data-title="{title}">
            <div class="catalog-cover-frame">{cover_html}</div>
            <div class="catalog-title">{title}</div>
            <div class="catalog-meta">{authors}</div>
            <div class="catalog-meta">{year if year else "Publication year unavailable"}</div>
            <div class="catalog-stats">
                <span><b>{avg_rating}</b> rating</span>
                <span>{ratings_count:,} reads</span>
            </div>
        </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_popular_reviewers_panel():
    show_all_key = "show_all_reviewers"
    if show_all_key not in st.session_state:
        st.session_state[show_all_key] = False

    reviewer_limit = 15 if st.session_state[show_all_key] else 5
    heading_col, more_col = st.columns([1, 0.22])
    with heading_col:
        st.markdown("<div class='reviewer-heading'><span>Popular Reviewers</span></div>", unsafe_allow_html=True)
    with more_col:
        if st.button(
            "More" if not st.session_state[show_all_key] else "Less",
            key="toggle_reviewers",
            type="tertiary",
        ):
            st.session_state[show_all_key] = not st.session_state[show_all_key]
            st.rerun()

    for reviewer in popular_reviewers(reviewer_limit):
        st.markdown(
            f"""
            <div class="reviewer-row">
                <img class="reviewer-avatar" src="{escape(reviewer["avatar"])}" alt="{escape(reviewer["name"])} avatar">
                <div>
                    <div class="reviewer-name">{escape(reviewer["name"])}</div>
                    <div class="reviewer-id">User {reviewer["user_id"]}</div>
                    <div class="reviewer-stats">
                        {reviewer["books"]:,} books, {reviewer["reviews"]:,} reviews
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="cant-find-panel">
            <div class="cant-find-title">Can't find a book?</div>
            <div class="cant-find-copy">
                Help keep Bookboxd up to date.<br>
                Find out how to <b>add or edit a book</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_books_pagination(current_page, total_pages):
    st.markdown(
        f"<div class='pagination-note'>Page {current_page:,} of {total_pages:,}</div>",
        unsafe_allow_html=True,
    )
    if current_page <= 5:
        visible_pages = list(range(1, min(5, total_pages) + 1))
    else:
        start_page = max(1, current_page - 2)
        end_page = min(total_pages, current_page + 2)
        visible_pages = list(range(start_page, end_page + 1))

    cols = st.columns([0.22, 0.22, 0.22, 0.22, 0.22, 0.16, 0.36, 0.22, 0.32, 0.62, 3.4])
    for col, page in zip(cols[:5], visible_pages):
        with col:
            if st.button(
                str(page),
                key=f"books_page_button_{page}_{total_pages}",
                disabled=(page == current_page),
                use_container_width=True,
            ):
                st.session_state["books_page_number"] = page
                st.rerun()

    with cols[5]:
        if total_pages > 5:
            st.markdown("<div class='pagination-ellipsis'>...</div>", unsafe_allow_html=True)

    with cols[6]:
        if total_pages not in visible_pages:
            if st.button(str(total_pages), key=f"books_last_page_{total_pages}", use_container_width=True):
                st.session_state["books_page_number"] = total_pages
                st.rerun()

    with cols[7]:
        if st.button("▶", key="books_next_page", disabled=(current_page >= total_pages), use_container_width=True):
            st.session_state["books_page_number"] = min(total_pages, current_page + 1)
            st.rerun()

    with cols[8]:
        st.markdown("<div class='pagination-note'>Go to</div>", unsafe_allow_html=True)

    with cols[9]:
        go_to_page = st.number_input(
            "Go to page",
            min_value=1,
            max_value=total_pages,
            value=current_page,
            step=1,
            label_visibility="collapsed",
            key="books_go_to_page",
        )
        if int(go_to_page) != current_page:
            st.session_state["books_page_number"] = int(go_to_page)
            st.rerun()


def render_dark_table(rows):
    html_rows = []
    for row in rows:
        html_rows.append(
            "<tr>"
            f"<td class='rank-cell'>{row['Rank']}</td>"
            f"<td>{escape(str(row['Book']))}</td>"
            f"<td>{escape(str(row['Author(s)']))}</td>"
            f"<td>{row['Predicted']}</td>"
            f"<td>{row['Avg rating']}</td>"
            f"<td>{row['# ratings']}</td>"
            "</tr>"
        )

    st.markdown(
        """
        <table class="dark-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Book</th>
                    <th>Author(s)</th>
                    <th>Predicted</th>
                    <th>Avg</th>
                    <th># Ratings</th>
                </tr>
            </thead>
            <tbody>
        """
        + "\n".join(html_rows)
        + """
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_model_guide(selected_model):
    blocks = {
        "Baseline": (
            "Simple Baseline",
            "Recommend books that everybody likes.",
            "Harry Potter, Lord of the Rings, The Hunger Games. This is our benchmark.",
        ),
        "User-Based CF": (
            "User-Based Collaborative Filtering",
            "Find users similar to you.",
            "If you and John both love fantasy books, and John also loved Mistborn, the model recommends Mistborn to you.",
        ),
        "Item-Based CF": (
            "Item-Based Collaborative Filtering",
            "Find books similar to books you already liked.",
            "You loved Harry Potter. Many people who liked Harry Potter also liked Percy Jackson. Recommend Percy Jackson.",
        ),
        "SVD": (
            "SVD Matrix Factorization",
            "Learn hidden reader and book patterns from the rating matrix.",
            "If the model learns you enjoy complex fantasy and another book has similar latent taste signals, SVD can recommend it even without exact author or title overlap.",
        ),
    }
    heading, idea, example = blocks[selected_model]
    html_block = (
        "<div class='guide-block active-guide'>"
        f"<div class='guide-heading'>{heading}</div>"
        f"<b>Idea:</b> {idea}"
        f"<div class='guide-example'><b>Example:</b> {example}</div>"
        "</div>"
    )

    st.markdown(
        f"<div class='model-guide'><div class='guide-title'>Selected Model Guide</div>{html_block}</div>",
        unsafe_allow_html=True,
    )


MODEL_EXPLANATIONS = {
    "User-Based CF": (
        "User-Based Collaborative Filtering",
        "Find users similar to you.",
    ),
    "Baseline": (
        "Simple Baseline",
        "Recommend books that everybody likes.",
    ),
    "Item-Based CF": (
        "Item-Based Collaborative Filtering",
        "Find books similar to books you already liked.",
    ),
    "SVD": (
        "SVD Matrix Factorization",
        "Learn hidden reader and book patterns from the rating matrix.",
    ),
}


def style_plot(fig, title=None):
    """Apply the app's dark visual system to Plotly figures."""
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#151c23",
        "font": {"color": "#c8d7e6"},
        "margin": {"l": 20, "r": 20, "t": 58, "b": 30},
        "title_font": {"color": "#ffffff", "size": 18},
        "legend_title_text": "",
    }
    if title is not None:
        layout["title"] = title
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="rgba(154,183,210,0.18)")
    fig.update_yaxes(gridcolor="rgba(154,183,210,0.18)")
    return fig


def add_bar_labels(fig, orientation="v"):
    """Show count/value labels on bar charts in a readable position."""
    if orientation == "h":
        fig.update_traces(texttemplate="%{x}", textposition="outside", cliponaxis=False)
        fig.update_layout(margin={"l": 20, "r": 90, "t": 58, "b": 30})
    else:
        fig.update_traces(texttemplate="%{y}", textposition="outside", cliponaxis=False)
        fig.update_layout(margin={"l": 20, "r": 20, "t": 58, "b": 30})
    return fig


with st.spinner("Loading Goodreads data..."):
    books, ratings, book_lookup = load_data()

logo_data_url = load_logo_data_url()

st.markdown(
    """
    <div class="letterboxd-nav">
        <div class="brand-wrap">
            <div class="brand-dots">
                <span class="brand-dot orange"></span>
                <span class="brand-dot green"></span>
                <span class="brand-dot blue"></span>
            </div>
            <div class="brand-name">Bookboxd</div>
        </div>
        <div class="letterboxd-links">
            <span>Sign In</span>
            <span>Create Account</span>
            <span>Books</span>
            <span>Lists</span>
            <span>Readers</span>
            <span>Journal</span>
            <span class="letterboxd-search"></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-content">
            <div class="hero-copy">
                <div class="main-title">
                    Track books you have read.<br>
                    Save those you want to read.<br>
                    Tell your friends what's good.
                </div>
                <div class="subtitle">
                    The social recommendation app for book lovers, built with collaborative filtering and Gemini.
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="letterboxd-section-kicker">Bookboxd lets you...</div>
    <div class="letterboxd-feature-grid">
        <div class="letterboxd-feature-card">
            <div class="letterboxd-feature-icon">eye</div>
            <div>Keep track of every book you have read, or start from the day you join.</div>
        </div>
        <div class="letterboxd-feature-card">
            <div class="letterboxd-feature-icon">heart</div>
            <div>Show some love for favorite books, lists and recommendations with a like.</div>
        </div>
        <div class="letterboxd-feature-card">
            <div class="letterboxd-feature-icon">list</div>
            <div>Write and share reviews, and follow readers whose taste you trust.</div>
        </div>
        <div class="letterboxd-feature-card">
            <div class="letterboxd-feature-icon">star</div>
            <div>Rate each book on a five-star scale to record and share your reaction.</div>
        </div>
        <div class="letterboxd-feature-card">
            <div class="letterboxd-feature-icon">date</div>
            <div>Keep a reading diary and see how your taste changes over time.</div>
        </div>
        <div class="letterboxd-feature-card">
            <div class="letterboxd-feature-icon">grid</div>
            <div>Compile and share lists of books on any topic and keep a want-to-read shelf.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Books",
    "Dataset & Evaluation",
    "Reader Profile",
    "Recommendation Deck",
    "AI Re-Ranking",
    "Preference Search",
    "Model Notes",
]
active_page = st.radio(
    "App section",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="active_page",
)

user_ids = sorted(ratings["user_id"].unique())
min_id, max_id = int(user_ids[0]), int(user_ids[-1])
controls_pages = {"Recommendation Deck", "AI Re-Ranking", "Reader Profile"}

if "selected_user_id" not in st.session_state:
    st.session_state["selected_user_id"] = int(user_ids[0])
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "User-Based CF"

MODEL_DROPDOWN_LABELS = {
    "Baseline": "1. Simple Baseline",
    "User-Based CF": "2. User-Based CF",
    "Item-Based CF": "3. Item-Based CF",
    "SVD": "4. SVD Matrix Factorization",
}
MODEL_DROPDOWN_ORDER = ["Baseline", "User-Based CF", "Item-Based CF", "SVD"]

if active_page in controls_pages:
    with st.sidebar:
        user_id = st.selectbox(
            "User ID",
            user_ids,
            index=user_ids.index(st.session_state["selected_user_id"]),
            help="Search or scroll to choose one of the valid users in Ratings.csv.",
        )
        st.session_state["selected_user_id"] = int(user_id)
        st.caption(f"Valid ID range: {min_id}-{max_id} | {len(user_ids):,} users")

        slider_user_id = int(user_id)
        book_rating_counts = ratings["book_id"].value_counts()
        seen_for_slider = set(ratings.loc[ratings["user_id"] == slider_user_id, "book_id"])
        unseen_counts = book_rating_counts[~book_rating_counts.index.isin(seen_for_slider)]

        if unseen_counts.empty:
            st.warning("This user has no unseen books available for recommendations.")
            st.stop()

        max_min_ratings = int(unseen_counts.max())
        default_min_ratings = min(20, max_min_ratings)
        current_min_ratings = min(
            int(st.session_state.get("min_ratings", default_min_ratings)),
            max_min_ratings,
        )

        model_choice = st.selectbox(
            "Collaborative-filtering model",
            MODEL_DROPDOWN_ORDER,
            index=MODEL_DROPDOWN_ORDER.index(st.session_state["model_choice"]),
            format_func=lambda model: MODEL_DROPDOWN_LABELS[model],
            help="Use the modeling notebook to justify which model should be deployed.",
        )
        st.session_state["model_choice"] = model_choice
        render_sidebar_model_guide(model_choice)

        min_ratings = st.slider(
            "Minimum ratings per book",
            min_value=1,
            max_value=max_min_ratings,
            value=current_min_ratings,
            step=1,
            key="min_ratings",
            help="Hide rarely-rated books so the list is based on more stable signals.",
        )
        available_candidates = int((unseen_counts >= min_ratings).sum())
        st.caption(f"{available_candidates:,} unseen books pass this filter.")

        max_candidates = max(1, available_candidates)
        default_top_n = min(10, max_candidates)
        current_top_n = min(
            int(st.session_state.get("top_n", default_top_n)),
            max_candidates,
        )
        top_n = st.slider(
            "Number of candidates",
            min_value=1,
            max_value=max_candidates,
            value=current_top_n,
            step=1,
            key="top_n",
        )

        if not SURPRISE_AVAILABLE:
            st.markdown(
                """
                <div class="demo-pill">
                    Demo mode: backup scorer is active because scikit-surprise is not installed.
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    user_id = int(st.session_state["selected_user_id"])
    model_choice = st.session_state["model_choice"]
    book_rating_counts = ratings["book_id"].value_counts()
    seen_for_slider = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    unseen_counts = book_rating_counts[~book_rating_counts.index.isin(seen_for_slider)]
    min_ratings = int(st.session_state.get("min_ratings", 20))
    min_ratings = max(1, min(min_ratings, int(unseen_counts.max())))
    available_candidates = int((unseen_counts >= min_ratings).sum())
    top_n = int(st.session_state.get("top_n", 10))
    top_n = max(1, min(top_n, max(1, available_candidates)))

if False:
    # Legacy sidebar block kept unreachable while the app uses page-aware controls above.
    user_id = st.selectbox(
        "User ID",
        user_ids,
        index=0,
        help="Search or scroll to choose one of the valid users in Ratings.csv.",
    )
    st.caption(f"Valid ID range: {min_id}-{max_id} | {len(user_ids):,} users")

    slider_user_id = int(user_id)

    book_rating_counts = ratings["book_id"].value_counts()
    seen_for_slider = set(ratings.loc[ratings["user_id"] == slider_user_id, "book_id"])
    unseen_counts = book_rating_counts[~book_rating_counts.index.isin(seen_for_slider)]

    if unseen_counts.empty:
        st.warning("This user has no unseen books available for recommendations.")
        st.stop()

    max_min_ratings = int(unseen_counts.max())
    default_min_ratings = min(20, max_min_ratings)
    current_min_ratings = min(
        int(st.session_state.get("min_ratings", default_min_ratings)),
        max_min_ratings,
    )

    model_choice = st.selectbox(
        "Collaborative-filtering model",
        MODEL_DROPDOWN_ORDER,
        format_func=lambda model: MODEL_DROPDOWN_LABELS[model],
        help="Use the modeling notebook to justify which model should be deployed.",
    )
    render_sidebar_model_guide(model_choice)

    min_ratings = st.slider(
        "Minimum ratings per book",
        min_value=1,
        max_value=max_min_ratings,
        value=current_min_ratings,
        step=1,
        key="min_ratings",
        help="Hide rarely-rated books so the list is based on more stable signals.",
    )
    available_candidates = int((unseen_counts >= min_ratings).sum())
    st.caption(f"{available_candidates:,} unseen books pass this filter.")

    max_candidates = max(1, available_candidates)
    default_top_n = min(10, max_candidates)
    current_top_n = min(
        int(st.session_state.get("top_n", default_top_n)),
        max_candidates,
    )
    top_n = st.slider(
        "Number of candidates",
        min_value=1,
        max_value=max_candidates,
        value=current_top_n,
        step=1,
        key="top_n",
    )

    if not SURPRISE_AVAILABLE:
        st.markdown(
            """
            <div class="demo-pill">
                Demo mode: backup scorer is active because scikit-surprise is not installed.
            </div>
            """,
            unsafe_allow_html=True,
        )


user_id = int(user_id)

with st.spinner(f"Fitting {model_choice} model..."):
    cf_model = fit_cf_model(model_choice)

recs = recommend(cf_model, user_id, min_ratings, top_n=top_n)
if not recs:
    st.warning("No books pass the filter - try lowering the minimum rating count.")
    st.stop()

user_history = (
    ratings[ratings["user_id"] == user_id]
    .merge(books, on="book_id")
    [["title", "authors", "rating"]]
    .sort_values("rating", ascending=False)
    .reset_index(drop=True)
)
user_history.columns = ["Book", "Author(s)", "Their rating"]
user_history.index = user_history.index + 1

avg_user_rating = round(float(ratings.loc[ratings["user_id"] == user_id, "rating"].mean()), 2)
top_author = (
    user_history["Author(s)"]
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .value_counts()
    .index[0]
    if len(user_history) else "N/A"
)

default_user_preference = (
    f"I want a book that fits user {user_id}'s reading history, "
    f"especially authors or styles similar to {top_author}."
)
if st.session_state.get("preference_user_id") != user_id:
    st.session_state["preference_user_id"] = user_id
    st.session_state["user_preference"] = default_user_preference
    st.session_state["search_preference"] = "I want a fantasy book with strong world-building and memorable characters."

def render_user_summary_cards():
    c1, c2 = st.columns(2)
    with c1:
        stat_box(f"{len(user_history):,}", f"Ratings by user {user_id}", user_specific=True)
    with c2:
        stat_box(avg_user_rating, "Reader avg rating", user_specific=True)


def render_reader_profile_header():
    avatar_id = (int(user_id) % 70) + 1
    avatar_url = f"https://i.pravatar.cc/184?img={avatar_id}"
    st.markdown(
        f"""
        <div class="reader-profile-card">
            <img class="reader-profile-avatar" src="{avatar_url}" alt="User {user_id} profile image">
            <div>
                <div class="reader-profile-name">User {user_id}</div>
                <div class="reader-profile-meta">
                    {len(user_history):,} books rated | Average rating {avg_user_rating} | Favorite author signal: {escape(str(top_author))}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reader_context(label):
    st.markdown(
        f"""
        <div class="prompt-panel">
            <div class="prompt-label">{label}</div>
            <div class="prompt-help">
                Current reader: user {user_id} | Model: {model_choice} | Minimum book ratings: {min_ratings} | Candidates: {top_n}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gemini_personalizer(candidates, key_prefix):
    st.markdown("<div class='section-title'>Personalize with Gemini</div>", unsafe_allow_html=True)
    st.markdown(
        "Gemini re-ranks the model's recommendations for your current mood. "
        "It only chooses from books already produced by the recommender."
    )

    mood_key = f"{key_prefix}_gemini_mood"
    results_key = f"{key_prefix}_gemini_results"
    if mood_key not in st.session_state:
        st.session_state[mood_key] = "a quiet summer evening"

    mood = st.text_input(
        "What are you in the mood for?",
        key=mood_key,
        placeholder="Example: a quiet summer evening",
    )

    gemini_key = get_gemini_key()
    if not gemini_key:
        st.info(
            "Add GEMINI_API_KEY as an environment variable or in .streamlit/secrets.toml "
            "to enable Gemini personalization."
        )
        return

    if st.button("Re-rank with Gemini", use_container_width=True, key=f"{key_prefix}_gemini_button"):
        try:
            with st.spinner("Asking Gemini to re-rank..."):
                st.session_state[results_key] = rerank_with_gemini(candidates, mood)
        except Exception as exc:
            st.session_state.pop(results_key, None)
            st.error(f"Gemini re-ranking failed: {exc}")

    gemini_results = st.session_state.get(results_key, [])
    if gemini_results:
        st.markdown("<div class='section-title'>Gemini Personalized Ranking</div>", unsafe_allow_html=True)
        for rank, row in enumerate(gemini_results, start=1):
            render_book_card(row, rank, row.get("llm_reason", ""))


if active_page == "Books":
    st.markdown("<div class='section-title'>Books</div>", unsafe_allow_html=True)
    st.markdown(
        "Browse the Goodreads catalog by popularity, rating, or publication year. Use search to find a title or author."
    )

    page_size = 15
    browse_left, browse_right = st.columns([1.1, 1.4])
    with browse_left:
        sort_choice = st.selectbox(
            "Browse by",
            ["All Books", "Popular", "Highest Rated", "Newest", "Oldest"],
            index=0,
            key="books_sort_choice",
        )
    with browse_right:
        book_search = st.text_input(
            "Find a book",
            placeholder="Search title or author",
            key="books_search",
        )

    catalog = books.copy()
    catalog["ratings_count_num"] = pd.to_numeric(catalog["ratings_count"], errors="coerce").fillna(0)
    catalog["average_rating_num"] = pd.to_numeric(catalog["average_rating"], errors="coerce").fillna(0)
    catalog["year_num"] = pd.to_numeric(catalog["original_publication_year"], errors="coerce")
    catalog["clean_title"] = catalog["title"].map(lambda value: clean_display_text(value, ""))
    catalog = catalog[catalog["clean_title"] != ""].copy()

    if book_search.strip():
        search_text = book_search.strip().lower()
        catalog = catalog[
            catalog["clean_title"].astype(str).str.lower().str.contains(search_text, na=False)
            | catalog["authors"].map(lambda value: clean_display_text(value, "")).astype(str).str.lower().str.contains(search_text, na=False)
        ].copy()

    if sort_choice == "All Books":
        catalog = catalog.sort_values("clean_title", ascending=True)
        caption = f"Showing {len(catalog):,} books in the catalog."
    elif sort_choice == "Popular":
        catalog = catalog.sort_values("ratings_count_num", ascending=False)
        caption = f"Popular books ranked by rating activity across {len(catalog):,} titles."
    elif sort_choice == "Highest Rated":
        catalog = catalog.sort_values(["average_rating_num", "ratings_count_num"], ascending=False)
        caption = f"Highest-rated books across {len(catalog):,} titles, with popularity used as the tiebreaker."
    elif sort_choice == "Newest":
        catalog = catalog.sort_values(["year_num", "ratings_count_num"], ascending=[False, False])
        caption = f"Newest books first across {len(catalog):,} titles."
    else:
        catalog = catalog.sort_values(["year_num", "ratings_count_num"], ascending=[True, False])
        caption = f"Oldest publication dates first across {len(catalog):,} titles."

    total_pages = max(1, int((len(catalog) + page_size - 1) // page_size))
    if "books_page_number" not in st.session_state:
        st.session_state["books_page_number"] = 1
    st.session_state["books_page_number"] = min(st.session_state["books_page_number"], total_pages)
    page_number = st.session_state["books_page_number"]
    start = (page_number - 1) * page_size
    end = start + page_size
    grid_catalog = catalog.iloc[start:end]

    st.caption(caption)
    if catalog.empty:
        st.warning("No books matched that search.")
    else:
        catalog_main, catalog_side = st.columns([2.35, 1])
        with catalog_main:
            st.caption(
                f"Page {page_number:,} of {total_pages:,} | "
                f"Showing books {start + 1:,}-{min(end, len(catalog)):,} of {len(catalog):,}"
            )
            render_catalog_grid(grid_catalog, columns_per_row=3)
            render_books_pagination(page_number, total_pages)
        with catalog_side:
            render_popular_reviewers_panel()


if active_page == "Recommendation Deck":
    render_user_summary_cards()
    st.markdown(f"<div class='section-title'>Top {top_n} Recommendations for User {user_id}</div>", unsafe_allow_html=True)

    rec_df = pd.DataFrame([
        {
            "Book": row["title"],
            "Author(s)": row["authors"],
            "# ratings": row["n_ratings"],
            "Avg rating": round(row["avg_rating_project"], 2),
            "Predicted": round(row["predicted"], 2),
        }
        for row in recs
    ])
    rec_df.index = rec_df.index + 1
    rec_df.index.name = "Rank"

    left, right = st.columns([1.15, 1])
    with left:
        table_rows = rec_df.reset_index().to_dict(orient="records")
        render_dark_table(table_rows)
    with right:
        for rank, row in enumerate(recs[:5], start=1):
            render_book_card(row, rank)

    st.markdown("<hr>", unsafe_allow_html=True)
    render_gemini_personalizer(recs, "recommendation_deck")

    if PLOTLY_AVAILABLE:
        st.markdown("<div class='section-title'>Recommendation Signals</div>", unsafe_allow_html=True)
        viz_df = rec_df.reset_index().copy()
        viz_df["Short Book"] = viz_df["Book"].str.slice(0, 34)

        v1, v2 = st.columns(2)
        with v1:
            fig = px.bar(
                viz_df.sort_values("Predicted"),
                x="Predicted",
                y="Short Book",
                orientation="h",
                color="Predicted",
                color_continuous_scale=["#00e054", "#40bcf4", "#ff8000"],
                title="Predicted Rating by Recommended Book",
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = add_bar_labels(fig, orientation="h")
            st.plotly_chart(style_plot(fig), use_container_width=True)
        with v2:
            fig = px.scatter(
                viz_df,
                x="# ratings",
                y="Predicted",
                size="Avg rating",
                color="Avg rating",
                hover_name="Book",
                color_continuous_scale=["#00e054", "#40bcf4", "#ff8000"],
                title="Prediction vs. Rating Support",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_plot(fig), use_container_width=True)

if active_page == "AI Re-Ranking":
    render_user_summary_cards()
    st.markdown("<div class='section-title'>AI Re-Ranking Layer</div>", unsafe_allow_html=True)
    st.markdown(
        f"Current candidate set for **user {user_id}** from **{model_choice}**. "
        "Gemini re-ranks this list only; it does not create new books."
    )

    preview_left, preview_right = st.columns([1, 1])
    with preview_left:
        st.markdown("<div class='section-title'>CF Candidates Sent to AI</div>", unsafe_allow_html=True)
        for rank, row in enumerate(recs[:5], start=1):
            render_book_card(row, rank)
    with preview_right:
        if PLOTLY_AVAILABLE:
            ai_preview_df = pd.DataFrame([
                {
                    "Book": row["title"][:34],
                    "Predicted": round(row["predicted"], 2),
                    "# ratings": row["n_ratings"],
                    "Avg rating": round(row["avg_rating_project"], 2),
                }
                for row in recs
            ])
            fig = px.bar(
                ai_preview_df.sort_values("Predicted"),
                x="Predicted",
                y="Book",
                orientation="h",
                color="Predicted",
                color_continuous_scale=["#00e054", "#40bcf4", "#ff8000"],
                title=f"User {user_id} Candidate Scores",
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = add_bar_labels(fig, orientation="h")
            st.plotly_chart(style_plot(fig), use_container_width=True)

    st.markdown(
        """
        <div class="prompt-panel">
            <div class="prompt-label">User Preference</div>
            <div class="prompt-help">
                Describe the reading mood, genre, or style you want Gemini to use when re-ranking the candidate books.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    preference = st.text_area(
        "User Preference",
        key="user_preference",
        label_visibility="collapsed",
        help="The LLM re-ranks the collaborative-filtering candidates above; it does not invent new books.",
    )

    gemini_key = get_gemini_key()
    if not gemini_key:
        st.info(
            "Add GEMINI_API_KEY as an environment variable or in .streamlit/secrets.toml "
            "to enable Gemini re-ranking."
        )
    elif st.button("Re-rank with Gemini", use_container_width=True):
        try:
            with st.spinner("Re-ranking candidates with Gemini..."):
                reranked = rerank_with_gemini(recs, preference)

            if not reranked:
                st.warning("Gemini returned no usable candidates. Try again with a shorter preference.")
            else:
                for rank, row in enumerate(reranked, start=1):
                    render_book_card(row, rank, row.get("llm_reason", ""))
        except Exception as exc:
            st.error(f"Gemini re-ranking failed: {exc}")

if active_page == "Preference Search":
    st.markdown("<div class='section-title'>Preference Search Across Models</div>", unsafe_allow_html=True)
    st.markdown(
        "Choose a reading preference or write your own prompt. The app then shows matching "
        "recommendations from all three models in separate result boxes."
    )

    st.markdown(
        """
        <div class="prompt-panel">
            <div class="prompt-label">Tell the AI what you want to read</div>
            <div class="prompt-help">
                Pick an option below or write your own prompt. The search stays inside books produced by the recommendation models.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    preference_options = {
        "Fantasy": "I want a fantasy book with strong world-building and memorable characters.",
        "Thriller": "I want a suspenseful thriller or mystery with twists and a fast pace.",
        "Romance": "I want an emotional romance or relationship-driven story.",
        "Classic": "I want a classic or literary book with strong writing and lasting influence.",
        "Funny": "I want a funny, light, witty book that is easy to enjoy.",
    }
    if "preference_search_ready" not in st.session_state:
        st.session_state["preference_search_ready"] = False
    if "selected_preference_label" not in st.session_state:
        st.session_state["selected_preference_label"] = ""

    st.markdown("<div class='field-label'>Choose a quick option</div>", unsafe_allow_html=True)
    preset_labels = list(preference_options.keys())
    preset_index = (
        preset_labels.index(st.session_state["selected_preference_label"])
        if st.session_state["selected_preference_label"] in preset_labels
        else None
    )
    preset_choice = st.radio(
        "Choose a quick option",
        preset_labels,
        index=preset_index,
        horizontal=True,
        label_visibility="collapsed",
        key="preference_preset_choice",
    )
    if preset_choice:
        st.session_state["search_preference"] = preference_options[preset_choice]
        st.session_state["selected_preference_label"] = preset_choice
        st.session_state["preference_search_ready"] = True

    search_preference = st.text_area(
        "Or write your own prompt",
        key="search_preference",
        help="Example: I want a fantasy book with strong world-building.",
    )

    if st.button("Search with this prompt", use_container_width=True, key="custom_preference_search"):
        st.session_state["selected_preference_label"] = "Custom prompt"
        st.session_state["preference_search_ready"] = bool(search_preference.strip())

    if not st.session_state["preference_search_ready"]:
        st.info("Choose Fantasy, Thriller, Romance, Classic, or Funny above, or type a prompt and click Search.")
        st.stop()

    selected_label = st.session_state["selected_preference_label"] or "Selected preference"
    st.markdown(
        f"<div class='model-note'><b>{escape(selected_label)} search:</b> {escape(search_preference)}</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("<div class='section-title'>Library Check</div>", unsafe_allow_html=True)
        catalog_matches = search_library_catalog(search_preference)
        if catalog_matches:
            st.markdown(
                "These books were found directly in the Goodreads catalog before the model recommendations."
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "Book": row["Book"],
                        "Author(s)": row["Author(s)"],
                        "Avg rating": row["Avg rating"],
                        "# ratings": row["# ratings"],
                    }
                    for row in catalog_matches
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "Book not found in the library. The app will still show recommendation-model matches "
                "based on your reading preference."
            )

    model_names = ["Baseline", "User-Based CF", "Item-Based CF", "SVD"]
    model_candidate_sets = {}
    for search_model_name in model_names:
        search_model = fit_cf_model(search_model_name)
        model_candidate_sets[search_model_name] = recommend(
            search_model,
            user_id,
            min_ratings,
            top_n=max(top_n, 10),
        )

    gemini_key = get_gemini_key()
    use_gemini_search = bool(gemini_key) and st.button(
        "Use Gemini to re-rank each model's candidates",
        use_container_width=True,
    )

    if not gemini_key:
        st.info(
            "Gemini key not found, so this tab is using a transparent local preference-match preview. "
            "It still only searches books produced by the recommender models."
        )

    model_descriptions = {
        "Baseline": "Recommends broadly liked books that match the selected preference.",
        "User-Based CF": "Starts from readers similar to the selected user, then matches the preference.",
        "Item-Based CF": "Starts from books similar to the user's history, then matches the preference.",
        "SVD": "Uses hidden reader-book patterns from the rating matrix, then matches the preference.",
    }
    for search_model_name, candidates in model_candidate_sets.items():
        with st.container(border=True):
            st.markdown(f"<div class='section-title'>{search_model_name}</div>", unsafe_allow_html=True)
            st.markdown(model_descriptions[search_model_name])
            if use_gemini_search:
                try:
                    ranked_candidates = rerank_with_gemini(candidates, search_preference)
                except Exception as exc:
                    st.warning(f"Gemini failed for {search_model_name}: {exc}")
                    ranked_candidates = local_preference_rank(candidates, search_preference)
            else:
                ranked_candidates = local_preference_rank(candidates, search_preference)

            result_rows = []
            for rank, row in enumerate(ranked_candidates[:5], start=1):
                result_rows.append({
                    "Rank": rank,
                    "Book": row["title"],
                    "Author(s)": row["authors"],
                    "Predicted": round(row["predicted"], 2),
                    "Avg rating": round(row["avg_rating_project"], 2),
                    "# ratings": row["n_ratings"],
                })
                render_book_card(row, rank, row.get("llm_reason", ""))

if active_page == "Reader Profile":
    st.markdown(f"<div class='section-title'>Reader Profile: User {user_id}</div>", unsafe_allow_html=True)
    render_reader_profile_header()
    render_user_summary_cards()
    p1, p2, p3 = st.columns(3)
    with p1:
        stat_box(top_author, "Most frequent author in history")
    with p2:
        stat_box(int((user_history["Their rating"] >= 4).sum()), "Books rated 4 or 5")
    with p3:
        stat_box(model_choice, "Selected model")

    st.markdown("<div class='section-title'>Reading History</div>", unsafe_allow_html=True)
    if PLOTLY_AVAILABLE:
        history_left, history_right = st.columns(2)
        with history_left:
            user_rating_counts = (
                user_history["Their rating"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            user_rating_counts.columns = ["Rating", "Count"]
            fig = px.bar(
                user_rating_counts,
                x="Rating",
                y="Count",
                color="Rating",
                color_continuous_scale=["#00e054", "#40bcf4", "#ff8000"],
                title=f"User {user_id} Rating Distribution",
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = add_bar_labels(fig, orientation="v")
            st.plotly_chart(style_plot(fig), use_container_width=True)
        with history_right:
            author_counts = (
                user_history["Author(s)"]
                .astype(str)
                .str.split(",")
                .explode()
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(10)
                .reset_index()
            )
            author_counts.columns = ["Author", "Count"]
            fig = px.bar(
                author_counts.sort_values("Count"),
                x="Count",
                y="Author",
                orientation="h",
                color="Count",
                color_continuous_scale=["#00e054", "#40bcf4", "#ff8000"],
                title="Most Frequent Authors in Reader History",
            )
            fig.update_layout(coloraxis_showscale=False)
            fig = add_bar_labels(fig, orientation="h")
            st.plotly_chart(style_plot(fig), use_container_width=True)

    st.dataframe(user_history, use_container_width=True)

if active_page == "Dataset & Evaluation":
    st.markdown("<div class='section-title'>Dataset & Evaluation</div>", unsafe_allow_html=True)
    overview = dataset_overview()
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        stat_box(f"{len(books):,}", "Books in catalog")
    with d2:
        stat_box(f"{len(ratings):,}", "Ratings analyzed")
    with d3:
        stat_box(overview["Users"], "Users")
    with d4:
        stat_box(overview["Books with ratings"], "Books with ratings")

    d5, d6, d7 = st.columns(3)
    with d5:
        stat_box(overview["Avg ratings per user"], "Avg ratings per user")
    with d6:
        stat_box(overview["Avg ratings per book"], "Avg ratings per book")
    with d7:
        stat_box("1-5 stars", "Rating scale")

    st.markdown("<div class='section-title'>Model Comparison Results</div>", unsafe_allow_html=True)
    st.markdown(
        "The final notebook compares RMSE, Precision@10, and Recall@10. "
        "The Popularity Baseline predicts ratings most accurately by RMSE, but "
        "**UBCF Pearson k=50** is selected for the app because it has the strongest "
        "Top-10 recommendation quality."
    )
    st.dataframe(model_comparison_table(), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>EDA Visuals</div>", unsafe_allow_html=True)
    rating_counts = ratings["rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Rating", "Count"]

    if PLOTLY_AVAILABLE:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            fig = px.bar(rating_counts, x="Rating", y="Count", title="Rating Distribution")
            fig = add_bar_labels(fig, orientation="v")
            st.plotly_chart(style_plot(fig), use_container_width=True)
        with chart_right:
            top_catalog = (
                books.sort_values("ratings_count", ascending=False)
                .head(10)[["title", "ratings_count"]]
            )
            fig2 = px.bar(top_catalog, x="ratings_count", y="title", orientation="h", title="Most Rated Books in Catalog")
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            fig2 = add_bar_labels(fig2, orientation="h")
            st.plotly_chart(style_plot(fig2), use_container_width=True)

        chart_three, chart_four = st.columns(2)
        with chart_three:
            books_by_year = (
                books.dropna(subset=["original_publication_year"])
                .assign(original_publication_year=lambda df: pd.to_numeric(df["original_publication_year"], errors="coerce"))
                .dropna(subset=["original_publication_year"])
            )
            books_by_year = books_by_year[
                (books_by_year["original_publication_year"] >= 1800)
                & (books_by_year["original_publication_year"] <= 2026)
            ].copy()
            books_by_year["Decade"] = (books_by_year["original_publication_year"] // 10 * 10).astype(int)
            decade_counts = books_by_year["Decade"].value_counts().sort_index().reset_index()
            decade_counts.columns = ["Decade", "Books"]
            fig3 = px.line(decade_counts, x="Decade", y="Books", markers=True, title="Catalog Books by Publication Decade")
            st.plotly_chart(style_plot(fig3), use_container_width=True)
        with chart_four:
            rec_years = pd.DataFrame(recs)
            rec_years["year_num"] = pd.to_numeric(rec_years["year"], errors="coerce")
            rec_years = rec_years.dropna(subset=["year_num"])
            if len(rec_years):
                fig4 = px.scatter(
                    rec_years,
                    x="year_num",
                    y="predicted",
                    size="n_ratings",
                    color="avg_rating_project",
                    hover_name="title",
                    color_continuous_scale=["#00e054", "#40bcf4", "#ff8000"],
                    title="Recommended Books by Year and Predicted Rating",
                )
                fig4.update_layout(coloraxis_showscale=False, xaxis_title="Publication Year", yaxis_title="Predicted Rating")
                st.plotly_chart(style_plot(fig4), use_container_width=True)
            else:
                st.info("Recommended books do not have enough publication-year data for this chart.")
    else:
        st.dataframe(rating_counts, use_container_width=True)
        st.info("Install plotly to show interactive charts.")

if active_page == "Model Notes":
    st.markdown("<div class='section-title'>Model Notes for the Demo</div>", unsafe_allow_html=True)
    if not SURPRISE_AVAILABLE:
        st.markdown(
            """
            <div class="model-note">
                <b>Demo mode:</b> This Python environment does not have scikit-surprise installed,
                so the app is using a backup scorer to keep the interface running. The formal
                Project 2 notebook still uses the required Surprise collaborative-filtering models.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        **Final app choice:** UBCF Pearson k=50 is used as the main retrieval model because it had the
        strongest Top-10 recommendation quality in the final notebook results. The Popularity Baseline had
        the best RMSE, but the app is built around useful ranked book lists, so Precision@10 and Recall@10
        matter more for the final model choice.

        **Collaborative-filtering step:** the app first creates a candidate list from user-book rating patterns.

        **User-Based CF:** recommends books by finding readers with similar rating patterns to the selected user.
        If similar readers liked a book that this user has not rated yet, that book can become a recommendation.
        This is intuitive for a demo because it answers: "What did readers like me enjoy?"

        **Item-Based CF:** recommends books by finding books that behave similarly across users.
        If a user liked one book, the model looks for other books that tend to be liked by the same kinds of readers.
        This answers: "What books are similar to books this reader already liked?"

        **Baseline:** a simple benchmark model based on average rating patterns rather than deeper similarity.
        It is included because a recommender should beat or at least be compared against a simple popularity/mean
        approach. If the baseline performs very well, that can mean popularity and average ratings are strong signals
        in this dataset.

        **SVD:** a matrix-factorization model that learns hidden reader-book patterns. It was competitive on RMSE,
        but it did not beat UBCF Pearson k=50 on Precision@10 and Recall@10 in the final results.

        **LLM step:** Gemini re-ranks only the books returned by the recommender using the user's stated preference
        and book metadata. It should explain actual Goodreads candidates, not create new books outside the dataset.

        **Design choice:** the minimum-ratings slider reduces unstable recommendations from books with very few ratings.
        """
    )

st.markdown(
    """
    <div class="app-footer">
        Goodreads - Helping Readers Find Their Perfect Match.
    </div>
    """,
    unsafe_allow_html=True,
)
