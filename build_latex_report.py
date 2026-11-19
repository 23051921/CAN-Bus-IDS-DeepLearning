# -*- coding: utf-8 -*-
"""
build_latex_report.py: Compiles the full master report LaTeX source CANBUS_PROJECT_MASTER_REPORT.tex.
Implements institutional KIIT format matching Lung_Cancer_Report.pdf with clickable references,
bounded figures, clean mathematical typography, and zero syntax leaks.
"""

import os
import re
import report_content as rc
import latex_helpers as lh

BASE_DIR = r"C:\Users\KIIT0001\Downloads\7th sem project"
TEX_PATH = os.path.join(BASE_DIR, "OVERLEAF_COMPILED_MASTER_REPORT.tex")

def generate_latex():
    tex = []
    
    # 1. Document Class and Packages
    tex.append(r"""\documentclass[12pt,a4paper,oneside]{report}

% ── Geometry & Margins ────────────────────────────────────────────────────────
\usepackage[a4paper,top=25mm,bottom=25mm,left=30mm,right=25mm]{geometry}
\setlength{\headheight}{15pt}
\addtolength{\topmargin}{-3pt}

% ── Core Mathematical Packages ────────────────────────────────────────────────
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{enumitem}

% ── Typography & Fonts ────────────────────────────────────────────────────────
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{setspace}
\setstretch{1.15}

% ── Graphics & Figures ────────────────────────────────────────────────────────
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}

% ── Tables ───────────────────────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{array}

% ── Color Definitions ────────────────────────────────────────────────────────
\usepackage{xcolor}
\definecolor{kiitnavy}{RGB}{27, 54, 93}       % #1B365D - KIIT Academic Navy
\definecolor{kiitslate}{RGB}{44, 82, 130}     % #2C5282 - Subheading Slate
\definecolor{charcoal}{RGB}{45, 55, 72}       % #2D3748 - Body Charcoal
\definecolor{mutedgray}{RGB}{113, 128, 150}   % #718096 - Metadata Gray
\definecolor{codegray}{RGB}{247, 249, 252}    % #F7F9FC - Code / Box Background
\definecolor{bordergray}{RGB}{203, 213, 224}  % #CBD5E0 - Hairline Rule

% ── Header & Footer Formatting ────────────────────────────────────────────────
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\scshape\color{mutedgray} AUTOMOTIVE CAN BUS INTRUSION DETECTION}
\fancyhead[R]{\small\scshape\color{mutedgray} KIIT, BBSR}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[L]{\small\scshape\color{mutedgray} AUTOMOTIVE CAN BUS INTRUSION DETECTION}
  \fancyhead[R]{\small\scshape\color{mutedgray} KIIT, BBSR}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0pt}
}

% ── Section Heading Styling ──────────────────────────────────────────────────
\usepackage{titlesec}
\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\color{kiitnavy}}{\chaptertitlename\ \thechapter}{10pt}{\Huge}
\titlespacing*{\chapter}{0pt}{-10pt}{18pt}

\titleformat{\section}
  {\normalfont\Large\bfseries\color{kiitnavy}}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{14pt}{6pt}

\titleformat{\subsection}
  {\normalfont\large\bfseries\color{kiitslate}}{\thesubsection}{1em}{}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries\color{charcoal}}{\thesubsubsection}{1em}{}
\titlespacing*{\subsubsection}{0pt}{8pt}{3pt}

% ── Algorithm Framing ────────────────────────────────────────────────────────
\usepackage{framed}

% ── Citations, Bibliography & TOC Binding ─────────────────────────────────────
\usepackage{cite}
\usepackage{url}
\usepackage[nottoc]{tocbibind}
\renewcommand{\bibname}{References}
\renewcommand{\contentsname}{Table of Contents}

% ── Interactive Clickable Hyperlinks ─────────────────────────────────────────
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=kiitnavy,
    citecolor=kiitnavy,
    filecolor=kiitnavy,
    urlcolor=kiitslate,
    plainpages=false,
    pdfpagelabels=true,
    pdfpagemode=UseOutlines,
    bookmarksopen=true,
    bookmarksnumbered=true,
    pdftitle={Automotive CAN Bus Intrusion Detection Using a Kinematic-Informed Framework Coupling Data-Localized Federated Learning and Multi-Scale Autoencoder Ensembles},
    pdfauthor={Mehul Batla and Kaliseyer Mukherjee},
    pdfsubject={B.Tech Project Report, KIIT Deemed to be University},
    pdfkeywords={CAN Bus, Intrusion Detection, Federated Learning, Deep SVDD, CUSUM, Inception 1D-CNN, ISO/SAE 21434}
}

% ── Document Environment ──────────────────────────────────────────────────────
\begin{document}
\nocite{panda2017fpga,hanselmann2020canet,nurnberger2016vatican,ruff2018deepsvdd,verma2022road,song2020carhacking,page1954cusum,mcmahan2017fedavg,yin2018byzantine,iso11898,isosae21434,iso14229}
""")

    # 2. Cover / Title Page (with pageanchor=false to prevent page.1 duplicate target)
    tex.append(r"""% ── Title Page ──────────────────────────────────────────────────────────────
\hypersetup{pageanchor=false}
\begin{titlepage}
\newgeometry{top=18mm,bottom=18mm,left=25mm,right=25mm}
\thispagestyle{empty}
\begin{center}
{\Large \textbf{A PROJECT REPORT}\par}
\vspace{0.1cm}
{\large \textit{on}\par}
\vspace{0.15cm}
{\LARGE \textbf{\color{kiitnavy} Automotive CAN Bus Intrusion Detection}\par}
\vspace{0.12cm}
{\large \color{kiitnavy} Using a Kinematic-Informed Framework Coupling Data-Localized Federated Learning and Multi-Scale Autoencoder Ensembles\par}
\vspace{0.25cm}
{\large Submitted to\par}
\vspace{0.1cm}
{\Large \textbf{KIIT Deemed to be University}\par}
\vspace{0.15cm}
{\normalsize In Partial Fulfilment of the Requirement for the Award of the Degree of\par}
\vspace{0.1cm}
{\Large \textbf{Bachelor of Technology in Computer Science \& Engineering}\par}
\vspace{0.25cm}
{\large \textit{By}\par}
\vspace{0.1cm}
{\large \textbf{Mehul Batla} \quad (Roll No. 23051921)\par}
\vspace{0.05cm}
{\large \textbf{Kaliseyer Mukherjee} \quad (Roll No. 23051915)\par}
\vspace{0.25cm}
{\large \textbf{Under the Guidance of}\par}
\vspace{0.1cm}
{\large \textbf{Dr. Amiya Ranjan Panda}\par}
\vspace{0.05cm}
{\normalsize Professor, School of Computer Engineering\par}
\vfill
\includegraphics[width=2.5cm]{dashboard_data/kiit_logo.png}\par
\vspace{0.2cm}
{\large \textbf{School of Computer Engineering}\par}
{\large \textbf{Kalinga Institute of Industrial Technology}\par}
{\normalsize \textbf{KIIT Deemed to be University, Bhubaneswar, ODISHA 751024}\par}
\vspace{0.15cm}
{\large \textbf{November 2026}\par}
\end{center}
\end{titlepage}
\restoregeometry
\clearpage
\hypersetup{pageanchor=true}
""")

    # 3. Front Matter with Roman Numeral Pagination
    tex.append(r"""% ── Front Matter ─────────────────────────────────────────────────────────────
\pagenumbering{roman}
\setcounter{page}{1}

% ── Bonafide Certificate ──────────────────────────────────────────────────────
\clearpage
\phantomsection
\addcontentsline{toc}{chapter}{Certificate}

\begin{center}
\vspace*{0.2cm}
\includegraphics[width=2.5cm]{dashboard_data/kiit_logo.png}\\[0.25cm]
{\large \textbf{School of Computer Engineering}}\\[0.1cm]
{\large \textbf{Kalinga Institute of Industrial Technology}}\\[0.1cm]
{\normalsize \textbf{KIIT Deemed to be University, Bhubaneswar, ODISHA 751024}}\\[0.6cm]
{\LARGE \textbf{\color{kiitnavy} CERTIFICATE}}\\[0.5cm]
\end{center}

\noindent This is to certify that the project entitled\par
\vspace{0.3cm}
\begin{center}
\textbf{\large ``Automotive CAN Bus Intrusion Detection Using a Kinematic-Informed Framework Coupling Data-Localized Federated Learning and Multi-Scale Autoencoder Ensembles''}
\end{center}
\vspace{0.3cm}

\begin{center}
\textit{submitted by}\\[0.3cm]
\begin{tabular}{cc}
\textbf{Mehul Batla} & \textbf{23051921} \\
\textbf{Kaliseyer Mukherjee} & \textbf{23051915} \\
\end{tabular}
\end{center}
\vspace{0.4cm}

\noindent is a record of bonafide work carried out by them, in the partial fulfilment of the requirement for the award of Degree of \textbf{Bachelor of Technology (Computer Science \& Engineering)} at \textbf{KIIT Deemed to be University, Bhubaneswar}. This work is done during year 2025--2026, under our guidance.

\vspace{2.5cm}

\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lr}
\textbf{Date: \rule{1.2cm}{0.4pt} / \rule{1.2cm}{0.4pt}} & \textbf{(Prof. Amiya Ranjan Panda)} \\
& \textit{Project Guide} \\
\end{tabular*}

\clearpage
""")

    # 4. Abstract
    tex.append(r"""% ── Abstract ──────────────────────────────────────────────────────────────────
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

""" + lh.format_text_to_latex(rc.ABSTRACT_TEXT) + r"""

\vspace{0.4cm}
\noindent \textbf{Keywords:} """ + rc.KEYWORDS.replace('_', r'\_').replace('&', r'\&') + r"""

\clearpage
""")

    # 5. Acknowledgments
    tex.append(r"""% ── Acknowledgments ───────────────────────────────────────────────────────────
\chapter*{Acknowledgments}
\addcontentsline{toc}{chapter}{Acknowledgments}

We would like to express our deepest gratitude and profound respect to our project supervisor, \textbf{Dr.~Amiya Ranjan Panda}~\cite{panda2017fpga}, Professor, School of Computer Engineering, KIIT Deemed to be University, for his invaluable guidance, inspiring encouragement, and meticulous technical scrutiny throughout the lifecycle of this research. His constructive feedback was instrumental in formulating our dual-window feature extraction pipeline, establishing geometric separation under volumetric flooding, and deploying our federated learning testbed.

We extend our sincere thanks to the \textbf{School of Computer Engineering} and the academic leadership of \textbf{KIIT Deemed to be University} for providing world-class laboratory infrastructure, computing facilities, and an encouraging research environment.

Finally, we express our heartfelt appreciation to our parents, family members, and peers for their continuous moral support and understanding throughout our B.Tech curriculum.

\vspace{2.0cm}
\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc}
\rule{6.0cm}{0.4pt} & \rule{6.0cm}{0.4pt} \\
\textbf{Mehul Batla} (23051921) & \textbf{Kaliseyer Mukherjee} (23051915) \\
Department of Computer Science \& Engineering & Department of Computer Science \& Engineering \\
KIIT Deemed to be University & KIIT Deemed to be University \\
\end{tabular*}

\clearpage
""")

    # 6. TOC, LOF, LOT
    tex.append(r"""% ── Tables of Contents, Figures & Tables ──────────────────────────────────────
\tableofcontents
\clearpage

\listoffigures
\clearpage

\listoftables
\clearpage
""")

    # 7. Main Matter (Arabic Pagination)
    tex.append(r"""% ── Main Matter ──────────────────────────────────────────────────────────────
\clearpage
\pagenumbering{arabic}
\setcounter{page}{1}
""")

    # CHAPTER 1
    tex.append(r"\chapter{Introduction}" + "\n\n")
    for sec in rc.CH1_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")
        if "1.3" in sec['title']:
            tex.append(lh.format_latex_figure(
                "vehicular_can_architecture.png",
                "Modern Vehicular E/E Architecture, CAN Bus Network Topology, and Multi-Scale Attack Vectors.",
                "fig:arch_vectors",
                width="0.95\\linewidth",
                max_height="0.80\\textheight"
            ))

    # CHAPTER 2
    tex.append(r"\chapter{Literature Review \& Research Gaps}" + "\n\n")
    for sec in rc.CH2_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")
        if "2.2" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_2_1_HEADERS,
                rc.TABLE_2_1_ROWS,
                "Comparative Taxonomy of In-Vehicle Intrusion Detection Systems",
                "tab:taxonomy",
                col_spec=r">{\raggedright\arraybackslash\bfseries}p{2.1cm} >{\raggedright\arraybackslash}p{2.3cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}p{1.7cm} >{\centering\arraybackslash}p{1.3cm} >{\raggedright\arraybackslash}X",
                font_size=r"\footnotesize"
            ))

    # CHAPTER 3
    tex.append(r"\chapter{Basic Concepts \& Theoretical Foundations}" + "\n\n")
    for sec in rc.CH3_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")
        if "3.2" in sec['title']:
            tex.append(lh.format_latex_figure(
                "fl_xai_attribution.png",
                "Feature Salience and Inception Multi-Scale Representation Attribution.",
                "fig:salience_attribution",
                width="0.90\\linewidth",
                max_height="0.45\\textheight"
            ))

    # CHAPTER 4
    tex.append(r"\chapter{System Architecture \& Requirements}" + "\n\n")
    for sec in rc.CH4_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        if "4.1" in sec['title']:
            tex.append(lh.format_latex_figure(
                "canbus_pipeline_flowchart.png",
                "End-to-End System Architecture and Multi-Tiered Information Flow.",
                "fig:system_arch",
                width="0.82\\linewidth",
                max_height="0.70\\textheight"
            ))
            tex.append(lh.format_section_body(sec['text']) + "\n\n")
        elif "4.2" in sec['title']:
            tex.append(lh.format_section_body(sec['text']) + "\n\n")
            tex.append(lh.format_latex_table(
                rc.TABLE_4_1_HEADERS,
                rc.TABLE_4_1_ROWS,
                "Software Requirements Specification (SRS) - Traceability Matrix",
                "tab:srs",
                col_spec=r">{\bfseries}p{1.6cm} p{1.8cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.8cm} >{\raggedright\arraybackslash}X",
                font_size=r"\footnotesize"
            ))
        else:
            tex.append(lh.format_section_body(sec['text']) + "\n\n")

    # CHAPTER 5
    tex.append(r"\chapter{Implementation \& Experimental Methodology}" + "\n\n")
    for sec in rc.CH5_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")
        if "5.1" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_5_1_HEADERS,
                rc.TABLE_5_1_ROWS,
                "ISO/PAS 8800:2024 Structured Dataset Card - ROAD Dataset",
                "tab:road_card",
                col_spec=r">{\bfseries}p{5.5cm} X",
                font_size=r"\small"
            ))
            tex.append(lh.format_latex_table(
                rc.TABLE_5_2_HEADERS,
                rc.TABLE_5_2_ROWS,
                "ISO/PAS 8800:2024 Structured Dataset Card - HCRL Car-Hacking Dataset",
                "tab:hcrl_card",
                col_spec=r">{\bfseries}p{5.5cm} X",
                font_size=r"\small"
            ))
        elif "5.2" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_5_3_HEADERS,
                rc.TABLE_5_3_ROWS,
                "Disentangled 45-D / 51-D Multi-Scale Feature Specification",
                "tab:feature_spec",
                col_spec=r">{\bfseries}p{2.8cm} p{1.8cm} p{2.0cm} X",
                font_size=r"\footnotesize"
            ))
            tex.append(lh.format_latex_figure(
                "dual_window_feature_pipeline.png",
                "Dual-Window Multi-Scale Feature Engineering Pipeline (Fast W=50, Slow W=200).",
                "fig:feature_pipeline",
                width="0.85\\linewidth",
                max_height="0.65\\textheight"
            ))
            tex.append(lh.format_latex_algorithm(
                rc.ALGORITHM_1_TEXT,
                "Algorithm 1: Dual-Window Multi-Scale Feature Extraction Pipeline",
                "alg:dual_window"
            ))
        elif "5.3" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_5_4_HEADERS,
                rc.TABLE_5_4_ROWS,
                "Baseline and Tuned Neural Network Structural Hyperparameters",
                "tab:hyperparameters",
                col_spec=r">{\bfseries}p{3.8cm} p{1.6cm} p{1.6cm} p{1.6cm} X",
                font_size=r"\small"
            ))
            tex.append(lh.format_latex_algorithm(
                rc.ALGORITHM_2_TEXT,
                "Algorithm 2: Telemetry-Gated Specialist Gateway Inference & Routing",
                "alg:specialist_gateway"
            ))
        elif "5.4" in sec['title']:
            tex.append(lh.format_latex_algorithm(
                rc.ALGORITHM_3_TEXT,
                "Algorithm 3: Data-Localized Robust Federated Aggregation with Byzantine Defense",
                "alg:robust_fl"
            ))

    # CHAPTER 6
    tex.append(r"\chapter{Assessment of Results \& Empirical Evaluation}" + "\n\n")
    for sec in rc.CH6_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")
        if "6.1" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_1_HEADERS,
                rc.TABLE_6_1_ROWS,
                "Master Leaderboard across 13 ROAD Masquerade Scenarios (Run 50)",
                "tab:road_leaderboard",
                col_spec=r">{\bfseries}p{3.8cm} *{6}{>{\centering\arraybackslash}X}",
                font_size=r"\small"
            ))
            tex.append(lh.format_latex_figure(
                "run50_ensemble_benchmark.png",
                "Master Benchmark: Specialist Gateway vs. Standalone Model Leaderboard on 13 ROAD Masquerade Scenarios.",
                "fig:road_benchmark",
                width="1.0\\linewidth",
                max_height="0.58\\textheight"
            ))
        elif "6.2" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_2_HEADERS,
                rc.TABLE_6_2_ROWS,
                "Blind Held-Out Generalization Performance on 5 Reserved Scenarios",
                "tab:held_out",
                col_spec=r">{\raggedright\arraybackslash\bfseries}p{4.0cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}p{1.8cm} >{\centering\arraybackslash}X",
                font_size=r"\small"
            ))
        elif "6.3" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_3_HEADERS,
                rc.TABLE_6_3_ROWS,
                "Formal Wilcoxon Paired Signed-Rank Significance Matrix (N=13)",
                "tab:wilcoxon",
                col_spec=r">{\raggedright\arraybackslash\bfseries}p{3.0cm} >{\centering\arraybackslash}p{1.2cm} >{\centering\arraybackslash}p{1.6cm} >{\centering\arraybackslash}p{2.5cm} >{\raggedright\arraybackslash}X",
                font_size=r"\small"
            ))
        elif "6.4" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_4_HEADERS,
                rc.TABLE_6_4_ROWS,
                "Operational Recall Elevation under Sequential CUSUM Filtering",
                "tab:cusum_table",
                col_spec=r">{\bfseries}p{3.2cm} c c c X",
                font_size=r"\small"
            ))
            tex.append(lh.format_latex_figure(
                "run50_cusum_benchmark.png",
                "Sequential CUSUM Filtering vs. Static Thresholding: Operational P95 Recall Elevation.",
                "fig:cusum_benchmark",
                width="0.90\\linewidth",
                max_height="0.45\\textheight"
            ))
        elif "6.5" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_5_HEADERS,
                rc.TABLE_6_5_ROWS,
                "Data-Localized Federated Learning Clean & Adversarial Benchmark",
                "tab:fl_table",
                col_spec=r">{\raggedright\arraybackslash\bfseries}p{2.5cm} >{\raggedright\arraybackslash}p{2.2cm} *{4}{>{\centering\arraybackslash}p{1.3cm}} >{\centering\arraybackslash}X",
                font_size=r"\footnotesize"
            ))
            tex.append(lh.format_latex_figure(
                "fl_convergence_50rounds.png",
                "Federated Learning Convergence Trajectories Across 50 Communication Rounds.",
                "fig:fl_convergence",
                width="0.90\\linewidth",
                max_height="0.42\\textheight"
            ))
            tex.append(lh.format_latex_figure(
                "run50_fl_comparison.png",
                "Federated Learning Robustness: Clean vs. Adversarial Byzantine Poisoning Mitigation.",
                "fig:fl_robustness",
                width="0.90\\linewidth",
                max_height="0.42\\textheight"
            ))
        elif "6.6" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_6_HEADERS,
                rc.TABLE_6_6_ROWS,
                "Independent Cross-Benchmark Validation on HCRL Car-Hacking",
                "tab:hcrl_table",
                col_spec=r">{\raggedright\arraybackslash\bfseries}p{2.8cm} >{\centering\arraybackslash}p{1.5cm} >{\centering\arraybackslash}p{1.6cm} >{\centering\arraybackslash}p{1.6cm} >{\centering\arraybackslash}p{1.5cm} >{\centering\arraybackslash}X",
                font_size=r"\small"
            ))
            tex.append(lh.format_latex_figure(
                "car_hacking_heatmap.png",
                "Cross-Dataset Evaluation Heatmap on HCRL Car-Hacking: Geometric Separation under Volumetric Flooding.",
                "fig:car_hacking_heatmap",
                width="0.90\\linewidth",
                max_height="0.45\\textheight"
            ))
        elif "6.7" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_6_7_HEADERS,
                rc.TABLE_6_7_ROWS,
                "Host-CPU Latency Profiling & Preliminary Computational Feasibility",
                "tab:latency_table",
                col_spec=r">{\raggedright\arraybackslash\bfseries}p{3.2cm} >{\raggedright\arraybackslash}p{2.1cm} >{\centering\arraybackslash}p{1.3cm} >{\centering\arraybackslash}p{1.3cm} >{\centering\arraybackslash}p{1.4cm} >{\centering\arraybackslash}X",
                font_size=r"\small"
            ))
            tex.append(lh.format_latex_figure(
                "run42_wcet_histogram.png",
                "Host-CPU Execution Latency Distribution and Worst-Case Execution Time Profile (WCET).",
                "fig:wcet_hist",
                width="0.90\\linewidth",
                max_height="0.45\\textheight"
            ))
            tex.append(lh.format_latex_figure(
                "run42_xai_root_cause.png",
                "Explainable AI (XAI) Attribution & Telemetry Root-Cause Analysis.",
                "fig:xai_attribution",
                width="0.95\\linewidth",
                max_height="0.45\\textheight"
            ))

    # CHAPTER 7
    tex.append(r"\chapter{Standards Adopted \& Threat Governance}" + "\n\n")
    for sec in rc.CH7_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")
        if "7.1" in sec['title']:
            tex.append(lh.format_latex_table(
                rc.TABLE_7_1_HEADERS,
                rc.TABLE_7_1_ROWS,
                "Automotive Standards Alignment & Compliance Governance Matrix",
                "tab:standards_matrix",
                col_spec=r">{\bfseries}p{2.0cm} p{3.0cm} p{1.8cm} p{2.6cm} X",
                font_size=r"\footnotesize"
            ))

    # CHAPTER 8
    tex.append(r"\chapter{Conclusion}" + "\n\n")
    for sec in rc.CH8_SECTIONS:
        sec_title = re.sub(r'^\d+\.\d+\s*', '', sec['title'])
        tex.append(f"\\section{{{lh.format_text_to_latex(sec_title)}}}\n\n")
        tex.append(lh.format_section_body(sec['text']) + "\n\n")

    # 8. References (Handled automatically by tocbibind and \renewcommand{\bibname}{References})
    tex.append(r"""% ── References ────────────────────────────────────────────────────────────────
\clearpage
\bibliographystyle{IEEEtran}
\bibliography{references}
\clearpage
""")

    # 9. Back Matter
    tex.append(r"""% ── Back Matter ──────────────────────────────────────────────────────────────
\chapter*{Individual Contribution Report}
\addcontentsline{toc}{chapter}{Individual Contribution Report: Kaliseyer Mukherjee}

\begin{center}
{\Large \textbf{Kaliseyer Mukherjee}}\\[0.1cm]
{\large Roll No. 23051915}\\[0.4cm]
\end{center}

\noindent \textbf{Abstract:} """ + lh.format_text_to_latex(rc.INDIVIDUAL_CONTRIBUTION_KALISEYER['abstract']) + r"""

\vspace{0.4cm}
\noindent {\large \textbf{\color{kiitnavy} Individual Contribution and Findings}}\\[0.2cm]
""" + lh.format_section_body(rc.INDIVIDUAL_CONTRIBUTION_KALISEYER['findings']) + r"""

\vspace{0.4cm}
\noindent {\large \textbf{\color{kiitnavy} Contribution to Report Preparation}}\\[0.2cm]
""" + lh.format_text_to_latex(rc.INDIVIDUAL_CONTRIBUTION_KALISEYER['report_prep']) + r"""

\vspace{0.4cm}
\noindent {\large \textbf{\color{kiitnavy} Contribution to Presentation}}\\[0.2cm]
""" + lh.format_text_to_latex(rc.INDIVIDUAL_CONTRIBUTION_KALISEYER['presentation']) + r"""

\vspace{2.0cm}
\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc}
\rule{6.0cm}{0.4pt} & \rule{6.0cm}{0.4pt} \\
\textbf{Full Signature of Supervisor} & \textbf{Full Signature of Student} \\
\end{tabular*}

\clearpage

\chapter*{Individual Contribution Report}
\addcontentsline{toc}{chapter}{Individual Contribution Report: Mehul Batla}

\begin{center}
{\Large \textbf{Mehul Batla}}\\[0.1cm]
{\large Roll No. 23051921}\\[0.4cm]
\end{center}

\noindent \textbf{Abstract:} """ + lh.format_text_to_latex(rc.INDIVIDUAL_CONTRIBUTION_MEHUL['abstract']) + r"""

\vspace{0.4cm}
\noindent {\large \textbf{\color{kiitnavy} Individual Contribution and Findings}}\\[0.2cm]
""" + lh.format_section_body(rc.INDIVIDUAL_CONTRIBUTION_MEHUL['findings']) + r"""

\vspace{0.4cm}
\noindent {\large \textbf{\color{kiitnavy} Contribution to Report Preparation}}\\[0.2cm]
""" + lh.format_text_to_latex(rc.INDIVIDUAL_CONTRIBUTION_MEHUL['report_prep']) + r"""

\vspace{0.4cm}
\noindent {\large \textbf{\color{kiitnavy} Contribution to Presentation}}\\[0.2cm]
""" + lh.format_text_to_latex(rc.INDIVIDUAL_CONTRIBUTION_MEHUL['presentation']) + r"""

\vspace{2.0cm}
\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc}
\rule{6.0cm}{0.4pt} & \rule{6.0cm}{0.4pt} \\
\textbf{Full Signature of Supervisor} & \textbf{Full Signature of Student} \\
\end{tabular*}

\clearpage

\chapter*{Plagiarism Verification Statement}
\addcontentsline{toc}{chapter}{Plagiarism Verification Statement}

""" + lh.format_section_body(rc.PLAGIARISM_REPORT_TEXT) + r"""

\vspace{2.5cm}
\noindent
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}cc}
\rule{6.0cm}{0.4pt} & \rule{4.0cm}{0.4pt} \\
\textbf{Full Signature of Supervisor} & \textbf{Date: \today} \\
\end{tabular*}

\end{document}
""")

    full_tex = "".join(tex)
    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write(full_tex)
    print(f"LaTeX document generated successfully at: {TEX_PATH} ({len(full_tex)} chars)")

if __name__ == "__main__":
    generate_latex()
