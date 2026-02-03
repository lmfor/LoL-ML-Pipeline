# LoL ML Pipeline

## Overview

This project is an end-to-end machine learning pipeline designed to **predict the winner of a professional League of Legends match** using **pre-game and draft-level information**. The focus is not on building a toy model, but on constructing a **realistic, industry-style ML system** that emphasizes clean data handling, explicit design decisions, and long-term extensibility.

The repository is structured to mirror how ML pipelines are designed in production environments, with clear separation between data ingestion, feature definition, modeling, and future deployment concerns.

---

## Problem Statement

Given contextual information available **before a match begins**, such as:

- teams and players involved  
- draft order and champion picks/bans  
- side selection  
- patch, season, and competitive context  

the goal is to estimate the **probability that a given team will win** the match.

This framing explicitly allows the model to learn:
- player and team strength
- draft advantages and synergies
- meta effects across patches, splits, and seasons

rather than restricting the model to anonymized or purely mechanical gameplay statistics.

---

## Design Philosophy

This project is built around the following core principles.

### Realistic ML workflows

The pipeline is structured to reflect how ML systems are built in practice:
- immutable raw data ingestion
- explicit schema enforcement
- deterministic feature selection
- separation between ingestion, feature processing, training, and evaluation

### No label leakage

Only information available **prior to match outcome** is used as model input.  
Post-game statistics and outcome-derived metrics are explicitly excluded from the feature set.

### Reproducibility and auditability

- Feature schemas are explicitly defined
- Dataset transformations are deterministic
- Model inputs are reproducible given a fixed dataset snapshot

### Extensibility toward production

Although the project currently focuses on offline development, the structure is intentionally designed to support:
- cloud-based data storage
- model artifact versioning
- deployment and monitoring
in later phases.

---

## Current Scope

At its current stage, the project focuses on:

- ingesting raw competitive match data
- enforcing a clean and intentional feature schema
- preparing data for downstream ML workflows
- laying the groundwork for TensorFlow-based modeling

Future stages will expand into:
- model training and evaluation
- benchmarking and error analysis
- cloud integration for storage, inference, and monitoring

---

## Feature Philosophy

The model intentionally includes:
- **identity-level features** (players, teams, leagues)
- **draft information** (picks, bans, side selection)
- **meta context** (patch, season, competitive split)

This allows the model to function similarly to a **context-aware rating system**, rather than a purely mechanical predictor, while remaining explicit about the assumptions being made.

---

## Intended Use

This project is intended as:
- a learning exercise in ML systems design
- a demonstration of production-oriented ML thinking
- a foundation for future experimentation with modeling, deployment, and monitoring

It is not intended to make claims about competitive performance or betting outcomes, but rather to explore how complex contextual information can be incorporated into a principled ML pipeline.

---

## Project Status

This repository is under active development.  
Interfaces, structure, and design decisions may evolve as the project expands into later phases.
