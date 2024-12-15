# Fraud Detection Analysis System

A comprehensive system for analyzing fraud detection data using parallel processing and providing visualizations through a REST API.

## Overview

This project provides a scalable solution for analyzing fraud detection data stored in DynamoDB. It uses parallel processing for data fetching and processing, and exposes the analysis through a Flask API endpoint.

## Execution Flowchart

```mermaid
graph TD
    A[Start] --> B[Initialize DynamoDB Connection]
    B --> C[Parallel Data Fetch]
    C --> D[Data Processing]
    D --> E[Data Analysis]
    E --> F[Generate Visualizations]
    F --> G[API Response]
    G --> H[End]

    subgraph Parallel Data Fetch
        C1[Segment 1] --> C
        C2[Segment 2] --> C
        C3[Segment n] --> C
    end

    subgraph Data Processing
        D1[Convert Types] --> D
        D2[Handle NaN Values] --> D
        D3[Process Categories] --> D
    end

    subgraph Data Analysis
        E1[Time Analysis] --> E
        E2[Amount Analysis] --> E
        E3[Card Analysis] --> E
        E4[Device Analysis] --> E
        E5[Statistical Analysis] --> E
    end

    subgraph Generate Visualizations
        F1[Transaction Time Plots] --> F
        F2[Amount Distribution Plots] --> F
        F3[Card Usage Charts] --> F
        F4[Device Analysis Charts] --> F
        F5[Statistical Summary] --> F
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bfb,stroke:#333,stroke-width:2px
    style E fill:#fbf,stroke:#333,stroke-width:2px
    style F fill:#fbb,stroke:#333,stroke-width:2px
```
