# Facial Analysis Mobile App - PMLDL Project

## Project Overview

A mobile application that uses computer vision to analyze facial features for entertainment purposes. The app provides two main functionalities:
- **Facial Attractiveness Score**: Estimates an attractiveness rating based on facial features
- **Celebrity Look-alike Finder**: Identifies which famous celebrity the user resembles most

## Technical Architecture

### Core ML Models
- **Attractiveness Classifier**: ResNet-50 CNN fine-tuned on SCUT-FBP5500 dataset
- **Look-alike Finder**: Inception Resnet V1 pretrained on VGGFace2, fine-tuned on Open Famous People Faces dataset

### Infrastructure Stack
- **Backend**: FastAPI with PostgreSQL
- **Frontend**: Flutter mobile application
- **Deployment**: Docker containerization

## Datasets

### SCUT-FBP5500 Dataset
- **Size**: 5,500 frontal faces (350×350px)
- **Diversity**: 2,000 Asian females, 2,000 Asian males, 750 Caucasian females, 750 Caucasian males
- **Labels**: Attractiveness scores (1.0-5.0) from human ratings
- **Split**: 60% training, 40% test sets

### Open Famous People Faces Dataset
- **Classes**: 258 celebrities with ≥5 images per class
- **Quality**: Varied image sizes and quality levels
- **Alignment**: Face-aligned using eye position and landmarks
- **Use Case**: Face re-identification and recognition

## Repository Structure

```
├── api/ # FastAPI application
│ ├── models/ # PyTorch ML models
│ ├── services/
│ ├── v1/
│ └── __main__.py # FastAPI entry point
├── datasets/ # Data files
│ └── scut/ # SCUT-FBP5500 dataset
├── models/ # Trained .pt model files
└── tests/ # Test suite
```
## Development Setup

### Prerequisites
- [make](https://www.gnu.org/software/make/), [uv](https://www.uvproject.xyz/), [docker](https://docs.docker.com/get-docker/), [docker-compose](https://docs.docker.com/compose/install/)

### Quick Start
1. **Pull data**: `make pull-data` (downloads datasets and models, ask for config file first)
2. **Start services**: `make compose-up` (launches backend via Docker)
3. **Local development**: `uv sync` then `uv run python -m api`

### Testing
Run `make test` to execute test suite

## Project Timeline
5 two-week sprints concluding November 9, 2025, focusing on iterative development of ML models, mobile integration, and deployment infrastructure.