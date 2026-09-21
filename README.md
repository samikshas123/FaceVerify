# FaceVerify

## Live Face Verification and Liveness Detection Web Application

FaceVerify is a biometric web application developed using Python, Streamlit, OpenCV, and InsightFace.

The system provides a complete face verification workflow that includes user registration, face sample collection, face embedding generation, live camera-based face detection, liveness detection, and face matching.

The main purpose of FaceVerify is to verify whether the person appearing in front of the live camera is the same person whose face was registered in the system.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [How FaceVerify Works](#how-faceverify-works)
- [Registration Workflow](#registration-workflow)
- [Login and Verification Workflow](#login-and-verification-workflow)
- [Liveness Detection](#liveness-detection)
- [Face Matching](#face-matching)
- [Important Difference Between Liveness and Face Matching](#important-difference-between-liveness-and-face-matching)
- [Verification Results](#verification-results)
- [System Workflow](#system-workflow)
- [Technologies Used](#technologies-used)
- [Machine Learning Models](#machine-learning-models)
- [Face Embeddings](#face-embeddings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [InsightFace Model Setup](#insightface-model-setup)
- [Running the Application](#running-the-application)
- [Application Pages](#application-pages)
- [Testing](#testing)
- [Expected Test Results](#expected-test-results)
- [Privacy and Data Handling](#privacy-and-data-handling)
- [Security](#security)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)
- [Evaluation Guide](#evaluation-guide)
- [Conclusion](#conclusion)
- [Author](#author)

---

# Project Overview

FaceVerify is designed to perform live biometric face verification.

The system first creates a registered facial representation during account registration. During login, the application uses a live camera to detect a face, check liveness, generate a live face embedding, and compare it with the registered face embedding.

The application does not consider a person verified simply because a live face is detected.

The system separates:

1. Account authentication
2. Face detection
3. Liveness detection
4. Face embedding generation
5. Face matching

This separation is important because a person can be a real live person but still be an unknown person.

For example:

```text
Unknown Person
      |
      v
Live Face Detected
      |
      v
Liveness Detected
      |
      v
Face Matching
      |
      v
Does Not Match Registered Face
      |
      v
UNKNOWN FACE