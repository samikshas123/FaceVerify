# FaceVerify

## Live Face Verification, Face Recognition and Liveness Detection Web Application

FaceVerify is a biometric web application developed using Python, Streamlit, OpenCV, InsightFace, and ONNX Runtime.

The system is designed to enroll individuals using multiple face samples, generate facial embeddings, and perform live camera-based face verification and recognition.

The application combines:

- User registration
- Secure password storage
- Face enrollment
- Face detection
- Face embedding generation
- Live camera processing
- Liveness detection
- Cosine similarity matching
- Unknown-face rejection
- Candidate management
- Face recognition against stored embeddings
- Evaluation support
- Error handling
- Privacy-aware biometric processing

The main purpose of FaceVerify is to determine whether a face appearing in front of the live camera matches a previously registered face.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Registration Workflow](#registration-workflow)
- [Login and Live Face Verification](#login-and-live-face-verification)
- [Liveness Detection](#liveness-detection)
- [Face Detection](#face-detection)
- [Face Embedding Generation](#face-embedding-generation)
- [Face Matching](#face-matching)
- [Unknown Face Rejection](#unknown-face-rejection)
- [Verification Results](#verification-results)
- [Face Recognition and Identification](#face-recognition-and-identification)
- [Technologies Used](#technologies-used)
- [Machine Learning Models](#machine-learning-models)
- [Face Embeddings](#face-embeddings)
- [Matching Threshold](#matching-threshold)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Python Environment](#python-environment)
- [InsightFace Model Setup](#insightface-model-setup)
- [Running the Application](#running-the-application)
- [Application Pages](#application-pages)
- [Testing](#testing)
- [Expected Results](#expected-results)
- [Evaluation](#evaluation)
- [Failure Cases and Error Handling](#failure-cases-and-error-handling)
- [Security](#security)
- [Privacy and Data Handling](#privacy-and-data-handling)
- [Cost](#cost)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [Assignment Requirement Mapping](#assignment-requirement-mapping)
- [Conclusion](#conclusion)
- [Author](#author)

---

# Project Overview

FaceVerify is a biometric face verification and recognition application.

The system allows a user to create an account and register their face using multiple face samples.

The captured registration samples are processed using InsightFace to generate facial embeddings.

During verification, the application uses a live camera stream to temporarily process camera frames. Verification frames are processed in memory and are not saved as verification photographs.

The live verification process consists of:

1. Detecting the face.
2. Checking the number of faces.
3. Performing liveness detection.
4. Generating a face embedding.
5. Comparing the live embedding with the registered embedding.
6. Returning the appropriate verification result.

The system separates liveness detection from identity verification.

A person being detected as a live person does not automatically mean that the person is the registered user.

For example:

```text
Person appears in front of camera
            |
            v
       Face Detection
            |
            v
      Face Count Check
            |
            v
      Liveness Detection
            |
            v
   Live Face Embedding
            |
            v
     Similarity Matching
            |
       +----+----+
       |         |
       v         v
    Match     No Match
       |         |
       v         v
FACE DETECTED  UNKNOWN FACE