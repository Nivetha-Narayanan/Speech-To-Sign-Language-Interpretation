# Speech-to-Sign Language Interpretation System for the Aurally Challenged

## ABSTRACT

Communication barriers between hearing and non-hearing individuals remain a significant challenge in inclusive social interaction, particularly in environments where sign language interpreters are not readily available.  
This project proposes an efficient and accessible solution through the development of a speech-to-sign language interpretation system tailored for the aurally challenged.

The system is designed using a hybrid **MATLAB–Python** framework:
- **MATLAB** captures real-time speech input through its audio recording interface.
- **Python** uses the **Wav2Vec2.0** deep learning model from Hugging Face to perform speech-to-text transcription.
- Each recognized word is mapped to an Indian Sign Language (ISL) video from a curated dataset.
- Sign animations are displayed using **OpenCV** inside a **MATLAB GUI** window.

This modular and portable system aids real-time communication for individuals with hearing impairments.

---

## INDEX

| S.No | Content                                      | 
|------|----------------------------------------------|
| 1    | [Introduction](#1-introduction)              | 
| 2    | [Proposed Work](#2-proposed-work) | 
| 3    | [Results and Discussion](#3-results-and-discussion) |
| 4    | [Project Codes](#4-project-codes)            | 
| 5    | [Conclusion](#5-conclusion)                  | 
| 6    | [Applications](#6-applications)              | 
| 7    | [Snapshot of Project Output](#7-snapshot-of-project-output) |

---

## 1. INTRODUCTION

Hearing and speech impairments pose significant challenges to effective communication.  
While sign language bridges this gap for the deaf community, its effectiveness is limited by the lack of understanding among the general public.

This project proposes:
- A **speech-to-sign language system** using deep learning
- Integration of **Wav2Vec2.0** for speech recognition
- Playback of **Indian Sign Language (ISL)** videos in a GUI environment

---

## 2. PROPOSED WORK 

### 🔹 Objective

To design a system that translates spoken English sentences into sign language animations using MATLAB for speech acquisition and Python for transcription and video handling.

### 🔹 System Steps

#### **Step 1: Speech Acquisition**
- MATLAB `audiorecorder` records 5 seconds of user speech at 16 kHz, mono, 16-bit.

#### **Step 2: Audio Storage**
- Audio data saved as `recorded_audio.wav`.

#### **Step 3: Speech-to-Text Transcription**
- Python script (`transcribe_wav2vec.py`) uses Hugging Face’s Wav2Vec2 model to convert speech into text.

#### **Step 4: Text Processing**
- Text is lowercased, capitalized, and split into words.
- Each word is searched in the ISL video dataset.

#### **Step 5: File Matching**
- Videos named as `word.mp4` are matched with transcribed words.
- Missing words generate a warning.

#### **Step 6: Display Output**
- Videos are played using **OpenCV** inside a MATLAB GUI window for an organized and user-friendly experience.

---

## 3. RESULTS AND DISCUSSION

- Accurate speech capture and transcription using Wav2Vec2.
- Smooth playback of matching videos.
- GUI made the interaction visually accessible.
- Words without videos were reported without breaking the flow.
- Room for future upgrades like extended vocabulary and noise handling.

---

## 4. PROJECT CODES
- **MATLAB** [sign_language_gui.m](./sign_language_gui.m)
- **PYTHON** [transcribe_wav2vec.py](./transcribe_wav2vec)


---
## 5. CONCLUSION
   This project shows how deep learning and interactive design can support accessible communication. Real-time translation of speech to sign language offers a powerful tool for inclusion.
### Future enhancements can include:  

🔹Extended vocabulary  
🔹Support for regional signs  
🔹Bidirectional communication     

---
## 6.APPLICATIONS
🔹Inclusive Education – Help hearing-impaired students follow spoken lectures.  
🔹Smart Homes – Translate voice alerts into sign language.  
🔹Customer Service – Enable smooth interaction at reception desks.  
🔹Workplace Accessibility – Follow meetings and announcements.  
🔹Public Announcements – ISL translation at airports, stations.  
🔹Mobile/Web Apps – Make the tool portable.  
🔹Training Centers – Deliver technical content to hearing-impaired learners. 
## 7.Snapshot of Project Output 

Watch a demo of our Sign Language Interpretation system below:  
[📂 Watch Output Video ](https://drive.google.com/drive/folders/11b9SaNsXCi0N_iBXscmAXPOEkcAB8Un5)





