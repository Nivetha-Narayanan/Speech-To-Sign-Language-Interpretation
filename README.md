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
| 2    | [Proposed Work & Block Diagram](#2-proposed-work) | 
| 3    | [Results and Discussion](#3-results-and-discussion) |
| 4    | [MATLAB Coding](#4-matlab-coding)            | 
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

## 2. PROPOSED WORK & BLOCK DIAGRAM

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

## 4. MATLAB CODING

```matlab
function sign_language_gui
% Create GUI Window
fig = uifigure('Name', 'Sign Language Interpreter', 'Position', [100 100 800 600]);

% Status Label
statusLabel = uilabel(fig, 'Text', 'Click "Start Recording" to begin.', 'Position', [50 540 700 30], 'FontSize', 14);

% Transcription Box
uilabel(fig, 'Text', '📝 Transcribed Text:', 'Position', [50 490 200 20], 'FontSize', 14);
textBox = uitextarea(fig, 'Position', [50 420 700 60], 'Editable', 'off', 'FontSize', 14);

% Video Display
videoDisplay = uiimage(fig, 'Position', [150 20 500 340]);

% Buttons
startBtn = uibutton(fig, 'Text', '🎤 Start Recording', 'Position', [180 380 200 40], 'FontSize', 16, 'ButtonPushedFcn', @(btn,event) startManualRecording(statusLabel));
stopBtn = uibutton(fig, 'Text', '⏹ Stop Recording', 'Position', [420 380 200 40], 'FontSize', 16, 'ButtonPushedFcn', @(btn,event) stopAndProcessRecording(statusLabel, textBox, videoDisplay));
setappdata(fig, 'recorder', []);
end

function startManualRecording(statusLabel)
recObj = audiorecorder(16000, 16, 1);
setappdata(gcf, 'recorder', recObj);
record(recObj);
statusLabel.Text = '🎙️ Recording... Click "Stop Recording" to finish.';
drawnow;
end

function stopAndProcessRecording(statusLabel, textBox, videoDisplay)
recObj = getappdata(gcf, 'recorder');
if isempty(recObj)
    statusLabel.Text = '⚠️ No recording in progress.';
    return;
end
stop(recObj);
statusLabel.Text = '🛑 Recording stopped. Processing...';
drawnow;

audioData = getaudiodata(recObj);
audiowrite('recorded_audio.wav', audioData, 16000);

% Set Python environment
try
    pyenv("Version", "C:\Users\nivet\AppData\Local\Programs\Python\Python311\python.exe");
catch
end

try
    transcribedText = py.transcribe_wav2vec.transcribe_audio('recorded_audio.wav');
    transcribedTextStr = char(transcribedText);
    textBox.Value = transcribedTextStr;
    statusLabel.Text = '✅ Transcription done. Playing videos...';
    drawnow;

    videoFolder = "C:\Users\nivet\OneDrive\ドキュメント\MATLAB\DSP_PRO\videos\INDIAN SIGN LANGUAGE ANIMATED VIDEOS";
    words = split(transcribedTextStr);

    for i = 1:length(words)
        word = strip(lower(words{i}));
        videoPath = fullfile(videoFolder, word + ".mp4");
        if isfile(videoPath)
            statusLabel.Text = "🎬 Playing: " + word;
            drawnow;
            playVideoInGUI(videoPath, videoDisplay);
        else
            statusLabel.Text = "⚠️ No video for: " + word;
            drawnow;
            pause(1);
        end
    end
    statusLabel.Text = '🎉 Done. You can try again.';
catch ME
    statusLabel.Text = ['❌ Error: ', ME.message];
end
end

function playVideoInGUI(videoPath, videoDisplay)
try
    v = VideoReader(videoPath);
    while hasFrame(v)
        frame = readFrame(v);
        videoDisplay.ImageSource = frame;
        drawnow;
        pause(1 / v.FrameRate);
    end
catch
    warning("Could not play video: " + videoPath);
end
end
```
---
## 5.Python Code 
```python

```
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





