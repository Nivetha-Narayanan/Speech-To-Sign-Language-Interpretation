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
