%{

    file_name: NegationExp
    author: Francesco Mantegna
    studentID: 188824
    date: /04/2017
    course: Computational Skills For Cognitive Science
    assignment n?: 2

%}

clear

subjnumber=input('enter subj number :', 's'); % change the prompt in a string and record the subject number from the command window in a variable

%% Setting up PTB's Screen
[win,screenRect]=Screen('OpenWindow',0,[17 127 227],[0 0 800 600]); % creates a window in PTB
Screen('TextFont', win, 'Arial', 1); % defines the font and the style (bold) for the instructions 
Screen('TextSize', win, 30); % defines the text size for the instructions
Screen('BlendFunction', win, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA'); % allows to combine color values of pixels already in the window with new color values from drawing commands

%% Creating audio
samplingRate = 44100; % quality of sound
InitializePsychSound(1); % perform basic initialization of the sound driver
pahandle = PsychPortAudio('Open', [], [], [], samplingRate,1);

%% Setting up Constant and Variables
intertrialint=1.5;
interstringint=1;
interaudioint=.6;

%% Downloading files
for i=1:4
    if i==1 || i==2
        myInstructions(i).type='audio files';
        myInstructions(i).format='*.wav';
    else
        myInstructions(i).type='images';
        myInstructions(i).format='*.jpg';
    end
    switch i
        case 1
             myInstructions(i).folder='Assignment2CSCS/Audio/single/topsound';
        case 2
             myInstructions(i).folder='Assignment2CSCS/Audio/single/bottomsound';
        case 3
             myInstructions(i).folder='Assignment2CSCS/topImage';
        case 4
             myInstructions(i).folder='Assignment2CSCS/bottomimage';
    end
    myInstructions(i).text=['Please upload 20 ',myInstructions(i).type,'\n from the folder: \n',myInstructions(i).folder];
    DrawFormattedText(win,myInstructions(i).text,'center','center',[10 5 5]); % defines the text and the color of instructions
    Screen('Flip',win) ; % display the instructions on the screen
    switch i
        case 1
            audfiles1=uigetfile([myInstructions(i).format],'MultiSelect','on');
        case 2
            audfiles2=uigetfile([myInstructions(i).format],'MultiSelect','on');
        case 3
            visfiles1=uigetfile([myInstructions(i).format],'MultiSelect','on');
        case 4
            visfiles2=uigetfile([myInstructions(i).format],'MultiSelect','on');
    end
end

% %% Draw text instructions about uploading files
% DrawFormattedText(win,'Please upload 20 audio files\n from the folder: \nAudio/single/topsound','center','center',[10 5 5]); % defines the text and the color of instructions
% Screen('Flip',win) ; % display the instructions on the screen
% 
% audfiles1=uigetfile('*.wav','MultiSelect','on');
% %% Draw text instructions about uploading files
% DrawFormattedText(win,'Please upload 20 audio files\n from the folder: \nAudio/single/bottomsound','center','center',[10 5 5]); % defines the text and the color of instructions
% Screen('Flip',win) ; % display the instructions on the screen
% 
% audfiles2=uigetfile('*.wav','MultiSelect','on');
% %% Draw text instructions about uploading files
% DrawFormattedText(win,'Please upload 20 images\n from the folder: \nAssignment2/topImage','center','center',[10 5 5]); % defines the text and the color of instructions
% Screen('Flip',win) ; % display the instructions on the screen
% 
% visfiles1=uigetfile('*.jpg','MultiSelect','on');
% %% Draw text instructions about uploading files
% DrawFormattedText(win,'Please upload 20 images\n from the folder: \nAssignment2/bottomimage','center','center',[10 5 5]); % defines the text and the color of instructions
% Screen('Flip',win) ; % display the instructions on the screen
% 
% visfiles2=uigetfile('*.jpg','MultiSelect','on');

for i=5:6
    switch i
        case 5
            myInstructions(i).folder='/Users/francesco/Documents/MATLAB/Computational  Skills for Cognitive Science/NegationExp/TValue';
            myInstructions(i).format='*.jpg';
        case 6
            myInstructions(i).folder='/Users/francesco/Documents/MATLAB/Computational  Skills for Cognitive Science/NegationExp/Audio/structure';
            myInstructions(i).format='*.wav';
    end
    filePattern = fullfile([myInstructions(i).folder],[myInstructions(i).format]);
    switch i
        case 5
            truefalse = dir(filePattern);
        case 6
            audstruct = dir(filePattern);
    end
end
% folder = '/Users/francesco/Documents/MATLAB/Computational  Skills for Cognitive Science/NegationExp/TValue';
% filePattern = fullfile(folder,'*.jpg');% adds to the coordinates of the folder the extension .jpeg preceded by * standing for each file name
% truefalse = dir(filePattern);
% folder = '/Users/francesco/Documents/MATLAB/Computational  Skills for Cognitive Science/NegationExp/Audio/structure';
% filePattern = fullfile(folder,'*.wav');% adds to the coordinates of the folder the extension .jpeg preceded by * standing for each file name
% audstruct = dir(filePattern);

%% Building structures for stimuli and strings
numvisStimuli=length(visfiles1);
randsixteen=randperm(numvisStimuli,16);
for i=1:numvisStimuli
    myStimuli(i).filenametop=visfiles1{i};
    myStimuli(i).filenamebot=visfiles2{i};
end

numaudStimuli=length(audfiles1);
for i=1:numaudStimuli
    myStimuli(i).filesoundtop=audfiles1{i};
    myStimuli(i).filesoundbot=audfiles1{i};
end

numTrials=0;
for k=randsixteen(1:16)
    numTrials=numTrials+1;
    myTrials(numTrials).audfilename1=audfiles1{k};
    myTrials(numTrials).audfilename2=audfiles2{k};
    myTrials(numTrials).audword1=myTrials(numTrials).audfilename1(1:end-4);
    myTrials(numTrials).audword2=myTrials(numTrials).audfilename2(1:end-4);
end

numTrials=0;
for k=randsixteen(1:16)
    numTrials=numTrials+1;
    myTrials(numTrials).visfilename1=visfiles1{k};
    myTrials(numTrials).visfilename2=visfiles2{k};
    myTrials(numTrials).visword1=myTrials(numTrials).visfilename1(1:end-4);
    myTrials(numTrials).visword2=myTrials(numTrials).visfilename2(1:end-4);
    myStrings(numTrials).taffabove=['There is a/an ' myTrials(numTrials).visword1 ' above a/an ' myTrials(numTrials).visword2] ;
    myStrings(numTrials).tnegabove=['There is not a/an ' myTrials(numTrials).visword2 ' above a/an ' myTrials(numTrials).visword1] ;
    myStrings(numTrials).faffabove=['There is a/an ' myTrials(numTrials).visword2 ' above a/an ' myTrials(numTrials).visword1] ;
    myStrings(numTrials).fnegabove=['There is not a/an ' myTrials(numTrials).visword1 ' above a/an ' myTrials(numTrials).visword2] ;
    myStrings(numTrials).taffbelow=['There is a/an ' myTrials(numTrials).visword2 ' below a/an ' myTrials(numTrials).visword1] ;
    myStrings(numTrials).tnegbelow=['There is not a/an ' myTrials(numTrials).visword1 ' below a/an ' myTrials(numTrials).visword2] ;
    myStrings(numTrials).faffbelow=['There is a/an ' myTrials(numTrials).visword1 ' below a/an ' myTrials(numTrials).visword2] ;
    myStrings(numTrials).fnegbelow=['There is not a/an ' myTrials(numTrials).visword2 ' below a/an ' myTrials(numTrials).visword1] ;
end
names = fieldnames(myStrings);
for i=1:8
    newstr(i).name=char(strcat(names{i},{'mem'}));
    for p=1:16
        myStrings(p).(newstr(i).name)=myStrings(p).(names{i});
        myStrings(p).(newstr(i).name)=strrep(myStrings(p).(names{i}),' is ',' was ');
    end
end
%{
for i=1:8
    for p=1:16
    eval(char([strcat('myStrings(p).', strcat(names{i},{'mem'}))]))=[myStrings(p).(names{i})];
    eval(char([strcat('myStrings(p).', strcat(names{i},{'mem'}))]))=[strrep(myStrings(p).(names{i}),'is','was')];
    end
end
%}

%% Preparing Rectangles for the images
sizeRatio=[2.5 3];
screenCntrX=(screenRect(3))/2;
screenYRatio=[2 1.5];
gridSize=2;
myCounter=0;
for i=1:2
    rectangleSize(i)=round((screenRect(3))/sizeRatio(i));
    screenCntrY=(screenRect(4))/screenYRatio(i);
    switch i
        case 1
            for xcoords=1 % defines x coordinates of the grid
                for ycoords=1:rectangleSize(i):rectangleSize(i)*gridSize % defines y coordinates of the grid
                    myCounter=myCounter+1;
                    imgRects(myCounter,:)=[xcoords ycoords xcoords+rectangleSize(i)/2 ycoords+rectangleSize(i)/2]; %[left top right bottom]
                end
            end
            allMax=max(imgRects);
            maxX=allMax(3);
            maxY=allMax(4);
            gridCntrX=round(maxX/2);
            gridCntrY=round(maxY/2);
            imgRects(:,[1 3])=imgRects(:,[1 3])+screenCntrX-gridCntrX;
            imgRects(:,[2 4])=imgRects(:,[2 4])+screenCntrY-gridCntrY;
        case 2
            rectangleSize(i)=round((screenRect(3))/sizeRatio(i));
            screenCntrY=(screenRect(4))/screenYRatio(i);
            myCounter=0;
            for xcoords=1:rectangleSize(i):rectangleSize(i)*gridSize % defines x coordinates of the grid
                for ycoords=1 % defines y coordinates of the grid
                    myCounter=myCounter+1;
                    tvRects(myCounter,:)=[xcoords ycoords xcoords+rectangleSize(i)/2 ycoords+rectangleSize(i)/2]; %[left top right bottom]
                end
            end
            allMax=max(tvRects);
            maxX=allMax(3);
            maxY=allMax(4);
            gridCntrX=round(maxX/2);
            gridCntrY=round(maxY/2);
            tvRects(:,[1 3])=tvRects(:,[1 3])+screenCntrX-gridCntrX;
            tvRects(:,[2 4])=tvRects(:,[2 4])+screenCntrY-gridCntrY;
    end
end
%strings y axis coordinates
stingcoordy=round((screenRect(4))/4);

% rectangleSize=round((screenRect(3))/2.5);
% gridSize=2;
% myCounter=0;
% for xcoords=1 % defines x coordinates of the grid
%     for ycoords=1:rectangleSize:rectangleSize*gridSize % defines y coordinates of the grid
%         myCounter=myCounter+1;
%         imgRects(myCounter,:)=[xcoords ycoords xcoords+rectangleSize/2 ycoords+rectangleSize/2]; %[left top right bottom]
%     end
% end
% screenCntrX=(screenRect(3))/2;
% screenCntrY=(screenRect(4))/2;
% allMax=max(imgRects);
% maxX=allMax(3);
% maxY=allMax(4);
% gridCntrX=maxX/2;
% gridCntrX=round(gridCntrX);
% gridCntrY=maxY/2;
% gridCntrY=round(gridCntrY);
% imgRects(:,[1 3])=imgRects(:,[1 3])+screenCntrX-gridCntrX;
% imgRects(:,[2 4])=imgRects(:,[2 4])+screenCntrY-gridCntrY;
% 
% %% Preparing Rectangles for the truth value
% tvrectangleSize=round((screenRect(3))/3);
% tvgridSize=2;
% myCounter=0;
% for xcoords=1:tvrectangleSize:tvrectangleSize*tvgridSize % defines x coordinates of the grid
%     for ycoords=1 % defines y coordinates of the grid
%         myCounter=myCounter+1;
%         tvRects(myCounter,:)=[xcoords ycoords xcoords+tvrectangleSize/2 ycoords+tvrectangleSize/2]; %[left top right bottom]
%     end
% end
% screenCntrX=(screenRect(3))/2;
% tvscreenPosY=round((screenRect(4))/1.5);
% allMax=max(tvRects);
% maxX=allMax(3);
% maxY=allMax(4);
% gridCntrX=maxX/2;
% gridCntrX=round(gridCntrX);
% gridCntrY=maxY/2;
% gridCntrY=round(gridCntrY);
% tvRects(:,[1 3])=tvRects(:,[1 3])+screenCntrX-gridCntrX;
% tvRects(:,[2 4])=tvRects(:,[2 4])+tvscreenPosY-gridCntrY;
% 
% %stingcoordx=round((screenRect(3))/7);
% stingcoordy=round((screenRect(4))/4);

%% Implementing the experiment
for condition = 1:2
    if condition ==1
        %% Draw text instructions for condition 1
        DrawFormattedText(win,'Welcome!\n\n You will be shown with 2 images at a time\n For each pair, you will have to judge the truth value\n of a sentence concerning their spatial relationship\n Press any key to continue.','center','center',[10 5 5]); % defines the text and the color of instructions
        Screen('Flip',win) ; % display the instructions on the screen
        KbWait(-1); % wait for any key digited by the user
        KbReleaseWait; % once the key has been digited the wait command is interrupted
    else
        %% Draw text instructions for conditon 2
        DrawFormattedText(win,'Alright!\n\n Now, we are going to do a Memory Test\n You are going to ear some utterances\n describing the relationship\n between the images presented previously\n You are supposed to evaluate them\n either true or false\n Press any key to continue.','center','center',[10 5 5]); % defines the text and the color of instructions
        Screen('Flip',win) ; % display the instructions on the screen
        KbWait(-1); % wait for any key digited by the user
        KbReleaseWait; % once the key has been digited the wait command is interrupted
        Screen('Flip',win) ; % clean the instructions from the screen
    end
    %% Loop over the whole numer of stimuli
    for trial=1:numTrials
        %% Encoding phase
            if condition == 1
                %{
                for i=1:2
                  eval(['Image',num2str(i)])=imread(eval(['myTrials(trial).visfilename',num2str(i)]));
                  eval(['tex',num2str(i)])=Screen('MakeTexture',win, eval(['Image',num2str(i)]));
                  Screen('DrawTexture',win,eval(['tex',num2str(i)]),[],imgRects(1,:));
                  Screen('Flip',win)
                end
                %}
                topImage=imread(myTrials(trial).visfilename1);
                bottomImage=imread(myTrials(trial).visfilename2);
                tex1=Screen('MakeTexture',win, topImage);
                tex2=Screen('MakeTexture',win, bottomImage);
                Screen('DrawTexture',win,tex1,[],imgRects(1,:));
                Screen('DrawTexture',win,tex2,[],imgRects(2,:));
                Screen('Flip',win)
                WaitSecs(intertrialint);
                Screen('Flip',win)

                randseq=randperm(numTrials,16);
                if  randseq(trial) <= 2
                        DrawFormattedText(win, myStrings(trial).taffabove,'center','center');
                        myTrials(trial).strings=myStrings(trial).taffabove;
                    elseif randseq(trial) >= 3 && randseq(trial) <= 4
                        DrawFormattedText(win, myStrings(trial).tnegabove,'center','center');
                        myTrials(trial).strings=myStrings(trial).tnegabove;
                    elseif randseq(trial) >= 5 && randseq(trial) <= 6
                        DrawFormattedText(win, myStrings(trial).faffabove,'center','center');
                        myTrials(trial).strings=myStrings(trial).faffabove;
                    elseif randseq(trial) >= 7 && randseq(trial) <= 8
                        DrawFormattedText(win, myStrings(trial).fnegabove,'center','center');
                        myTrials(trial).strings=myStrings(trial).fnegabove;
                    elseif randseq(trial) >= 9 && randseq(trial) <= 10
                        DrawFormattedText(win, myStrings(trial).taffbelow,'center','center');
                        myTrials(trial).strings=myStrings(trial).taffbelow;
                    elseif randseq(trial) >= 11 && randseq(trial) <= 12
                        DrawFormattedText(win, myStrings(trial).tnegbelow,'center','center');
                        myTrials(trial).strings=myStrings(trial).tnegbelow;
                    elseif randseq(trial) >= 13 && randseq(trial) <= 14
                        DrawFormattedText(win, myStrings(trial).faffbelow,'center','center');
                        myTrials(trial).strings=myStrings(trial).faffbelow;
                    elseif randseq(trial) >= 15 && randseq(trial) <= 16
                        DrawFormattedText(win, myStrings(trial).fnegbelow,'center','center');
                        myTrials(trial).strings=myStrings(trial).fnegbelow;
                end
                    Screen('Flip',win)
                    WaitSecs(interstringint);
          %% Memory Test
          else                
                    randseq=randperm(numTrials,16);
                if  randseq(trial) <= 2
                        myTrials(trial).utterances=myStrings(trial).taffabovemem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(1).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(3).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename1);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename2);
                    end
                    elseif randseq(trial) >= 3 && randseq(trial) <= 4
                        myTrials(trial).utterances=myStrings(trial).tnegabovemem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(2).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(3).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename2);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename1);
                    end
                    elseif randseq(trial) >= 5 && randseq(trial) <= 6
                        myTrials(trial).utterances=myStrings(trial).faffabovemem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(1).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(3).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename2);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename1);
                    end
                    elseif randseq(trial) >= 7 && randseq(trial) <= 8
                        myTrials(trial).utterances=myStrings(trial).fnegabovemem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(2).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(3).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename1);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename2);
                    end
                    elseif randseq(trial) >= 9 && randseq(trial) <= 10
                        myTrials(trial).utterances=myStrings(trial).taffbelowmem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(1).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(4).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename2);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename1);
                    end
                    elseif randseq(trial) >= 11 && randseq(trial) <= 12
                        myTrials(trial).utterances=myStrings(trial).tnegbelowmem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(2).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(4).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename1);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename2);
                    end
                    elseif randseq(trial) >= 13 && randseq(trial) <= 14
                        myTrials(trial).utterances=myStrings(trial).faffbelowmem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(1).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(4).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename1);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename2);
                    end
                    elseif randseq(trial) >= 15 && randseq(trial) <= 16
                        myTrials(trial).utterances=myStrings(trial).fnegbelowmem;
                        DrawFormattedText(win, myTrials(trial).utterances,'center',stingcoordy);
                        [stimulus1,samplingRate1]=audioread(audstruct(2).name);
                        [stimulus2,samplingRate2]=audioread(audstruct(4).name);
                    if strcmp (myTrials(trial).audword1,myTrials(trial).visword1) && strcmp (myTrials(trial).audword2,myTrials(trial).visword2)
                        [stimulus3,samplingRate3]=audioread(myTrials(trial).audfilename2);
                        [stimulus4,samplingRate4]=audioread(myTrials(trial).audfilename1);
                    end
                end
                    %{
                    stimulus1 = resample(stimulus1,samplingRate,samplingRate1);
                    PsychPortAudio('FillBuffer', pahandle,stimulus1');
                    PsychPortAudio('Start', pahandle);
                    WaitSecs(interaudioint);
                    stimulus3 = resample(stimulus3,samplingRate,samplingRate3);
                    PsychPortAudio('FillBuffer', pahandle,stimulus3');
                    PsychPortAudio('Start', pahandle);
                    WaitSecs(interaudioint);
                    stimulus2 = resample(stimulus2,samplingRate,samplingRate2);
                    PsychPortAudio('FillBuffer', pahandle,stimulus2');
                    PsychPortAudio('Start', pahandle);
                    WaitSecs(interaudioint);
                    stimulus4 = resample(stimulus4,samplingRate,samplingRate1);
                    PsychPortAudio('FillBuffer', pahandle,stimulus4');
                    PsychPortAudio('Start', pahandle);
                    WaitSecs(interaudioint);
                    %}
                    stimulus1 = resample(stimulus1,samplingRate,samplingRate1);
                    buffer1=PsychPortAudio('CreateBuffer',pahandle, stimulus1');
                    stimulus3 = resample(stimulus3,samplingRate,samplingRate3);
                    buffer2=PsychPortAudio('CreateBuffer',pahandle, stimulus3');
                    stimulus2 = resample(stimulus2,samplingRate,samplingRate2);
                    buffer3=PsychPortAudio('CreateBuffer',pahandle, stimulus2');
                    stimulus4 = resample(stimulus4,samplingRate,samplingRate4);
                    buffer4=PsychPortAudio('CreateBuffer',pahandle, stimulus4');

                    PsychPortAudio('UseSchedule', pahandle, 1);

                    for i=1:4
                        % Play buffer(i) from startSample 0.0 seconds to endSample 1.0 
                        % seconds. Play one repetition of each soundbuffer...
                        PsychPortAudio('AddToSchedule', pahandle, eval(['buffer',num2str(i)]), 1, 0.0, 1.5, 1);
                    end
                    PsychPortAudio('Start', pahandle, [], 0, 1);
                    stillgoing=1;
                    while stillgoing==1
                        s = PsychPortAudio('GetStatus', pahandle); % query current playback status
                            if s.Active == 0
                            % Schedule finished, engine stopped. Before adding new
                            % slots we first must delete the old ones, ie., reset the
                            % schedule:
                            PsychPortAudio('UseSchedule', pahandle, 2);
                            stillgoing=0;
                            break
                            end
                    end

                    tcheck=imread(truefalse(2).name);
                    fcross=imread(truefalse(1).name);
                    tex3=Screen('MakeTexture',win, tcheck);
                    tex4=Screen('MakeTexture',win, fcross);
                    Screen('DrawTexture',win,tex3,[],tvRects(1,:));
                    Screen('DrawTexture',win,tex4,[],tvRects(2,:));
                    myTrials(trial).resp_onset=Screen('Flip',win);

                    noclickYet = 1; % set the value of noclickYet 'true' again in order to start with another while loop
                        while noclickYet == 1 % while this is true
                            [mouseX,mouseY,buttons] = GetMouse(win); % returns the current (x,y) position of the cursor and the up/down state of the mouse button
                                if buttons(1) % the element is true '1' if the corresponding mouse button is pressed and false '0' otherwise 
                                        if mouseX>tvRects(1,1) && mouseX<tvRects(1,3) && mouseY>tvRects(1,2) && mouseY<tvRects(1,4) % whenever the coordinates of the mouse click are inside the first rectangle of the grid
                                            myTrials(trial).response='t';
                                            myTrials(trial).timeresponse=GetSecs;
                                            Screen('Flip',win);
                                            WaitSecs(intertrialint);
                                            KbReleaseWait % stay here while key down
                                            noclickYet = 0; % set the value of noclickYet 'false' in order to momentarily exit the while loop
                                        break
                                        elseif mouseX>tvRects(2,1) && mouseX<tvRects(2,3) && mouseY>tvRects(2,2) && mouseY<tvRects(2,4) % whenever the coordinates of the mouse click are inside the second rectangle of the grid
                                            myTrials(trial).response='f';
                                            myTrials(trial).timeresponse=GetSecs;
                                            Screen('Flip',win);
                                            WaitSecs(intertrialint);
                                            KbReleaseWait % stay here while key down
                                            noclickYet = 0; % set the value of noclickYet 'false' in order to momentarily exit the while loop
                                        break
                                        end
                                end
                        end

                        myTrials(trial).RTs=(myTrials(trial).timeresponse)-(myTrials(trial).resp_onset);

                    if    strcmp(myTrials(trial).utterances,myStrings(trial).taffabovemem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='taffabove'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    elseif strcmp(myTrials(trial).utterances,myStrings(trial).tnegabovemem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='tnegabove'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    elseif strcmp(myTrials(trial).utterances,myStrings(trial).faffabovemem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='faffabove'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    elseif strcmp(myTrials(trial).utterances,myStrings(trial).fnegabovemem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='fnegabove'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    elseif strcmp(myTrials(trial).utterances,myStrings(trial).taffbelowmem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='taffbelow'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    elseif strcmp(myTrials(trial).utterances,myStrings(trial).taffbelowmem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='taffbelow'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    elseif strcmp(myTrials(trial).utterances,myStrings(trial).tnegbelowmem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='tnegbelow'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                     elseif strcmp(myTrials(trial).utterances,myStrings(trial).faffbelowmem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='faffbelow'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                     elseif strcmp(myTrials(trial).utterances,myStrings(trial).fnegbelowmem) % compares the strings in the column .name with the name of the images which have been shown for encoding and returns a truth-value
                          a='fnegbelow'; % if strcmp returns '1', that is 'true', in the column .encoding will appear the string 'yes' 
                          tvalue=a(1:end-8);
                          if strcmp(myTrials(trial).response,tvalue)
                             myTrials(trial).memaccuracy=1;
                          else
                             myTrials(trial).memaccuracy=0;
                          end
                    end
            end
            pause(.5);
            
    end
    if  condition ==2
        memaccuracy=round(mean([[myTrials.memaccuracy]']),2)*100;
        % memaccuracy=mean([[myTrials.memaccuracy]'])*100;
        meanRTs=mean([myTrials.RTs],1)';
        % meanRTs=mean([myTrials(trial).RTs]');
        DrawFormattedText(win,['Your memory accuracy is: ' num2str(memaccuracy,'%.2f') '\n\n Press "Enter" to go on with the experiment.'],'center','center',[10 5 5]); % defines the text and the color of instructions
        Screen('Flip',win) ; % display the instructions on the screen
        KbWait(-1); % wait for any key digited by the user
        KbReleaseWait; % once the key has been digited the wait command is interrupted
    end
end

sca