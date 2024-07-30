#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Dezember 15, 2023, at 09:23
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from pumping
import random
from psychopy import sound

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'BART_exp'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = "data/sub-{sub}_task-balloonanalogrisktask_run-{run}_beh".format(sub = expInfo['participant'], run = expInfo['session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\PsychoPy\\BART\\BART_exp_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1200], fullscr=True, screen=1,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1,1,1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Start" ---
    text_start = visual.TextStim(win=win, name='text_start',
        text="Ihnen werden nun in 3 Blöcken jeweils 30 Ballone präsentiert. Bitte versuchen Sie, möglichst viel Geld zu sammeln. Zwischen den Blöcken wird es jeweils eine Pause von 5 Minuten geben. \n \n" + "Block " + str(int(expInfo['session'])) + " beginnt, sobald Sie auf \"Start\" drücken."
    
    ,
        font='Open Sans',
        units='norm', pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue_start = visual.TextBox2(
         win, text='Start', placeholder='Type here...', font='Arial',
         pos=(0.9, -0.9),units='norm',     letterHeight=0.05,
         size=(0.2, 0.2), borderWidth=3.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='bottom-right', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='continue_start',
         depth=-1, autoLog=True,
    )
    mouse_start = event.Mouse(win=win)
    x, y = [None, None]
    mouse_start.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "trials" ---
    # Run 'Begin Experiment' code from reward
    bank_value = '{:.2f}'.format(0.00);
    stim_balloon = visual.ImageStim(
        win=win,
        name='stim_balloon', units='norm', 
        image='images/balloon.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=256.0, interpolate=False, depth=-2.0)
    stim_balloon_burst = visual.ImageStim(
        win=win,
        name='stim_balloon_burst', units='norm', 
        image='images/balloon_burst.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    text_ballon_value = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Sans Open',
         pos=(-0.6, -0.90),units='norm',     letterHeight=0.08,
         size=(0.7,0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='top-left',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_ballon_value',
         depth=-4, autoLog=True,
    )
    text_bank = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Sans Open',
         pos=(-0.6, 0.65),units='norm',     letterHeight=0.08,
         size=(0.7,0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='top-left',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_bank',
         depth=-5, autoLog=True,
    )
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    sound_bang = sound.Sound('sounds/bang.wav', secs=0, stereo=True, hamming=True,
        name='sound_bang')
    sound_bang.setVolume(1.0)
    button_balloon = visual.ButtonStim(win, 
        text='Ballon aufpumpen', font='Open Sans',
        pos=(0, -0.78),units='norm',
        letterHeight=0.055,
        size=(0.4, 0.15), borderWidth=0.0,
        fillColor=[0.7255, -0.8431, -0.5294], borderColor='black',
        color='black', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_balloon',
        depth=-10
    )
    button_balloon.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "Feedback" ---
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "End" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='Sie sind am Ende dieses Blocks angelangt. Melden Sie sich bitte beim Versuchsleiter.',
        font='Open Sans',
        units='norm', pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_end = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Start.started', globalClock.getTime())
    continue_start.reset()
    # setup some python lists for storing info about the mouse_start
    mouse_start.x = []
    mouse_start.y = []
    mouse_start.leftButton = []
    mouse_start.midButton = []
    mouse_start.rightButton = []
    mouse_start.time = []
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from code_start
    mouseIsDown = False
    # keep track of which components have finished
    StartComponents = [text_start, continue_start, mouse_start]
    for thisComponent in StartComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start* updates
        
        # if text_start is starting this frame...
        if text_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start.frameNStart = frameN  # exact frame index
            text_start.tStart = t  # local t and not account for scr refresh
            text_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_start.started')
            # update status
            text_start.status = STARTED
            text_start.setAutoDraw(True)
        
        # if text_start is active this frame...
        if text_start.status == STARTED:
            # update params
            pass
        
        # *continue_start* updates
        
        # if continue_start is starting this frame...
        if continue_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_start.frameNStart = frameN  # exact frame index
            continue_start.tStart = t  # local t and not account for scr refresh
            continue_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_start.started')
            # update status
            continue_start.status = STARTED
            continue_start.setAutoDraw(True)
        
        # if continue_start is active this frame...
        if continue_start.status == STARTED:
            # update params
            pass
        # *mouse_start* updates
        
        # if mouse_start is starting this frame...
        if mouse_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_start.frameNStart = frameN  # exact frame index
            mouse_start.tStart = t  # local t and not account for scr refresh
            mouse_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_start.started', t)
            # update status
            mouse_start.status = STARTED
            mouse_start.mouseClock.reset()
            prevButtonState = mouse_start.getPressed()  # if button is down already this ISN'T a new click
        if mouse_start.status == STARTED:  # only update if started and not finished!
            buttons = mouse_start.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    x, y = mouse_start.getPos()
                    mouse_start.x.append(x)
                    mouse_start.y.append(y)
                    buttons = mouse_start.getPressed()
                    mouse_start.leftButton.append(buttons[0])
                    mouse_start.midButton.append(buttons[1])
                    mouse_start.rightButton.append(buttons[2])
                    mouse_start.time.append(mouse_start.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        # Run 'Each Frame' code from code_start
        if mouse_start.getPressed()[0] == 1 and mouseIsDown == False: 
            mouseIsDown = True
            mouseDownContinue = False
            if continue_start.contains(mouse_start):
                mouseDownContinue = True
        
        if mouse.getPressed()[0] == 0 and mouseIsDown:
            mouseIsDown = False
            if continue_start.contains(mouse_start) and mouseDownContinue == True:
                mouseClickedOnContinue = True
                continueRoutine=False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StartComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in StartComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Start.stopped', globalClock.getTime())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_start.x', mouse_start.x)
    thisExp.addData('mouse_start.y', mouse_start.y)
    thisExp.addData('mouse_start.leftButton', mouse_start.leftButton)
    thisExp.addData('mouse_start.midButton', mouse_start.midButton)
    thisExp.addData('mouse_start.rightButton', mouse_start.rightButton)
    thisExp.addData('mouse_start.time', mouse_start.time)
    thisExp.nextEntry()
    # the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pump_loop = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('balloon.xlsx'),
        seed=None, name='pump_loop')
    thisExp.addLoop(pump_loop)  # add the loop to the experiment
    thisPump_loop = pump_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPump_loop.rgb)
    if thisPump_loop != None:
        for paramName in thisPump_loop:
            globals()[paramName] = thisPump_loop[paramName]
    
    for thisPump_loop in pump_loop:
        currentLoop = pump_loop
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPump_loop.rgb)
        if thisPump_loop != None:
            for paramName in thisPump_loop:
                globals()[paramName] = thisPump_loop[paramName]
        
        # --- Prepare to start Routine "trials" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trials.started', globalClock.getTime())
        # Run 'Begin Routine' code from balloon_parameters
        n_pumps = 0;
        popped = False;
        
        # Run 'Begin Routine' code from reward
        balloon_value = 0.00;
        
        text_ballon_value.reset()
        text_bank.reset()
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        gotValidClick = False  # until a click is received
        mouse.mouseClock.reset()
        # Run 'Begin Routine' code from pumping
        mouseIsDown = False
        
        sound_bang.setSound('sounds/bang.wav', secs=0, hamming=True)
        sound_bang.setVolume(1.0, log=False)
        sound_bang.seek(0)
        # reset button_balloon to account for continued clicks & clear times on/off
        button_balloon.reset()
        # keep track of which components have finished
        trialsComponents = [stim_balloon, stim_balloon_burst, text_ballon_value, text_bank, mouse, sound_bang, button_balloon]
        for thisComponent in trialsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from reward
            balloon_value = '{:.2f}'.format(n_pumps * 0.05);
            
            # *stim_balloon* updates
            
            # if stim_balloon is starting this frame...
            if stim_balloon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_balloon.frameNStart = frameN  # exact frame index
                stim_balloon.tStart = t  # local t and not account for scr refresh
                stim_balloon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_balloon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stim_balloon.started')
                # update status
                stim_balloon.status = STARTED
                stim_balloon.setAutoDraw(True)
            
            # if stim_balloon is active this frame...
            if stim_balloon.status == STARTED:
                # update params
                stim_balloon.setSize([0.1 + n_pumps*0.006, 0.3 + n_pumps*0.020], log=False)
            
            # *stim_balloon_burst* updates
            
            # if stim_balloon_burst is starting this frame...
            if stim_balloon_burst.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_balloon_burst.frameNStart = frameN  # exact frame index
                stim_balloon_burst.tStart = t  # local t and not account for scr refresh
                stim_balloon_burst.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_balloon_burst, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stim_balloon_burst.started')
                # update status
                stim_balloon_burst.status = STARTED
                stim_balloon_burst.setAutoDraw(True)
            
            # if stim_balloon_burst is active this frame...
            if stim_balloon_burst.status == STARTED:
                # update params
                stim_balloon_burst.setSize([0.3, 0.9], log=False)
            
            # if stim_balloon_burst is stopping this frame...
            if stim_balloon_burst.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim_balloon_burst.tStartRefresh + 0-frameTolerance:
                    # keep track of stop time/frame for later
                    stim_balloon_burst.tStop = t  # not accounting for scr refresh
                    stim_balloon_burst.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stim_balloon_burst.stopped')
                    # update status
                    stim_balloon_burst.status = FINISHED
                    stim_balloon_burst.setAutoDraw(False)
            
            # *text_ballon_value* updates
            
            # if text_ballon_value is starting this frame...
            if text_ballon_value.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_ballon_value.frameNStart = frameN  # exact frame index
                text_ballon_value.tStart = t  # local t and not account for scr refresh
                text_ballon_value.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_ballon_value, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ballon_value.started')
                # update status
                text_ballon_value.status = STARTED
                text_ballon_value.setAutoDraw(True)
            
            # if text_ballon_value is active this frame...
            if text_ballon_value.status == STARTED:
                # update params
                text_ballon_value.setText("Momentaner Ballonwert:\n" + str(balloon_value) + " CHF", log=False)
            
            # *text_bank* updates
            
            # if text_bank is starting this frame...
            if text_bank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_bank.frameNStart = frameN  # exact frame index
                text_bank.tStart = t  # local t and not account for scr refresh
                text_bank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_bank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_bank.started')
                # update status
                text_bank.status = STARTED
                text_bank.setAutoDraw(True)
            
            # if text_bank is active this frame...
            if text_bank.status == STARTED:
                # update params
                text_bank.setText("Auf der Bank:\n" + str(bank_value) + " CHF", log=False)
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
            # Run 'Each Frame' code from collect
            if mouse.getPressed()[2]:
                continueRoutine=False
            # Run 'Each Frame' code from pumping
            if mouse.getPressed()[0] == 1 and mouseIsDown == False: 
                mouseIsDown = True
                mouseDownButton = False
                if button_balloon.contains(mouse):
                    mouseDownButton = True
                    
            # Check if the mouse is released
            if mouse.getPressed()[0] == 0 and mouseIsDown:
                mouseIsDown = False
                if button_balloon.contains(mouse) and mouseDownButton == True:
                    mouseClickedOnBUtton = True
                    n_pumps = n_pumps + 1;
                
                    if n_pumps > max_pumps:
                        stim_balloon_burst.draw()
                        sound_bang.play()
                        continueRoutine=False
                        popped = True
                    
            
                
            
            
            # if sound_bang is starting this frame...
            if sound_bang.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_bang.frameNStart = frameN  # exact frame index
                sound_bang.tStart = t  # local t and not account for scr refresh
                sound_bang.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_bang.started', t)
                # update status
                sound_bang.status = STARTED
                sound_bang.play()  # start the sound (it finishes automatically)
            
            # if sound_bang is stopping this frame...
            if sound_bang.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_bang.tStartRefresh + 0-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_bang.tStop = t  # not accounting for scr refresh
                    sound_bang.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('sound_bang.stopped', t)
                    # update status
                    sound_bang.status = FINISHED
                    sound_bang.stop()
            # update sound_bang status according to whether it's playing
            if sound_bang.isPlaying:
                sound_bang.status = STARTED
            elif sound_bang.isFinished:
                sound_bang.status = FINISHED
            # *button_balloon* updates
            
            # if button_balloon is starting this frame...
            if button_balloon.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button_balloon.frameNStart = frameN  # exact frame index
                button_balloon.tStart = t  # local t and not account for scr refresh
                button_balloon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_balloon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_balloon.started')
                # update status
                button_balloon.status = STARTED
                button_balloon.setAutoDraw(True)
            
            # if button_balloon is active this frame...
            if button_balloon.status == STARTED:
                # update params
                pass
                # check whether button_balloon has been pressed
                if button_balloon.isClicked:
                    if not button_balloon.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_balloon.timesOn.append(button_balloon.buttonClock.getTime())
                        button_balloon.timesOff.append(button_balloon.buttonClock.getTime())
                    elif len(button_balloon.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_balloon.timesOff[-1] = button_balloon.buttonClock.getTime()
                    # run callback code when button_balloon is clicked
                    pass
            # take note of whether button_balloon was clicked, so that next frame we know if clicks are new
            button_balloon.wasClicked = button_balloon.isClicked and button_balloon.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials" ---
        for thisComponent in trialsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trials.stopped', globalClock.getTime())
        # Run 'End Routine' code from balloon_parameters
        pump_loop.addData('n_pumps', n_pumps)
        pump_loop.addData('earnings', balloon_value)
        pump_loop.addData('popped', popped)
        # Run 'End Routine' code from reward
        if popped == True:
            feedback = f'Ups. Der Ballon ist geplatzt.'
        else:
            feedback = f'Sie haben {balloon_value} CHF auf die Bank übertragen.'
            bank_value = '{:.2f}'.format(float(bank_value) + float(balloon_value))
        
            
        # store data for pump_loop (TrialHandler)
        pump_loop.addData('mouse.x', mouse.x)
        pump_loop.addData('mouse.y', mouse.y)
        pump_loop.addData('mouse.leftButton', mouse.leftButton)
        pump_loop.addData('mouse.midButton', mouse.midButton)
        pump_loop.addData('mouse.rightButton', mouse.rightButton)
        pump_loop.addData('mouse.time', mouse.time)
        pump_loop.addData('button_balloon.numClicks', button_balloon.numClicks)
        if button_balloon.numClicks:
           pump_loop.addData('button_balloon.timesOn', button_balloon.timesOn)
           pump_loop.addData('button_balloon.timesOff', button_balloon.timesOff)
        else:
           pump_loop.addData('button_balloon.timesOn', "")
           pump_loop.addData('button_balloon.timesOff', "")
        # the Routine "trials" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Feedback.started', globalClock.getTime())
        text_feedback.setText(feedback)
        # keep track of which components have finished
        FeedbackComponents = [text_feedback]
        for thisComponent in FeedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_feedback* updates
            
            # if text_feedback is starting this frame...
            if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_feedback.frameNStart = frameN  # exact frame index
                text_feedback.tStart = t  # local t and not account for scr refresh
                text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback.started')
                # update status
                text_feedback.status = STARTED
                text_feedback.setAutoDraw(True)
            
            # if text_feedback is active this frame...
            if text_feedback.status == STARTED:
                # update params
                pass
            
            # if text_feedback is stopping this frame...
            if text_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_feedback.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_feedback.tStop = t  # not accounting for scr refresh
                    text_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_feedback.stopped')
                    # update status
                    text_feedback.status = FINISHED
                    text_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FeedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Feedback" ---
        for thisComponent in FeedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Feedback.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'pump_loop'
    
    
    # --- Prepare to start Routine "End" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('End.started', globalClock.getTime())
    key_end.keys = []
    key_end.rt = []
    _key_end_allKeys = []
    # keep track of which components have finished
    EndComponents = [text_end, key_end]
    for thisComponent in EndComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # *key_end* updates
        waitOnFlip = False
        
        # if key_end is starting this frame...
        if key_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_end.frameNStart = frameN  # exact frame index
            key_end.tStart = t  # local t and not account for scr refresh
            key_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_end.started')
            # update status
            key_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_end.status == STARTED and not waitOnFlip:
            theseKeys = key_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_end_allKeys.extend(theseKeys)
            if len(_key_end_allKeys):
                key_end.keys = _key_end_allKeys[-1].name  # just the last key pressed
                key_end.rt = _key_end_allKeys[-1].rt
                key_end.duration = _key_end_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in EndComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('End.stopped', globalClock.getTime())
    # check responses
    if key_end.keys in ['', [], None]:  # No response was made
        key_end.keys = None
    thisExp.addData('key_end.keys',key_end.keys)
    if key_end.keys != None:  # we had a response
        thisExp.addData('key_end.rt', key_end.rt)
        thisExp.addData('key_end.duration', key_end.duration)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='tab')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
