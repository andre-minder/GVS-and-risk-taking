#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Dezember 15, 2023, at 09:20
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

# Run 'Before Experiment' code from update_choice
import re
# Run 'Before Experiment' code from throw_dice
from psychopy import sound
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'GDT_exp'  # from the Builder filename that created this script
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
    filename = "data/sub-{sub}_task-gameofdicetask_run-{run}_beh".format(sub = expInfo['participant'], run = expInfo['session'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\PsychoPy\\GDT\\GDT_exp_lastrun.py',
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
    win.mouseVisible = False
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
        text="Sie haben nun in 3 Blöcken jeweils die Möglichkeit, 18 Mal hintereinander zu würfeln. Bitte versuchen Sie möglichst viel Geld zu sammeln. Zwischen den Blöcken wird es jeweils eine Pause von 5 Minuten geben. \n \n" + "Block " + str(int(expInfo['session'])) + " beginnt, sobald Sie auf \"Start\" drücken."
    
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
    
    # --- Initialize components for Routine "trial" ---
    box_dice = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         pos=(-0.7, 0.55),units='norm',     letterHeight=0.05,
         size=(0.4, 0.6), borderWidth=3.0,
         color=None, colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='box_dice',
         depth=0, autoLog=True,
    )
    img_dice_throw_trials = visual.ImageStim(
        win=win,
        name='img_dice_throw_trials', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.7, 0.53), size=(0.30, 0.45),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=512.0, interpolate=True, depth=-1.0)
    text_dice_header = visual.TextBox2(
         win, text='Mögliche Kombinationen', placeholder='Type here...', font='Open Sans',
         pos=(-0.3, 0.1),units='norm',     letterHeight=0.08,
         size=(1, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_dice_header',
         depth=-2, autoLog=True,
    )
    text_gain = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=(-0.13, 0.7),units='norm',     letterHeight=0.08,
         size=(1, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_gain',
         depth=-3, autoLog=True,
    )
    value_gain = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=(0.3, 0.7),units='norm',     letterHeight=0.08,
         size=(1, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='value_gain',
         depth=-4, autoLog=True,
    )
    text_bank = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=(-0.17, 0.55),units='norm',     letterHeight=0.1,
         size=(1, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_bank',
         depth=-5, autoLog=True,
    )
    value_bank = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         pos=(0.3, 0.55),units='norm',     letterHeight=0.1,
         size=(1, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='value_bank',
         depth=-6, autoLog=True,
    )
    img_dice_1 = visual.ImageStim(
        win=win,
        name='img_dice_1', units='norm', 
        image='images/dice_1.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.8, -0.1), size=(0.10, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-7.0)
    img_dice_2 = visual.ImageStim(
        win=win,
        name='img_dice_2', units='norm', 
        image='images/dice_2.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.60, -0.1), size=(0.10, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-8.0)
    img_dice_3 = visual.ImageStim(
        win=win,
        name='img_dice_3', units='norm', 
        image='images/dice_3.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.4, -0.1), size=(0.10, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-9.0)
    img_dice_4 = visual.ImageStim(
        win=win,
        name='img_dice_4', units='norm', 
        image='images/dice_4.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.2, -0.1), size=(0.10, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-10.0)
    img_dice_5 = visual.ImageStim(
        win=win,
        name='img_dice_5', units='norm', 
        image='images/dice_5.png', mask=None, anchor='center',
        ori=0.0, pos=(0.0, -0.1), size=(0.10, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-11.0)
    img_dice_6 = visual.ImageStim(
        win=win,
        name='img_dice_6', units='norm', 
        image='images/dice_6.png', mask=None, anchor='center',
        ori=0.0, pos=(0.2, -0.1), size=(0.10, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-12.0)
    img_dice_1_2 = visual.ImageStim(
        win=win,
        name='img_dice_1_2', units='norm', 
        image='images/dice_12.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.7, -0.30), size=(0.15, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-13.0)
    img_dice_3_4 = visual.ImageStim(
        win=win,
        name='img_dice_3_4', units='norm', 
        image='images/dice_34.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.30), size=(0.15, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-14.0)
    img_dice_5_6 = visual.ImageStim(
        win=win,
        name='img_dice_5_6', units='norm', 
        image='images/dice_56.png', mask=None, anchor='center',
        ori=0.0, pos=(0.1, -0.30), size=(0.15, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-15.0)
    img_dice_1_2_3 = visual.ImageStim(
        win=win,
        name='img_dice_1_2_3', units='norm', 
        image='images/dice_123.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, -0.5), size=(0.20, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-16.0)
    img_dice_4_5_6 = visual.ImageStim(
        win=win,
        name='img_dice_4_5_6', units='norm', 
        image='images/dice_456.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.1, -0.5), size=(0.2, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-17.0)
    img_dice_1_2_3_4 = visual.ImageStim(
        win=win,
        name='img_dice_1_2_3_4', units='norm', 
        image='images/dice_1234.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.7, -0.7), size=(0.25, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-18.0)
    img_dice_2_3_4_5 = visual.ImageStim(
        win=win,
        name='img_dice_2_3_4_5', units='norm', 
        image='images/dice_2345.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.7), size=(0.25, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-19.0)
    img_dice_3_4_5_6 = visual.ImageStim(
        win=win,
        name='img_dice_3_4_5_6', units='norm', 
        image='images/dice_3456.png', mask=None, anchor='center',
        ori=0.0, pos=(0.1, -0.7), size=(0.25, 0.15),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=64.0, interpolate=True, depth=-20.0)
    text_value_header = visual.TextBox2(
         win, text='Gewinn / Verlust', placeholder='Type here...', font='Open Sans',
         pos=(0.6, 0.1),units='norm',     letterHeight=0.08,
         size=(0.5, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_value_header',
         depth=-21, autoLog=True,
    )
    text_value_1000 = visual.TextBox2(
         win, text='1000 CHF', placeholder='Type here...', font='Open Sans',
         pos=(0.6, -0.1),units='norm',     letterHeight=0.08,
         size=(0.5, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_value_1000',
         depth=-22, autoLog=True,
    )
    text_value_500 = visual.TextBox2(
         win, text=' 500 CHF', placeholder='Type here...', font='Open Sans',
         pos=(0.6, -0.3),units='norm',     letterHeight=0.08,
         size=(0.5, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_value_500',
         depth=-23, autoLog=True,
    )
    text_value_200 = visual.TextBox2(
         win, text=' 200 CHF', placeholder='Type here...', font='Open Sans',
         pos=(0.6, -0.5),units='norm',     letterHeight=0.08,
         size=(0.5, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_value_200',
         depth=-24, autoLog=True,
    )
    text_value_100 = visual.TextBox2(
         win, text=' 100 CHF', placeholder='Type here...', font='Open Sans',
         pos=(0.6, -0.7),units='norm',     letterHeight=0.08,
         size=(0.5, 0.5), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='text_value_100',
         depth=-25, autoLog=True,
    )
    mouse_task = event.Mouse(win=win)
    x, y = [None, None]
    mouse_task.mouseClock = core.Clock()
    # Run 'Begin Experiment' code from update_choice
    
    
    # Run 'Begin Experiment' code from throw_dice
    bank = 1000
    gain_sound = sound.Sound('A', secs=1.5, stereo=True, hamming=True,
        name='gain_sound')
    gain_sound.setVolume(1.0)
    loss_sound = sound.Sound('A', secs=1.5, stereo=True, hamming=True,
        name='loss_sound')
    loss_sound.setVolume(1.0)
    
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
        
        if mouse_start.getPressed()[0] == 0 and mouseIsDown:
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
    dice_loop = data.TrialHandler(nReps=3.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('dice.xlsx'),
        seed=None, name='dice_loop')
    thisExp.addLoop(dice_loop)  # add the loop to the experiment
    thisDice_loop = dice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisDice_loop.rgb)
    if thisDice_loop != None:
        for paramName in thisDice_loop:
            globals()[paramName] = thisDice_loop[paramName]
    
    for thisDice_loop in dice_loop:
        currentLoop = dice_loop
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
        # abbreviate parameter names if possible (e.g. rgb = thisDice_loop.rgb)
        if thisDice_loop != None:
            for paramName in thisDice_loop:
                globals()[paramName] = thisDice_loop[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        box_dice.reset()
        img_dice_throw_trials.setImage(f"images/{diceImage}")
        text_dice_header.reset()
        text_gain.reset()
        value_gain.reset()
        text_bank.reset()
        value_bank.reset()
        value_bank.setText(str(bank) + " Franken")
        text_value_header.reset()
        text_value_1000.reset()
        text_value_500.reset()
        text_value_200.reset()
        text_value_100.reset()
        # setup some python lists for storing info about the mouse_task
        mouse_task.x = []
        mouse_task.y = []
        mouse_task.leftButton = []
        mouse_task.midButton = []
        mouse_task.rightButton = []
        mouse_task.time = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from update_choice
        dice = [img_dice_1, img_dice_2, img_dice_3,
                img_dice_4, img_dice_5, img_dice_6,
                img_dice_1_2, img_dice_3_4, img_dice_5_6,
                img_dice_1_2_3, img_dice_4_5_6,
                img_dice_1_2_3_4, img_dice_2_3_4_5,
                img_dice_3_4_5_6]
                
        mouseIsDown = False
        selected = False
        
        
        # Run 'Begin Routine' code from throw_dice
        thrown = False;
        choice = ""
        msg = "0 Franken"
        
        gain_sound.setSound('sounds/win.wav', secs=1.5, hamming=True)
        gain_sound.setVolume(1.0, log=False)
        gain_sound.seek(0)
        loss_sound.setSound('sounds/loss.wav', secs=1.5, hamming=True)
        loss_sound.setVolume(1.0, log=False)
        loss_sound.seek(0)
        # keep track of which components have finished
        trialComponents = [box_dice, img_dice_throw_trials, text_dice_header, text_gain, value_gain, text_bank, value_bank, img_dice_1, img_dice_2, img_dice_3, img_dice_4, img_dice_5, img_dice_6, img_dice_1_2, img_dice_3_4, img_dice_5_6, img_dice_1_2_3, img_dice_4_5_6, img_dice_1_2_3_4, img_dice_2_3_4_5, img_dice_3_4_5_6, text_value_header, text_value_1000, text_value_500, text_value_200, text_value_100, mouse_task, gain_sound, loss_sound]
        for thisComponent in trialComponents:
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
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *box_dice* updates
            
            # if box_dice is starting this frame...
            if box_dice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                box_dice.frameNStart = frameN  # exact frame index
                box_dice.tStart = t  # local t and not account for scr refresh
                box_dice.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(box_dice, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'box_dice.started')
                # update status
                box_dice.status = STARTED
                box_dice.setAutoDraw(True)
            
            # if box_dice is active this frame...
            if box_dice.status == STARTED:
                # update params
                pass
            
            # *img_dice_throw_trials* updates
            
            # if img_dice_throw_trials is starting this frame...
            if img_dice_throw_trials.status == NOT_STARTED and thrown:
                # keep track of start time/frame for later
                img_dice_throw_trials.frameNStart = frameN  # exact frame index
                img_dice_throw_trials.tStart = t  # local t and not account for scr refresh
                img_dice_throw_trials.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_throw_trials, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_throw_trials.started')
                # update status
                img_dice_throw_trials.status = STARTED
                img_dice_throw_trials.setAutoDraw(True)
            
            # if img_dice_throw_trials is active this frame...
            if img_dice_throw_trials.status == STARTED:
                # update params
                pass
            
            # if img_dice_throw_trials is stopping this frame...
            if img_dice_throw_trials.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > img_dice_throw_trials.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    img_dice_throw_trials.tStop = t  # not accounting for scr refresh
                    img_dice_throw_trials.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'img_dice_throw_trials.stopped')
                    # update status
                    img_dice_throw_trials.status = FINISHED
                    img_dice_throw_trials.setAutoDraw(False)
            
            # *text_dice_header* updates
            
            # if text_dice_header is starting this frame...
            if text_dice_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_dice_header.frameNStart = frameN  # exact frame index
                text_dice_header.tStart = t  # local t and not account for scr refresh
                text_dice_header.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_dice_header, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_dice_header.started')
                # update status
                text_dice_header.status = STARTED
                text_dice_header.setAutoDraw(True)
            
            # if text_dice_header is active this frame...
            if text_dice_header.status == STARTED:
                # update params
                pass
            
            # *text_gain* updates
            
            # if text_gain is starting this frame...
            if text_gain.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                text_gain.frameNStart = frameN  # exact frame index
                text_gain.tStart = t  # local t and not account for scr refresh
                text_gain.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_gain, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_gain.started')
                # update status
                text_gain.status = STARTED
                text_gain.setAutoDraw(True)
            
            # if text_gain is active this frame...
            if text_gain.status == STARTED:
                # update params
                text_gain.setText('Gewinn / Verlust: ', log=False)
            
            # *value_gain* updates
            
            # if value_gain is starting this frame...
            if value_gain.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                value_gain.frameNStart = frameN  # exact frame index
                value_gain.tStart = t  # local t and not account for scr refresh
                value_gain.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(value_gain, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'value_gain.started')
                # update status
                value_gain.status = STARTED
                value_gain.setAutoDraw(True)
            
            # if value_gain is active this frame...
            if value_gain.status == STARTED:
                # update params
                value_gain.setText(msg
                , log=False)
            
            # *text_bank* updates
            
            # if text_bank is starting this frame...
            if text_bank.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
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
                text_bank.setText('Guthaben:  ', log=False)
            
            # *value_bank* updates
            
            # if value_bank is starting this frame...
            if value_bank.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                value_bank.frameNStart = frameN  # exact frame index
                value_bank.tStart = t  # local t and not account for scr refresh
                value_bank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(value_bank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'value_bank.started')
                # update status
                value_bank.status = STARTED
                value_bank.setAutoDraw(True)
            
            # if value_bank is active this frame...
            if value_bank.status == STARTED:
                # update params
                pass
            
            # *img_dice_1* updates
            
            # if img_dice_1 is starting this frame...
            if img_dice_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_1.frameNStart = frameN  # exact frame index
                img_dice_1.tStart = t  # local t and not account for scr refresh
                img_dice_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_1.started')
                # update status
                img_dice_1.status = STARTED
                img_dice_1.setAutoDraw(True)
            
            # if img_dice_1 is active this frame...
            if img_dice_1.status == STARTED:
                # update params
                pass
            
            # *img_dice_2* updates
            
            # if img_dice_2 is starting this frame...
            if img_dice_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_2.frameNStart = frameN  # exact frame index
                img_dice_2.tStart = t  # local t and not account for scr refresh
                img_dice_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_2.started')
                # update status
                img_dice_2.status = STARTED
                img_dice_2.setAutoDraw(True)
            
            # if img_dice_2 is active this frame...
            if img_dice_2.status == STARTED:
                # update params
                pass
            
            # *img_dice_3* updates
            
            # if img_dice_3 is starting this frame...
            if img_dice_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_3.frameNStart = frameN  # exact frame index
                img_dice_3.tStart = t  # local t and not account for scr refresh
                img_dice_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_3.started')
                # update status
                img_dice_3.status = STARTED
                img_dice_3.setAutoDraw(True)
            
            # if img_dice_3 is active this frame...
            if img_dice_3.status == STARTED:
                # update params
                pass
            
            # *img_dice_4* updates
            
            # if img_dice_4 is starting this frame...
            if img_dice_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_4.frameNStart = frameN  # exact frame index
                img_dice_4.tStart = t  # local t and not account for scr refresh
                img_dice_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_4.started')
                # update status
                img_dice_4.status = STARTED
                img_dice_4.setAutoDraw(True)
            
            # if img_dice_4 is active this frame...
            if img_dice_4.status == STARTED:
                # update params
                pass
            
            # *img_dice_5* updates
            
            # if img_dice_5 is starting this frame...
            if img_dice_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_5.frameNStart = frameN  # exact frame index
                img_dice_5.tStart = t  # local t and not account for scr refresh
                img_dice_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_5.started')
                # update status
                img_dice_5.status = STARTED
                img_dice_5.setAutoDraw(True)
            
            # if img_dice_5 is active this frame...
            if img_dice_5.status == STARTED:
                # update params
                pass
            
            # *img_dice_6* updates
            
            # if img_dice_6 is starting this frame...
            if img_dice_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_6.frameNStart = frameN  # exact frame index
                img_dice_6.tStart = t  # local t and not account for scr refresh
                img_dice_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_6.started')
                # update status
                img_dice_6.status = STARTED
                img_dice_6.setAutoDraw(True)
            
            # if img_dice_6 is active this frame...
            if img_dice_6.status == STARTED:
                # update params
                pass
            
            # *img_dice_1_2* updates
            
            # if img_dice_1_2 is starting this frame...
            if img_dice_1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_1_2.frameNStart = frameN  # exact frame index
                img_dice_1_2.tStart = t  # local t and not account for scr refresh
                img_dice_1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_1_2.started')
                # update status
                img_dice_1_2.status = STARTED
                img_dice_1_2.setAutoDraw(True)
            
            # if img_dice_1_2 is active this frame...
            if img_dice_1_2.status == STARTED:
                # update params
                pass
            
            # *img_dice_3_4* updates
            
            # if img_dice_3_4 is starting this frame...
            if img_dice_3_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_3_4.frameNStart = frameN  # exact frame index
                img_dice_3_4.tStart = t  # local t and not account for scr refresh
                img_dice_3_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_3_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_3_4.started')
                # update status
                img_dice_3_4.status = STARTED
                img_dice_3_4.setAutoDraw(True)
            
            # if img_dice_3_4 is active this frame...
            if img_dice_3_4.status == STARTED:
                # update params
                pass
            
            # *img_dice_5_6* updates
            
            # if img_dice_5_6 is starting this frame...
            if img_dice_5_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_5_6.frameNStart = frameN  # exact frame index
                img_dice_5_6.tStart = t  # local t and not account for scr refresh
                img_dice_5_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_5_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_5_6.started')
                # update status
                img_dice_5_6.status = STARTED
                img_dice_5_6.setAutoDraw(True)
            
            # if img_dice_5_6 is active this frame...
            if img_dice_5_6.status == STARTED:
                # update params
                pass
            
            # *img_dice_1_2_3* updates
            
            # if img_dice_1_2_3 is starting this frame...
            if img_dice_1_2_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_1_2_3.frameNStart = frameN  # exact frame index
                img_dice_1_2_3.tStart = t  # local t and not account for scr refresh
                img_dice_1_2_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_1_2_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_1_2_3.started')
                # update status
                img_dice_1_2_3.status = STARTED
                img_dice_1_2_3.setAutoDraw(True)
            
            # if img_dice_1_2_3 is active this frame...
            if img_dice_1_2_3.status == STARTED:
                # update params
                pass
            
            # *img_dice_4_5_6* updates
            
            # if img_dice_4_5_6 is starting this frame...
            if img_dice_4_5_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_4_5_6.frameNStart = frameN  # exact frame index
                img_dice_4_5_6.tStart = t  # local t and not account for scr refresh
                img_dice_4_5_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_4_5_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_4_5_6.started')
                # update status
                img_dice_4_5_6.status = STARTED
                img_dice_4_5_6.setAutoDraw(True)
            
            # if img_dice_4_5_6 is active this frame...
            if img_dice_4_5_6.status == STARTED:
                # update params
                pass
            
            # *img_dice_1_2_3_4* updates
            
            # if img_dice_1_2_3_4 is starting this frame...
            if img_dice_1_2_3_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_1_2_3_4.frameNStart = frameN  # exact frame index
                img_dice_1_2_3_4.tStart = t  # local t and not account for scr refresh
                img_dice_1_2_3_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_1_2_3_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_1_2_3_4.started')
                # update status
                img_dice_1_2_3_4.status = STARTED
                img_dice_1_2_3_4.setAutoDraw(True)
            
            # if img_dice_1_2_3_4 is active this frame...
            if img_dice_1_2_3_4.status == STARTED:
                # update params
                pass
            
            # *img_dice_2_3_4_5* updates
            
            # if img_dice_2_3_4_5 is starting this frame...
            if img_dice_2_3_4_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_2_3_4_5.frameNStart = frameN  # exact frame index
                img_dice_2_3_4_5.tStart = t  # local t and not account for scr refresh
                img_dice_2_3_4_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_2_3_4_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_2_3_4_5.started')
                # update status
                img_dice_2_3_4_5.status = STARTED
                img_dice_2_3_4_5.setAutoDraw(True)
            
            # if img_dice_2_3_4_5 is active this frame...
            if img_dice_2_3_4_5.status == STARTED:
                # update params
                pass
            
            # *img_dice_3_4_5_6* updates
            
            # if img_dice_3_4_5_6 is starting this frame...
            if img_dice_3_4_5_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                img_dice_3_4_5_6.frameNStart = frameN  # exact frame index
                img_dice_3_4_5_6.tStart = t  # local t and not account for scr refresh
                img_dice_3_4_5_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(img_dice_3_4_5_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'img_dice_3_4_5_6.started')
                # update status
                img_dice_3_4_5_6.status = STARTED
                img_dice_3_4_5_6.setAutoDraw(True)
            
            # if img_dice_3_4_5_6 is active this frame...
            if img_dice_3_4_5_6.status == STARTED:
                # update params
                pass
            
            # *text_value_header* updates
            
            # if text_value_header is starting this frame...
            if text_value_header.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_value_header.frameNStart = frameN  # exact frame index
                text_value_header.tStart = t  # local t and not account for scr refresh
                text_value_header.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_value_header, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_value_header.started')
                # update status
                text_value_header.status = STARTED
                text_value_header.setAutoDraw(True)
            
            # if text_value_header is active this frame...
            if text_value_header.status == STARTED:
                # update params
                pass
            
            # *text_value_1000* updates
            
            # if text_value_1000 is starting this frame...
            if text_value_1000.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_value_1000.frameNStart = frameN  # exact frame index
                text_value_1000.tStart = t  # local t and not account for scr refresh
                text_value_1000.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_value_1000, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_value_1000.started')
                # update status
                text_value_1000.status = STARTED
                text_value_1000.setAutoDraw(True)
            
            # if text_value_1000 is active this frame...
            if text_value_1000.status == STARTED:
                # update params
                pass
            
            # *text_value_500* updates
            
            # if text_value_500 is starting this frame...
            if text_value_500.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_value_500.frameNStart = frameN  # exact frame index
                text_value_500.tStart = t  # local t and not account for scr refresh
                text_value_500.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_value_500, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_value_500.started')
                # update status
                text_value_500.status = STARTED
                text_value_500.setAutoDraw(True)
            
            # if text_value_500 is active this frame...
            if text_value_500.status == STARTED:
                # update params
                pass
            
            # *text_value_200* updates
            
            # if text_value_200 is starting this frame...
            if text_value_200.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_value_200.frameNStart = frameN  # exact frame index
                text_value_200.tStart = t  # local t and not account for scr refresh
                text_value_200.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_value_200, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_value_200.started')
                # update status
                text_value_200.status = STARTED
                text_value_200.setAutoDraw(True)
            
            # if text_value_200 is active this frame...
            if text_value_200.status == STARTED:
                # update params
                pass
            
            # *text_value_100* updates
            
            # if text_value_100 is starting this frame...
            if text_value_100.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_value_100.frameNStart = frameN  # exact frame index
                text_value_100.tStart = t  # local t and not account for scr refresh
                text_value_100.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_value_100, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_value_100.started')
                # update status
                text_value_100.status = STARTED
                text_value_100.setAutoDraw(True)
            
            # if text_value_100 is active this frame...
            if text_value_100.status == STARTED:
                # update params
                pass
            # *mouse_task* updates
            
            # if mouse_task is starting this frame...
            if mouse_task.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_task.frameNStart = frameN  # exact frame index
                mouse_task.tStart = t  # local t and not account for scr refresh
                mouse_task.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_task, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_task.started', t)
                # update status
                mouse_task.status = STARTED
                mouse_task.mouseClock.reset()
                prevButtonState = mouse_task.getPressed()  # if button is down already this ISN'T a new click
            if mouse_task.status == STARTED:  # only update if started and not finished!
                buttons = mouse_task.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        x, y = mouse_task.getPos()
                        mouse_task.x.append(x)
                        mouse_task.y.append(y)
                        buttons = mouse_task.getPressed()
                        mouse_task.leftButton.append(buttons[0])
                        mouse_task.midButton.append(buttons[1])
                        mouse_task.rightButton.append(buttons[2])
                        mouse_task.time.append(mouse_task.mouseClock.getTime())
            # Run 'Each Frame' code from update_choice
            if mouse_task.getPressed()[0] == 1 and mouseIsDown == False:
                mouseIsDown = True
                mouseDownDiceIndex = -1
                for i, die in enumerate(dice):
                    if die.contains(mouse_task):
                        mouseDownDiceIndex = i
                        
            if mouse_task.getPressed()[0] == 0 and mouseIsDown:
                mouseIsDown = False
                for i, die in enumerate(dice):
                    if die.contains(mouse_task) and i == mouseDownDiceIndex:
                        mouseClickedOnDice = True
                        selected = True
                        for j, comb in enumerate(dice):
                            comb.contrast = 1
                            die.contrast = 0.5
                            choice = re.findall(r'\d+', str(die.name))
                            
                   
                            
            
                            
                            
            # Run 'Each Frame' code from throw_dice
            if mouse_task.getPressed()[2] == 1 and selected == True:
                thrown = True;
                selected == False;
            
                choice = list(map(int, choice))
                if thrown == True and d_number in choice:
                    if len(choice) == 4:
                        msg = "+100 Franken"
                        gain = 100
                    elif len(choice) == 3:
                        msg = "+200 Franken"
                        gain = 200
                    elif len(choice) == 2:
                        msg = "+500 Franken"
                        gain = 500
                    elif len(choice) == 1:
                        msg = "+1000 Franken"
                        gain = 1000
                else:
                    if len(choice) == 4:
                        msg = "-100 Franken"
                        gain = -100
                    elif len(choice) == 3:
                        msg = "-200 Franken"
                        gain = -200
                    elif len(choice) == 2:
                        msg = " -500 Franken"
                        gain = -500
                    elif len(choice) == 1:
                        msg = "-1000 Franken"
                        gain = -1000
               
                        
                
            # Run 'Each Frame' code from finish_routine
            if thrown == True:
                
                for j, comb in enumerate(dice):
                    comb.contrast = 1
                
                if img_dice_throw_trials.status == FINISHED:
                    continueRoutine = False
            
            # if gain_sound is starting this frame...
            if gain_sound.status == NOT_STARTED and thrown and gain > 0:
                # keep track of start time/frame for later
                gain_sound.frameNStart = frameN  # exact frame index
                gain_sound.tStart = t  # local t and not account for scr refresh
                gain_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('gain_sound.started', t)
                # update status
                gain_sound.status = STARTED
                gain_sound.play()  # start the sound (it finishes automatically)
            
            # if gain_sound is stopping this frame...
            if gain_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > gain_sound.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    gain_sound.tStop = t  # not accounting for scr refresh
                    gain_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('gain_sound.stopped', t)
                    # update status
                    gain_sound.status = FINISHED
                    gain_sound.stop()
            # update gain_sound status according to whether it's playing
            if gain_sound.isPlaying:
                gain_sound.status = STARTED
            elif gain_sound.isFinished:
                gain_sound.status = FINISHED
            
            # if loss_sound is starting this frame...
            if loss_sound.status == NOT_STARTED and thrown and gain < 100:
                # keep track of start time/frame for later
                loss_sound.frameNStart = frameN  # exact frame index
                loss_sound.tStart = t  # local t and not account for scr refresh
                loss_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('loss_sound.started', t)
                # update status
                loss_sound.status = STARTED
                loss_sound.play()  # start the sound (it finishes automatically)
            
            # if loss_sound is stopping this frame...
            if loss_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > loss_sound.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    loss_sound.tStop = t  # not accounting for scr refresh
                    loss_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('loss_sound.stopped', t)
                    # update status
                    loss_sound.status = FINISHED
                    loss_sound.stop()
            # update loss_sound status according to whether it's playing
            if loss_sound.isPlaying:
                loss_sound.status = STARTED
            elif loss_sound.isFinished:
                loss_sound.status = FINISHED
            
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
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        # store data for dice_loop (TrialHandler)
        dice_loop.addData('mouse_task.x', mouse_task.x)
        dice_loop.addData('mouse_task.y', mouse_task.y)
        dice_loop.addData('mouse_task.leftButton', mouse_task.leftButton)
        dice_loop.addData('mouse_task.midButton', mouse_task.midButton)
        dice_loop.addData('mouse_task.rightButton', mouse_task.rightButton)
        dice_loop.addData('mouse_task.time', mouse_task.time)
        # Run 'End Routine' code from throw_dice
        bank = bank + gain;
        # Run 'End Routine' code from finish_routine
        dice_loop.addData('choice', len(choice))
        dice_loop.addData('gain', gain)
        
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 3.0 repeats of 'dice_loop'
    
    
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
