#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:47:31 2018

@author: stephanosarampatzes
"""

import re

# "\" is to escape RegEx symbols and use them literally

# pattern for basic sad emojis
def sad(text):
    pattern_sad = "(\=\'\()|(\=\()|(\:\'\()|(\:\[)|(\:\/)|(\:\-\/)|(\:\()|(\:\-\()|(\:\_\()|(\:\-\|)"
    text = re.sub(pattern_sad, 'sad', text) 
    return(text)

# pattern for basic happy emojis
def happy(text):
    pattern_happy = "(\:\-\))|(\:\))|(\:\])|(\=\])|(\=\))|(\;\-\))|(\;\))"
    text = re.sub(pattern_happy, 'happy', text)
    return(text)    

# pattern for basic sarcastic emojis
def sarcastic(text):
    pattern_sarcasm = "(\:\.\-\))|(\:\'\D)|(\:\'\))"
    text = re.sub(pattern_sarcasm, 'sarcastic', text)
    return(text)

# all together...
def emoticons(text, sad_emoji = True, happy_emoji = True,
              sarcastic_emoji = True, love_emoji = True):
    
    if sad_emoji:
        text = sad(text)

    if happy_emoji:
        text = happy(text)
    
    if sarcastic_emoji:
        text = sarcastic(text)
        
    if love_emoji:
        text = re.sub("<3", 'love', text)
    
    return(text)

