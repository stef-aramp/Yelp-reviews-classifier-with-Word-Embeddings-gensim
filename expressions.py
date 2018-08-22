#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:29:48 2018

@author: stephanosarampatzes
"""

import re

# repeated "ha" in laughs
def laugh(text):
    text = text.lower()
    text = re.sub(r'\S*\whaha\w+', 'haha', text)
    return(text)
            
# "yay" expresson
def yay(text):
    text = text.lower()
    text = re.sub(r'(y{1,})+(a{1,})+(y{1,})', 'yay', text) 
    return(text)
    
# "yum" expressions
    
def yum(text):
    pattern1 = "(y{1,})+(u{1,})+(m{1,})+(y{1,})"
    pattern2 = "(y{1,})+(u{1,})+(m{1,})"
    text = text.lower().split()
    
    for word in text:
        if re.match(pattern1, word):
            text = ' '.join([re.sub(pattern1, 'yum', word) if re.match(pattern1, word) else word for word in text])
            return(text)

        elif re.match(pattern2, word):
            text = ' '.join([re.sub(pattern2, 'yum', word) if re.match(pattern2, word) else word for word in text])
            return(text)
        
        else:
            continue
    return(' '.join([word for word in text]))

# expressions like ahh, aww, aaa or zzz
def repeat1(text):
    text = text.lower()
    text = re.sub(r"\S*aa(h{1,})", 'ahh', text)
    return(text)

# "aaa" in sequence
# run this function after "laugh" and "repeat1" to not have overlaping
def repeat2(text):
    text = text.lower()
    text = re.sub(r"(aa+)", 'a', text)
    return(text)

# "yeah" expressions
def repeat3(text):
    text = text.lower()
    text = re.sub(r"(e+)(a+)(h+)", 'eah', text)
    return(text)

# "sleeping" expressions
def repeat4(text):
    text = text.lower()
    text = re.sub(r"(\bzz+\b)", 'zzz', text)
    return(text)

# compile them all
def expressions(text, laughing=True, yaying=True, yuming=True,
                r1 = True, r2 = True, r3 = True, r4 = True):
    if laughing:
        text=laugh(text)
        
    if yaying:
        text = yay(text)
        
    if yuming:
        text = yum(text)
        
    if r1:
        text = repeat1(text)
        
    if r2:
        text = repeat2(text)
    
    if r3:
        text = repeat3(text)
    
    if r4:
        text = repeat4(text)
        
    return(text)
    
