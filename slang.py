#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:39:22 2018

@author: stephanosarampatzes
"""

slang_map = {
         'sorta' : 'sort of',
         'broski' : 'brother',
         'bruh' : 'brother',
         'brav' : 'brother',
         'bro' : 'brother',
         'smh' : 'shaking my head',
         'tbh' : 'to be honest',
         'rn' : 'right now',
         'omg' : 'oh my god',
         'omfg' : 'oh my fucking god',
         'lol' : 'laughing out loud',
         'lel' : 'laughing extremely loud',
         'tgif': 'thank god is friday',
         'sup' : 'what is up',        
         'wassup' : 'what is up',
         'gotchu' : 'got you',
         'dunno' : 'do not know',
         'mula' : 'money',
         'gwap' : 'money',
         'fam' : 'family',
         'innit' : 'is not it' ,
         'coz' : 'because',           
         'gal' : 'girl',
         'gyal' : 'girl',
         'boi' : 'boy',
         'aight' : 'alright',
         'stfu' : 'shut the fuck up',
         'sw' : 'still waiting',
         'bae' : 'before anyone else',
         'bestie' : 'best friend',
         'brb' : 'be right back',
         'afk' : 'away from keyboard',
         'bffe' : 'best friends for ever',
         'bff' : 'best friends forever',
         'thou' : 'thousand',
         'tho' : 'though',
         'wtf' : 'what the fuck',
         'af' : 'as fuck',
         'asf' : 'as fuck',
         'asap' : 'as soon as possible',
         'aka' : 'also known as',
         'lmao' : 'laughing my ass off',
         'rolf' : 'rolling on laughing floor',
         'yolo' : 'you only live once', 
         'yo' : 'your',
         'ya' : 'you',
         'yah' : 'you',
         'kiddo' : 'kid',
         'mofo' : 'mother fucker',      
         'mofos' : 'mother fuckers',    
         'mo' : 'more',
         'den' : 'then',                
         'dem' : 'them',                
         'wit' : 'with',
         'gotta' : 'got to',
         'gtg' : 'got to go',
         'booze' : 'alcohol',           
         'nah' : 'no',
         'dm' : 'direct message',
         'imo' : 'in my opinion',
         'irl' : 'in real life',
         'idk' : 'i do not know',
         'fyah' : 'fire',
         'luv' : 'love',
         'tha' : 'the',
         'idc' : 'i do not care',
         'dat' : 'that',                 
         'xo' : 'hugs and kisses',       
         'omm' : 'oh my momma',
         'ish' : 'it is',
         'oml' : 'oh my lord',
         'icymi' : 'in case you missed it',
         'lil' : 'little',
         'ma' : 'my',
         'mi' : 'my',
         's/o' : 'shout out',
         'g.o.a.t' : 'greatest of all time',
         'goat' : 'greatest of all time',
         'sis' : 'sister',
         'otp' : 'one true pairing',
         'insta' : 'instagram',
         'lk' : 'like back',
         'fb' : 'follow back',
         'fff' : 'follow for follow',
         'rt' : 'retweet',
         'hundo p' : '100%',
         '1hunnit' : '100%',
         '1hunnid' : '100%',
         'hunnid' : '100',
         'otw' : 'on the way',
         'foreal' : 'for real',
         'fo' : 'for',
         'dayum' : 'damn',
         'cuz' : 'cousin',
         'ofc' : 'of course',
         'wanna' : 'want to',
         'gonna' : 'going to',
         'gotta' : 'got to',
         'dawg' : 'close friend',
         'brb' : 'be right back',
         'btw' : 'by the way',
         'dafuq' : 'what the fuck',
         'og' : 'original',
         'u' : 'you',
         'bf' : 'boyfriend',
         'gf' : 'girlfriend',
         'w/' : 'with',
         'atm' : 'at the moment',
         'ats' : 'all the shit',
         'shawty' : 'woman',
         'errbody' : 'everybody',
         'otf' : 'only the family',
         'wth' : 'what the hell',
         'ftw' : 'for the win',
         'wth' : 'what the hell',
         'tbt' : 'throwback Thursday',
         'tba' : 'to be announced',
         'tf' : 'the fuck',
         'hella' : 'really',
         "c'mon" : 'come on',
         'c' : 'see',
         'prolly' : 'probably',
         'fax' : 'facts',
         'deff' : 'definitely',
         'wyd' : 'what you doing',
         'hbu' : 'how about you',
         'lit' : 'amazing',
         'boh' : 'blackout hammered',
         'biz' : 'business',
         'gunna' : 'going to',
         'alr' : 'alright',
         'wat' : 'what',
         'grandpa' : 'grandfather',
         'grandma' : 'grandmother',
         '2day' : 'today',
         '2moro' : 'tomorrow',
         '2morrow' : 'tomorrow',
         'tmrw' : 'tomorrow',
         'pls' : 'please',
         'ppl' : 'people',
         'gimme' : 'give me',
         'shoulda' : 'should',
         'rip' : 'rest in peace',
         'r.i.p' : 'rest in peace',
         'nite' : 'night'
         }

import re

## order dictionary by key
# import collections
# slang_map = collections.OrderedDict(sorted(slang.items()))

def slanger(text, mapping = slang_map):
    # compile dictionary keys separated with '|'
    contr_pattern = re.compile('({})'.format('|'.join(mapping.keys())),
                                      flags = re.IGNORECASE|re.DOTALL) # https://goo.gl/Y5S4a7
    
    def expand_match(contraction):
        
        match = contraction.group(0) # group -> regex
        # match from map
        expanded_contr = mapping.get(match)\
                                if mapping.get(match)\
                                else mapping.get(match.lower())
        # fill out the rest characters
        expanded_contr = expanded_contr[0:]
        
        return(expanded_contr)                        
    # replace slang with meaning
    expanded_text = contr_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return(expanded_text)
    
    
# save as pickle(optional)
"""
import pickle

with open('slanger_def.pickle', 'wb') as out:
    pickle.dump(slanger, out)
 
with open('slanger_def.pickle', 'rb') as fp:
    slanger = pickle.load(fp)
"""
