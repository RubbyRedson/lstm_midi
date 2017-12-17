# Docs and Notes

## LSTM Base
Currenly, the code found over at [LSTM by Example](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537) is the foundation of this repository. The sample code is used to train an LSTM on sequences of words, and is hardcoded to take an input of size 3. 

## Improvements for MIDI
### Input Format
A critical consideration is the input format. Music notes can be subdivided into smaller chunks representing the whole, much like how integers can be subdivided into fractions:

__Example:__

The following representations take up the same amount of "time":

♩ = 1/4 = quarter note  
♪♪ = 2 * 1/8 = 2 * eight notes = 2/8 = 1/4  
♬♬ = 4 * 1/16 = 4 * sixteenth notes = 4/16 = 1/4  

Using this, we can subdivide our input MIDI files into musical Planck time by choosing a note length as the smallest unit of time. For example, we can choose to divide all input data into sequences of sixteenth notes ♬ and simply generate multiple sixteenth notes when longer notes are parsed. Victor has made much progress on this front.

#### Implications
##### Discretization and Quantization
Smaller discretizations imply longer input data for our LTSM. A musical piece with a time signature of 4 quarter notes per bay (4/4) and a length of 10 bars can be represented using either 4*10 = 40 quarter notes ♩ or 4 * 4 * 10 = 160 sixteenth notes ♬. 

Similarly, smaller divisions allow the LSTM to learn more expressive music, such as swingy jazz or syncopated rhythms. Larger note divisions would be forced to average quick-moving note sequences, whereas smaller divisions capture them appropriately.

