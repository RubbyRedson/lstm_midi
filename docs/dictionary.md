# About
The way the dictionary is created is to take the integer value from the `Note Value` field of the note on event and cast it to a [ASCII char](http://www.asciitable.com/) 
with the `char(val)` python function. 

It should be noted that **the first 33 chars are pretty dodgy** and should probably be removed to the `Extended ASCII Codes` instead. 

# Note on/off 
This screenshot is taken from [this github](https://github.com/colxi/midi-parser-js/wiki/MIDI-File-Format-Specifications).

<img width="653" alt="skarmavbild 2017-12-19 kl 23 12 40" src="https://user-images.githubusercontent.com/5947764/34181677-730f6f36-e513-11e7-9acd-7c7b56e0fc40.png">

# Example
When you do this you get the following dictionary.

``` 
dictionary = {
0:'',
1:'',
2:'',
3:'',
4:'',
5:'',
6:'',
7:'',
8:'',
9:'	',
10:'',
11:'',
12:'',
13:'',
14:'',
15:'',
16:'',
17:'',
18:'',
19:'',
20:'',
21:'',
22:'',
23:'',
24:'',
25:'',
26:'',
27:'',
28:'',
29:'',
30:'',
31:'',
32:' ',
33:'!',
34:'"',
35:'#',
36:'$',
37:'%',
38:'&',
39:'\'',
40:'(',
41:')',
42:'*',
43:'+',
44:',',
45:'-',
46:'.',
47:'/',
48:'0',
49:'1',
50:'2',
51:'3',
52:'4',
53:'5',
54:'6',
55:'7',
56:'8',
57:'9',
58:':',
59:';',
60:'<',
61:'=',
62:'>',
63:'?',
64:'@',
65:'A',
66:'B',
67:'C',
68:'D',
69:'E',
70:'F',
71:'G',
72:'H',
73:'I',
74:'J',
75:'K',
76:'L',
77:'M',
78:'N',
79:'O',
80:'P',
81:'Q',
82:'R',
83:'S',
84:'T',
85:'U',
86:'V',
87:'W',
88:'X',
89:'Y',
90:'Z',
91:'[',
92:'\\',
93:']',
94:'^',
95:'_',
96:'`',
97:'a',
98:'b',
99:'c',
100:'d',
101:'e',
102:'f',
103:'g',
104:'h',
105:'i',
106:'j',
107:'k',
108:'l',
109:'m',
110:'n',
111:'o',
112:'p',
113:'q',
114:'r',
115:'s',
116:'t',
117:'u',
118:'v',
119:'w',
120:'x',
121:'y',
122:'z',
123:'{',
124:'|',
125:'}',
126:'~'
127:''
}
```

# Code
I used the following script to generate the dictionary, to be copy pasted. (It's not bulletproof and you need to do some manually editing also).

```
print("dictionary = {")
for i in range(0, 127):
	print(str(i) + ":" + "'" + chr(i) +  "',")

print("}")
``` 
