from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
import random


compliments = """
1. You have very smooth hair.
2. You deserve a promotion.
3. Good effort!
4. What a fine sweater!
5. I appreciate all of your opinions.
6. I like your style.
7. Your T-shirt smells fresh.
8. I love what you've done with the place.
9. You are like a spring flower; beautiful and vivacious.
10. I am utterly disarmed by your wit.
11. I really enjoy the way you pronounce the word 'ruby'.
12. You complete me.
13. Well done!
14. I like your Facebook status.
15. That looks nice on you.
16. I like those shoes more than mine.
17. Nice motor control!
18. You have a good taste in websites.
19. Your mouse told me that you have very soft hands.
20. You are full of youth.
21. I like your jacket.
22. I like the way you move.
23. You have a good web-surfing stance.
24. You should be a poster child for poster children.
25. Nice manners!
26. I appreciate you more than Santa appreciates chimney grease.
27. I wish I was your mirror.
28. I find you to be a fountain of inspiration.
29. You have perfect bone structure.
30. I disagree with anyone who disagrees with you.
31. Way to go!
32. Have you been working out?
33. With your creative wit, I'm sure you could come up with better compliments than me.
34. I like your socks.
35. You are so charming.
36. Your cooking reminds me of my mother's.
37. You're tremendous!
38. You deserve a compliment!
39. Hello, good looking.
40. Your smile is breath taking.
41. How do you get your hair to look that great?
42. You are quite strapping.
43. I am grateful to be blessed by your presence.
44. Say, aren't you that famous model from TV?
45. Take a break; you've earned it.
46. Your life is so interesting!
47. The sound of your voice sends tingles of joy down my back.
48. I enjoy spending time with you.
49. I would share my dessert with you.
50. You can have the last bite.
51. May I have this dance?
52. I would love to visit you, but I live on the Internet.
53. I love the way you click.
54. You're invited to my birthday party.
55. All of your ideas are brilliant!
56. If I freeze, it's not a computer virus.  I was just stunned by your beauty.
57. You're spontaneous, and I love it!
58. You should try out for everything.
59. You make my data circuits skip a beat.
60. You are the gravy to my mashed potatoes.
61. You get an A+!
62. I'm jealous of the other websites you visit, because I enjoy seeing you so much!
63. I would enjoy a roadtrip with you.
64. If I had to choose between you or Mr. Rogers, it would be you.
65. I like you more than the smell of Grandma's home-made apple pies.
66. You would look good in glasses OR contacts.
67. Let's do this again sometime.
68. You could go longer without a shower than most people.
69. I feel the need to impress you.
70. I would trust you to pick out a pet fish for me.
71. I'm glad we met.
72. Do that again!
73. Will you sign my yearbook?
74. You're so smart!
75. We should start a band.
76. You're cooler than ice-skating Fonzi.
77. I made this website for you.
78. I heard you make really good French Toast.
79. You're cooler than Pirates and Ninjas combined.
80. Oh, I can keep going.
81. I like your pants.
82. You're pretty groovy, dude.
83. When I grow up, I want to be just like you.
84. I told all my friends about how cool you are.
85. You can play any prank, and get away with it.
86. You have ten of the best fingers I have ever seen!
87. I can tell that we are gonna be friends.
88. I just want to gobble you up!
89. You're sweeter than than a bucket of bon-bons!
90. Treat yourself to another compliment!
91. You're pretty high on my list of people with whom I would want to be stranded on an island.
92. You're #1 in my book!
93. Well played.
94. You are well groomed.
95. You could probably lead a rebellion.
96. Is it hot in here or is it just you?
97. <3
98. You are more fun than a Japanese steakhouse.
99. Your voice is more soothing than Morgan Freeman's.
100. I like your sleeves. They're real big.
101. You could be drinking whole milk if you wanted to.
102. You're so beautiful, you make me walk into things when I look at you.
103. I support all of your decisions.
104. You are as fun as a hot tub full of chocolate pudding.
105. I usually don't say this on a first date, but will you marry me?
106. I don't speak much English, but with you all I really need to say is beautiful.
107. Being awesome is hard, but you'll manage.
108. Your skin is radiant.
109. You will still be beautiful when you get older.
110. You could survive a zombie apocalypse.
111. You make me :)
112. I wish I could move your furniture.
113. I think about you while I'm on the toilet.
114. You're so rad.
115. You're more fun than a barrel of monkeys.
116. You're nicer than a day on the beach.
117. Your glass is the fullest.
118. I find you very relevant.
119. You look so perfect.
120. The only difference between exceptional and amazing is you.
121. Last night I had the hiccups, and the only thing that comforted me to sleep was repeating your name over and over.
122. I like your pearly whites!
123. Your eyebrows really make your pretty eyes stand out.
124. Shall I compare thee to a summer's day?  Thou art more lovely and more temperate.
125. I love you more than bacon!
126. You intrigue me.
127. You make me think of beautiful things, like strawberries.
128. I would share my fruit Gushers with you.
129. You're more aesthetically pleasant to look at than that one green color on this website.
130. Even though this goes against everything I know, I think I'm in love with you.
131. You're more fun than bubble wrap.
132. Your smile could illuminate the depths of the ocean.
133. You make babies smile.
134. You make the gloomy days a little less gloomy.
135. You are warmer than a Snuggie.
136. You make me feel like I am on top of the world.
137. Playing video games with you would be fun.
138. Let's never stop hanging out.
139. You're more cuddly than the Downy Bear.
140. I would do your taxes any day.
141. You are a bucket of awesome.
142. You are the star of my daydreams.
143. If you really wanted to, you could probably get a bird to land on your shoulder and hang out with you.
144. My mom always asks me why I can't be more like you.
145. You look great in this or any other light.
146. You listen to the coolest music.
147. You and Chuck Norris are on equal levels.
148. Your body fat percentage is perfectly suited for your height.
149. I am having trouble coming up with a compliment worthy enough for you.
150. If we were playing kickball, I'd pick you first.
151. You're cooler than ice on the rocks.
152. You're the bee's knees.
153. I wish I could choose your handwriting as a font.
154. You definitely know the difference between your and you're.
155. You have good taste.
156. I named all my appliances after you.
157. Your mind is a maze of amazing!
158. Don't worry about procrastinating on your studies, I know you'll do great!
159. I like your style!
160. Hi, I'd like to know why you're so beautiful.
161. If I could count the seconds I think about you, I will die in the process!
162. If you were in a chemistry class with me, it would be 10x less boring.
163. If you broke your arm, I would carry your books for you.
164. I love the way your eyes crinkle at the corners when you smile.
165. You make me want to be the person I am capable of being.
166. You're a skilled driver.
167. You are the rare catalyst to my volatile compound.
168. You're a tall glass of water!
169. I'd like to kiss you. Often.
170. You are the wind beneath my wings.
171. Looking at you makes my foot cramps go away instantaneously.
172. I like your face.
173. You are a champ!
174. You are infatuating.
175. Even my cat likes you.
176. There isn't a thing about you that I don't like.
177. You're so cool, that on a scale of from 1-10, you're elevendyseven.
178. OH, you OWN that ponytail.
179. Your shoes are untied. But for you, it's cool.
180. You have the best laugh ever.
181. We would enjoy a cookout with you!
182. Your name is fun to say.
183. I love you more than a drunk college student loves tacos.
184. My camera isn't worthy to take your picture.
185. You are the sugar on my rice krispies.
186. Nice belt!
187. I could hang out with you for a solid year and never get tired of you.
188. You're real happening in a far out way.
189. I bet you could take a punch from Mike Tyson.
190. Your feet are perfect size!
191. You have very nice teeth.
192. Can you teach me how to be as awesome as you?
193. Our awkward silences aren't even awkward.
194. Don't worry. You'll do great.
195. I enjoy you more than a good sneeze. A GOOD one.
196. You could invent words and people would use them.
197. You have powerful sweaters.
198. If you were around, I would enjoy doing my taxes.
199. You look like you like to rock.
200. You are better than unicorns and sparkles combined!
201. You are the watermelon in my fruit salad. Yum!
202. I dig you.
203. You look better whether the lights are on or off.
204. I am enchanted to meet you.
205. I bet even your farts smell good.
206. I would trust my children with you.
207. You make me forget what I was going to...
208. Your smile makes me smile.
209. I'd wake up for an 8 a.m. class just so I could sit next to you.
210. You have the moves like Jagger.
211. You're so hot that you denature my proteins.
212. All I want for Christmas is you!
213. You are the world's greatest hugger.
214. You have a perfectly symmetrical face.
215. If you were in a movie you wouldn't get killed off.
216. Your red ruby lips and wiggly hips make me do flips!
217. I definitely wouldn't kick you out of bed.
218. They should name an ice cream flavor after you.
219. You're the salsa to my tortilla chips. You spice up my life!
220. You smell nice.
221. You don't need make-up, make-up needs you.
222. Me without you is like a nerd without braces, a shoe with out laces, asentencewithoutspaces.
223. Just knowing someone as cool as you will read this makes me smile.
224. I would volunteer to take your place in the Hunger Games.
225. If I had a nickel for everytime you did something stupid, I'd be broke!
226. I'd let you steal the white part of my Oreo.
227. I'd trust you to perform open heart surgery on me... blindfolded!
228. Nice butt! - According to your toilet seat
229. Perfume strives to smell like you.
230. I've had the time of my life, and I owe it all to you!
231. The Force is strong with you.
232. I like the way your nostrils are placed on your nose.
233. I would hold the elevator doors open for you if they were closing.
234. Your every thought and motion contributes to the beauty of the universe.
235. You make me want to frolic in a field.
""".split("\n")


moar ="""There is a true and sincere friendship between you and your friends.
You find beauty in ordinary things, do not lose this ability.
Ideas are like children; there are none so wonderful as your own.
It takes more than good memory to have good memories.
A thrilling time is in your immediate future.
Your blessing is no more than being safe and sound for the whole lifetime.
Plan for many pleasures ahead.
The joyfulness of a man prolongeth his days.
Your everlasting patience will be rewarded sooner or later.
Make two grins grow where there was only a grouch before.
Something you lost will soon turn up.
Your heart is pure, and your mind clear, and your soul devout.
Excitement and intrigue follow you closely wherever you go!
A pleasant surprise is in store for you.
May life throw you a pleasant curve.
As the purse is emptied the heart is filled.
Be mischievous and you will not be lonesome.
You have a deep appreciation of the arts and music.
Your flair for the creative takes an important place in your life.
Your artistic talents win the approval and applause of others.
Pray for what you want, but work for the things you need.
Your many hidden talents will become obvious to those around you.
Don't forget, you are always on our minds.
Your greatest fortune is the large number of friends you have.
A firm friendship will prove the foundation on your success in life.
Don't ask, don't say. Everything lies in silence.
Look for new outlets for your own creative abilities.
Be prepared to accept a wondrous opportunity in the days ahead!
Fame, riches and romance are yours for the asking.
Good luck is the result of good planning.
Good things are being said about you.
Smiling often can make you look and feel younger.
Someone is speaking well of you.
The time is right to make new friends.
You will inherit some money or a small piece of land.
Your life will be happy and peaceful.
A friend is a present you give yourself.
A member of your family will soon do something that will make you proud.
A quiet evening with friends is the best tonic for a long day.
A single kind word will keep one warm for years.
Anger begins with folly, and ends with regret.
Generosity and perfection are your everlasting goals.
Happy news is on its way to you.
He who laughs at himself never runs out of things to laugh at.
If your desires are not extravagant they will be granted.
Let there be magic in your smile and firmness in your handshake.
If you want the rainbow, you must to put up with the rain. D. Parton
Nature, time and patience are the three best physicians.
Strong and bitter words indicate a weak cause.
The beginning of wisdom is to desire it.
You will have a very pleasant experience.
You will inherit some money or a small piece of land.
You will live a long, happy life.
You will spend old age in comfort and material wealth.
You will step on the soil of many countries.
You will take a chance in something in the near future.
You will witness a special ceremony.
Your everlasting patience will be rewarded sooner or later.
Your great attention to detail is both a blessing and a curse.
Your heart is a place to draw true happiness.
Your ability to juggle many tasks will take you far.
A friend asks only for your time, not your money.
You will be invited to an exciting event.""".split("\n")


def get_fortune(compliments, moar):
    complimentz = compliments + moar
    fortune_cookies = [x.split(". ")[-1] for x in complimentz]
    cookiez = ".    " + random.choice(fortune_cookies)
    return cookiez


def gen_filename():
    spaces = str(datetime.datetime.now()).replace(" ", "_")
    colons = spaces.replace(":", "_")
    dashes = colons.replace("-", "_")
    period = dashes.replace(".", "_")
    return period + ".jpg"


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(file_reader, channels = 3,
                                         name='png_reader')
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                    name='gif_reader'))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
      image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                          name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label
