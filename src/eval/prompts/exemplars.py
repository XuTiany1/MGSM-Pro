# prompts/exemplars.py
"""
Few-shot exemplars per language.

To add a language: add one entry to EXEMPLARS keyed by the language code
used in load_dataset (must match the folder name).
"""

EXEMPLARS = {
    "english": [
        ("Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "chinese": [
        ("问题：罗杰有 5 个网球。他又买了 2 罐网球。每罐有 3 个网球。他现在有多少个网球？", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("问题：服务器机房里有九台电脑。从周一到周四，每天又安装了五台电脑。服务器机房里现在有多少台电脑？", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("问题：利亚有 32 块巧克力，她妹妹有 42 块。如果她们吃了 35 块，她们一共还剩下多少块？", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("问题：肖恩有五个玩具。圣诞节他从他爸爸妈妈那里各得到了两个玩具。他现在有多少个玩具？", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("问题：迈克尔有 58 个高尔夫球。周二，他丢失了 23 个高尔夫球。周三，他又丢失了 2 个。周三结束时他有多少个高尔夫球？", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("问题：奥利维亚有 23 美元。她买了五个单价 3 美元的百吉饼。她还剩多少钱？", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("问题：杰森有 20 根棒棒糖。他给了丹尼一些棒棒糖。现在杰森有 12 根棒棒糖。杰森给了丹尼多少根棒棒糖？", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("问题：如果停车场里有 3 辆车，又来了 2 辆车，停车场里有多少辆车？", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "french": [
        ("Question : Roger a 5 balles de tennis. Il achète 2 autres boîtes de balles de tennis en plus. Si chaque boîte contient 3 balles de tennis, combien de balles de tennis a-t-il maintenant ?", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("Question : Il y avait neuf ordinateurs dans la salle des serveurs. Cinq ordinateurs supplémentaires ont été installés chaque jour, du lundi au jeudi. Combien d'ordinateurs y a-t-il maintenant dans la salle des serveurs ?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("Question : Léa avait 32 chocolats et sa sœur en avait 42. Si elles en ont mangé 35, combien de morceaux leur reste-t-il en tout ?", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("Question : Shawn a cinq jouets. Pour Noël, papa et maman lui ont offert deux jouets chacun. Combien de jouets a-t-il maintenant ?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("Question : Michael avait 58 balles de golf. Mardi, il a perdu 23 balles de golf. Mercredi, il en a perdu 2 autres. De combien de balles de golf disposait-il à la fin du mercredi ?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("Question : Elle a acheté cinq bagels à 3 $ chacun. Combien d'argent lui reste-t-il ?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("Question : Jason avait 20 sucettes. Il en a donné à Denny et il ne lui en reste plus que 12 à présent. Combien de sucettes Jason a-t-il donné à Denny ?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("Question : S'il y a 3 voitures dans le parking et que 2 autres arrivent, combien de voitures y a-t-il maintenant dans le parking ?", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "swahili": [
        ("Roger ana mipira 5 ya tenisi. Ananunua mikebe 2 zaidi ya mipira ya tenisi. Kila mkebe una mipira 3 ya tenisi. Ana mipira mingapi ya tenisi kwa sasa?", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("Kuna kompyuta tisa katika chumba cha seva. Kompyuta tano zaidi zilisakinishwa kila siku, kuanzia Jumatatu hadi Alhamisi. Kuna kompyuta ngapi wkenye chumba cha seva kufikia sasa?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("Leah alikuwa na chokoleti 32 na dadake alikuwa na 42. Iwapo walikula 35, wamesalia na chokoleti ngapi kwa jumla?", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("Shawn ana wanasesere watano. Siku ya Krismasi, alipata wanasesere wawili kutoka kwa mamake na babake kila mmoja. Sasa ana wanasesere wangapi kwa sasa?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("Michael alikuwa na mipira 58 ya gofu. Jumanne, alipoteza mipira 23 ya gofu. Jumatano, alipoteza mipira 2 zaidi. Alikuwa na mipira mingapi ya gofu kufikia mwishoni mwa Jumatano?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("Alinunua bageli tano kwa $3 kila moja. Amesalia na pesa ngapi?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("Jason alikuwa na pipi 20. Alimpa Denny pipi nyingine, Sasa Jason amesalia na pipi 12. Jason alimpa Denny pipi ngapi?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("Ikiwa kuna magari 3 katika eneo la maegesho na magari 2 zaidi yameongezeka, kuna magari mangapi kwa jumla katika eneo la maegesho?", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "amharic": [
        ("ጥያቄ: ሮጀር 5 የቴኒስ ኳሶች አሉት። 2 ተጨማሪ የቴኒስ ኳስ ጣሳ ገዛ ። እያንዳንዳቸው ጣሳወች 3 የቴኒስ ኳሶች አሏቸው። አሁን በጠቅላላ ስንት የቴኒስ ኳሶች አሉት?", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("ጥያቄ: በሰርቨር ክፍሉ ውስጥ ዘጠኝ ኮምፒውተሮች ነበሩ። ከሰኞ እስከ ሐሙስ አምስት ተጨማሪ ኮምፒውተሮች በየቀኑ ቢጨመሩ አሁን በሰርቨር ክፍሉ ውስጥ ስንት ኮምፒውተሮች ይኖራሉ?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("ጥያቄ: ሊያ 32 ቸኮሌት ነበራት እህቷ 42 ነበሯት። 35ቱን ቸኮሌት ቢበሉት በጠቅላላ ስንት ቸኮሌት ይቀራቸዋል?", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("ጥያቄ: ሾን አምስት መጫወቻዎች አለው። ለገና፣ ከእናቱ እና ከአባቱ ከእያንዳንዳቸው ሁለት መጫወቻዎችን አግኝቷል። አሁን ስንት መጫወቻዎች አሉት?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("ጥያቄ: ሚካኤል 58 የጎልፍ ኳሶች ነበሩት። ማክሰኞለት 23 የጎልፍ ኳሶችን ጠፉበት። እሮብ ላይ፣ ሌላ 2 ጠፉበት። እሮብ ምሽት ላይ ስንት የጎልፍ ኳሶች ይኖሩታል?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("ጥያቄ: አሊቪያ 23 ዶላር አላት ። 5 ቤግል ለእያንዳቸው 3 ከፍለ ብትገዛ ስንት ብር ይቀራታል ?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("ጥያቄ: ዳሰን 20 ሎሊፖፕ ነበረዉ ። ለዳኒ የሆነ ያህል ሎሊፖፖች ሰጠው። አሁን ደሰን 12 ሎሊፖፕ ኣለዉ። የስን ለ ዳኒ ስንት ሎሊፖፕ ሰጠው ?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("ጥያቄ: በፓርኪንግ ቦታው 3 መኪኖች ቢኖሩ እና ሁለት መኪኖች ደሞ አሁን ደረሱ ። በጠቃላይ ስንት መኪኖች አሉ ?", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "igbo": [
        ("Roger nwere bọọlụ tenisi 5. Ọ zụtara mkpọ bọọlụ tenisi 2 karia. Mkpọ ọbụla nwere bọọlụ tenisi 3. Bọọlụ tenisi ole ka o nwere ugbua?", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("E nwere kọmpụta itoolu n'ime ụlọ server. A na etinye kọmpụta ise kwa ụbọchị, malite na monday ruo thursday. Kọmpụta ole ka enwere n'ime ụlọ server ugbua?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("Leah nweburu chokoleti 32 ma nwanne ya nwaayị nweburu 42. Ọ buru na ha riri 35, iberibe ole ka ha nwere fọdụrụ na ngụkọta?", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("Shawn nwere ihe egwuregwu ise. Maka ekeresimesi, o nwetara ihe egwuregwu abụọ n'otun'otu site na nne ya na nna ya. ihe egwuregwu ole ka o nwere ugbu a?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("Michael nwereri bọọlụ golf 58. Na tuesday, o tufuru bọọlụ golf 23. Na wednesday, abụọ ọzọ furu efu. Bọọlụ golf ole ka o nwere na ngwụcha wednesday?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("Olivia nwere $23. Ọ zụtara bagels ise maka $3 n'otun'otu. Ego ole ka o nwere fọdụrụ?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("Jason nwere lollipop 20. O nyere Denny ụfọdụ lollipop. Ugbua Jason nwere lollipop 12. Lollipop ole ka Jason nyere Denny?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("Ọ bụrụ na enwere ụgbọala 3 n'ebe ọnọdụ ụgbọala ma 2 ọzọ batara, ụgbọala ole nọọ n'ebe ọnọdụ ụgbọala?", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "japanese": [
        ("問題：ロジャーは5個のテニスボールがあります。テニスボールの缶を2つ追加で買います。それぞれの缶には3つのテニスボールが入っています。彼は今いくつのテニスボールがありますか？", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("問題：サーバールームには9台のコンピューターがありました。月曜日から木曜日まで毎日5台のコンピューターをインストールしました。今サーバールームには難題のコンピューターがありますか？", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("問題：リアは32個のチョコレートを持っていました、彼女の妹は42個持っていました。彼女達が35個食べたとしたら、全部で何個残っていますか？", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("問題：ショーンは5個のおもちゃを持っています。クリスマスに、彼は父と母からそれぞれ2つずつおもちゃをもらいました。今彼はいくつのおもちゃがありますか？", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("問題：マイケルは58個のゴルフボールを持っています。火曜日、彼は 23個のゴルフボールを失くしました。水曜日、さらに2個失くしました。水曜日の終わりには、彼は何このゴルフボールを持っていましたか？", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("問題：オリビアには$23あります。彼女はそれぞれ$3のベーグルを5つ買いました。彼女にはいくらのお金が残っていますか？", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("問題：ジェイソンは20個の飴を持っています。彼はデニーに飴をいくつかあげました。今、ジェイソンには12個の飴があります。ジェイソンはデニーにいくつ飴をあげましたか？", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("問題：駐車場に3台の車があり、2台の車が到着するとしたら、駐車場には何台の車がありますか？", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "yoruba": [
        ("Ìbéèrè: Roger ní bọ́ọ́lù aláfajọ̀ 5. Ó ra agolo bọ́ọ́lù aláfajọ̀gbá 2 kún-un. Agolo kọ̀ọ̀kan ní bọ́ọ́lù aláfajọ̀gbá 3. Bọ́ọ́lù aláfajọ̀gbá mélòó ni ó ní báyìí?", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("Ìbéèrè: Kọ̀mpútà mẹ́sàn-án ni ó wà ní yàrá ojú òpó ayélujára. Wọ́n ṣe àtòpọ kọ̀mpútà márùn-ún sí i ní ọjọ́ kọ̀ọ̀kan, láti ọjọ́ Ajé sí ọjọ́ Ìbọ. Kọ̀mpútà mélòó ni ó wà ní yàrá ojú òpó ayélujára náà báyìí?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("Ìbéèrè: Leah ní ṣokolétì 32 bẹ́ẹ̀ sì ni arábìnrin rẹ̀ sì ní 42. Tí wọ́n bá jẹ 35, ẹyọ mélòó ni wọ́n ní nílẹ̀ lápapọ̀?", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("Ìbéèrè: Shawn ní ohun ìṣeré márùn-ún. Fún ọdún kérésìmesì, ó gba ohun ìṣeré méjì ọ̀tọ̀ọ̀tọ̀ ní ọwọ́ ìyá àti bàbá rẹ̀. Ohun ìṣeré mélòó ni ó ní báyìí?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("Ìbéèrè: Michael ní bọ́ọ́lù aláfigigbá sínú ihò. Ní ọjọ́ Ìṣẹ́gùn, ó sọ bọ́ọ́lù aláfigigbá sínú ihò 23 nù. Ní ọjọ́ Ìrú, ó sọ méjì 2 nù sí i. Bọ́ọ́lù aláfigigbá sínú ihò mélòó ni ó kù nígbà tí ọjọ́ Ìrú yóò fi parí?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("Ìbéèrè: Olivia ní $23. Ó ra ẹ̀gba ọwọ́ márùn-ún ní $3 fún ìkọ̀ọ̀kan. Èló ni ó ṣẹ́kù ní ọwọ́ rẹ̀?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("Ìbéèrè: Jason ní pọ́ngilá 20. Ó fún Denny ní pọ́ngilá díẹ̀. Ní báyìí Jason ní pọ́ngilá 12. Pọ́ngilá mélòó ni Jason fún Denny?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("Ìbéèrè: Ní ọkọ̀ ayọ́kẹ́lẹ́ 3 bá wà ní ààyè ìgbọ́kọ̀sí tí ọkọ̀ ayọ́kẹ́lẹ́ 2 míràn tún dé, ọkọ̀ ayọ́kẹ́lẹ́ mélòó ni ó wà ní ààyè ìgbọ́kọ̀sí náà?", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ],
    "twi": [
        ("Rogger wɔ tennis bɔɔlo 5. Watɔ tennis bɔɔlo konko 2 biem. Konko biara hyɛ mu tennis bɔɔlo 3.", "Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."),
        ("Na computa nkron na ɛwɔ server dan no mu. Wɔ hyehyɛɛ computa enum biem kaa ho dabiara, efiri ɛdwoada kɔpem yawoada. Computa dodoɔ sɛn na ɛwɔ server dan no mu?", "Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29."),
        ("Na Leah wɔ chocolates 32 ɛna ne nua baa ɛwɔ 42. Sɛ ɔmo dii 35 a, aka dodoɔ sɛn na ɛmo wɔ.", "Step-by-Step Answer: Leah had 32 chocolates and Leah’s sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39."),
        ("Shawn wɔ agodeɛ num. Christmas mu no, ɔnyaa agodeɛ mmienu firi ne maame ne ne papa hɔ. ɔwɔ agodeɛ ahen seisia?", "Step-by-Step Answer: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9."),
        ("Na Michael wɔ gold bɔɔlo 58. ɛbenada no, ɔyeraa golf bɔɔlo 23. Wukuada no, ɔyeraa 2 biem. ɔwɔ golf bɔɔlo dodoɔ sɛn eberɛ a wukuada kɔɔ nawieyɛ?", "Step-by-Step Answer: Michael started with 58 golf balls and lost 23, so he has 58 - 23 = 35. After he lost 2 more, he has 35 - 2 = 33 balls now. The answer is 33."),
        ("Olivia wɔ $23. ɔtɔɔ bagels num ebiara yɛ $3. Sika dodoɔ sɛn na aka?", "Step-by-Step Answer: 5 bagels for $3 each should cost 5 * 3 = 15 dollars. Olivia had $23 in the beginning, so now she has 23 - 15 = 8 dollars left. The answer is 8."),
        ("Na Jason wɔ tɔfe 20. ɔmaa Denny tɔfe no bi. Afei Jason wɔ tɔfe 12. Tɔfe dodoɔ sɛn na Jason de maa Denny?", "Step-by-Step Answer: Jason started with 20 lollipops, but now he only has 12, so he gave Denny 20 - 12 = 8 lollipops. The answer is 8."),
        ("Sɛ ɛhyɛn mmiɛnsa na ɛwɔ bia a wɔkora hyɛn na ɛhyɛn mmienu duru a, ɛhyɛn dodoɔ sɛn na ɛwɔ ɛhyɛn korabia hɔ.", "Step-by-Step Answer: There are 3 cars in the beginning, 2 more arrive, so now there should be 3 + 2 = 5 cars. The answer is 5."),
    ]
}

DEFAULT_LANGUAGE = "english"


def get_exemplars_prefix(language: str, n_shots: int) -> str:
    if n_shots <= 0:
        return ""

    key = language.strip().lower()
    pool = EXEMPLARS.get(key)
    if pool is None:
        print(f"[exemplars] No exemplars for '{language}', falling back to '{DEFAULT_LANGUAGE}'.")
        pool = EXEMPLARS[DEFAULT_LANGUAGE]

    shots = pool[:n_shots]
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in shots) + "\n\n"