import re
punctuation_tuple = ('!', ',', '?', '.', "'", '"')
text = "hello there, won't you cool!@ !cool!!@cool!cool"


split_text = text.lower().split()

print(split_text)

for word in split_text:
    print(re.split('([!,?.\'\"])+', word))
    #print(word.replace(punctuation) for punctuation in punctuation_tuple)
    # print(re.split('', word))