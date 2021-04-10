import re
punctuation_tuple = ('!', ',', '?', '.', "'", '"')
text = "hello there,not won't cool!!!tits cool!@ cool@! !cool !cool!!@cool!cool @cool!cool"
text = "does not work,"
#text = "cool@! !cool"

split_text = text.lower().split()

print(split_text)
empty_str = ''

for i, word in enumerate(split_text):

    if any(punctuation in punctuation_tuple for punctuation in word):

        split_word = re.split('([!,?.\'\"])+', word)

        print(split_word)

        for j, s_word in enumerate(split_word):

            if any(punctuation in punctuation_tuple for punctuation in s_word):
                if j > 0:
                    new_word = empty_str.join(split_word[j+1:])
                    print(new_word)
                    split_text.insert(i + 1, new_word)
                    print(split_text)
                    new_word = ''

    # print(split_word)
    # print(empty_str.join(re.split('([!,?.\'\"])+', word)))
    # print(empty_str.join(split_word[i:]))
    #print(word.replace(punctuation) for punctuation in punctuation_tuple)
    # print(re.split('', word))